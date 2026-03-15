import asyncio
import nest_asyncio
import re

# --- FIX: Asyncio Loop for Streamlit (REQUIRED FOR IBKR) ---
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

nest_asyncio.apply()

import os
import io
import requests
from fpdf import FPDF
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
import plotly.io as pio
import streamlit as st

from data.finance_ingestor import FinanceEngine
from notifications.daily_bot import get_ai_analysis
from ml.ml_engine import MLEngine
from ml.backtester import Backtester
from data.portfolio_optimizer import PortfolioOptimizer
from data.asset_screener import AssetScreener
import io

load_dotenv()

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Data Investment Hub", layout="wide")


# --- FUNCIONES DE UTILIDAD ---

def send_telegram_alert(tickers, vol_data, price_data=None):
    """Envía un resumen completo de la cartera por Telegram, incluyendo datos cuantitativos si están disponibles."""
    lines = ["📊 *Resumen de Cartera — Data Investment Hub*\n"]

    # --- Bloque 1: Estado actual de precios ---
    if price_data is not None:
        lines.append("💹 *Precios del Día:*")
        for t in tickers:
            try:
                last_p = price_data[t].iloc[-1]
                prev_p = price_data[t].iloc[-2]
                chg = ((last_p / prev_p) - 1) * 100
                emoji = "🟢" if chg >= 0 else "🔴"
                lines.append(f"  {emoji} *{t}*: `{last_p:.2f}` ({chg:+.2f}%)")
            except Exception:
                lines.append(f"  ⚠️ Sin datos para {t}")
        lines.append("")

    # --- Bloque 2: Volatilidades ---
    top_vol = vol_data.idxmax()
    lines.append("🌊 *Volatilidades Anualizadas:*")
    for t, v in vol_data.sort_values(ascending=False).items():
        bar = "▓" * int(v * 10)
        lines.append(f"  • {t}: `{v:.2%}` {bar}")
    lines.append(f"\n⚠️ Activo más volátil: *{top_vol}*\n")

    # --- Bloque 3: Markowitz (si se ejecutó en la sesión actual) ---
    markowitz_res = st.session_state.get("markowitz_res")
    if markowitz_res:
        opt = markowitz_res.get('opt_results', {})
        lines.append("📐 *Distribución Óptima (Markowitz):*")
        for ticker, w in opt.get('weights', {}).items():
            bar = "█" * int(w * 20)
            lines.append(f"  • *{ticker}*: `{w:.1%}` {bar}")
        sharpe = opt.get('sharpe_ratio', 0)
        exp_ret = opt.get('expected_return', 0) * 100  # Convertir de decimal a %
        exp_vol = opt.get('expected_volatility', 0) * 100  # Convertir de decimal a %
        lines.append(f"  Sharpe: `{sharpe:.2f}` | Retorno: `{exp_ret:.2f}%` | Riesgo: `{exp_vol:.2f}%`\n")

    # --- Bloque 4: Modelo ML (si se entrenó en la sesión actual) ---
    bt_results = st.session_state.get("bt_results")
    if bt_results:
        acc = bt_results['report'].get('WFO_Accuracy', 'N/A')
        acc_str = f"{acc:.2f}%" if isinstance(acc, float) else str(acc)
        metrics = bt_results.get('metrics', {})
        total_ret = metrics.get('Retorno Total (%)', metrics.get('Total Return [%]', 'N/A'))
        bench_ret = metrics.get('Retorno Anualizado (%)', metrics.get('Benchmark Return [%]', 'N/A'))
        lines.append(f"🤖 *Modelo ML ({bt_results['model_type']}):*")
        lines.append(f"  Precisión WFO: `{acc_str}`")
        if isinstance(total_ret, float) and isinstance(bench_ret, float):
            lines.append(f"  Retorno Total IA: `{total_ret:.2f}%` | Retorno Anualizado: `{bench_ret:.2f}%`\n")
        else:
            lines.append(f"  Retorno Total IA: `{total_ret}` | Retorno Anualizado: `{bench_ret}`\n")

    message = "\n".join(lines)

    url = f"https://api.telegram.org/bot{os.getenv('BOT_TOKEN')}/sendMessage"
    data = {"chat_id": os.getenv('BOT_ID'), "text": message, "parse_mode": "Markdown"}
    try:
        # A3: timeout explícito y verificación de respuesta HTTP
        resp = requests.post(url, data=data, timeout=5)
        return resp.ok
    except Exception as e:
        # A2: except tipado con log del error
        import logging
        logging.getLogger(__name__).error(f"Error Telegram send_telegram_alert: {e}")
        return False


def create_full_pdf(data, vol, corr, tickers, fig_main, fig_vol, fig_corr, ai_reports, bt_results=None, markowitz_res=None):
    """Genera un PDF completo con texto, imágenes de gráficos y tablas"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Portada
    pdf.add_page()
    pdf.set_font("helvetica", "B", 20)
    pdf.cell(0, 20, "Informe Integral de Inversión", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, f"Generado el: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(10)

    # 1. Sección de Métricas (KPIs)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "1. Resumen de Mercado", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    for t in tickers:
        last_p = data[t].iloc[-1]
        pdf.cell(0, 8, f"- {t}: {last_p:.2f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # 2. Gráfico de Rendimiento
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "2. Evolución del Rendimiento (Base 100)", new_x="LMARGIN", new_y="NEXT")
    img_main = pio.to_image(fig_main, format="png", width=800, height=400)
    pdf.image(io.BytesIO(img_main), x=10, w=190)
    pdf.ln(5)

    # 3. Volatilidad y Correlación
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "3. Análisis de Riesgo y Correlación", new_x="LMARGIN", new_y="NEXT")

    # Imagen Volatilidad
    img_vol = pio.to_image(fig_vol, format="png", width=600, height=300)
    pdf.image(io.BytesIO(img_vol), x=10, w=100)

    # Imagen Correlación
    img_corr = pio.to_image(fig_corr, format="png", width=600, height=450)
    pdf.image(io.BytesIO(img_corr), x=110, y=30, w=90)  # Al lado de la volatilidad

    pdf.ln(80)  # Espacio para las imágenes anteriores

    # 4. Tabla de Datos (Últimos 10 días)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "4. Datos Históricos Recientes", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 8)

    # Dibujar una tabla simple
    cols = ["Date"] + tickers
    temp_df = data.tail(10).reset_index()
    temp_df["Date"] = temp_df["Date"].dt.strftime('%Y-%m-%d')

    # Cabecera tabla
    pdf.set_fill_color(200, 220, 255)
    for col in cols:
        pdf.cell(30, 8, col, border=1, fill=True)
    pdf.ln()

    # Filas tabla
    for i, row in temp_df.iterrows():
        for col in cols:
            pdf.cell(30, 7, str(row[col])[:8], border=1)
        pdf.ln()

    # --- NUEVA SECCIÓN: INTELIGENCIA ARTIFICIAL ---
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(30, 70, 150)  # Un azul elegante
    pdf.cell(0, 15, "Dictamen de Inteligencia Artificial (IA)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)  # Volver al negro

    for ticker in tickers:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, f"Análisis Detallado de {ticker}:", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 10)
        # El texto de la IA puede ser largo, usamos multi_cell
        analysis_text = ai_reports.get(ticker, "No se generó análisis para este activo.")
        pdf.multi_cell(0, 7, analysis_text)
        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())  # Línea divisoria suave
        pdf.ln(5)

    # --- FINAL SECCIÓN: CUANTITATIVO Y MACHINE LEARNING ---
    if markowitz_res or bt_results:
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 15, "Insights Cuantitativos y Machine Learning", new_x="LMARGIN", new_y="NEXT")

        if markowitz_res:
            opt = markowitz_res.get('opt_results', {})
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 10, "Portafolio Óptimo (Markowitz Sharpe Max):", new_x="LMARGIN", new_y="NEXT")
            
            # Gráfico de pesos
            img_pie = pio.to_image(markowitz_res['fig'], format="png", width=600, height=400)
            pdf.image(io.BytesIO(img_pie), x=10, y=pdf.get_y(), w=120)
            
            pdf.set_y(pdf.get_y() + 85)
            pdf.set_font("helvetica", "", 10)
            exp_ret = opt.get('expected_return', 0) * 100  # Decimal a %
            exp_vol = opt.get('expected_volatility', 0) * 100  # Decimal a %
            pdf.cell(0, 8, f"- Retorno Esperado: {exp_ret:.2f}%", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 8, f"- Volatilidad (Riesgo): {exp_vol:.2f}%", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 8, f"- Sharpe Ratio: {opt.get('sharpe_ratio', 0):.2f}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(10)

        if bt_results:
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 10, f"Predicciones IA - Modelo {bt_results['model_type']}:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 10)
            
            acc = bt_results['report'].get('WFO_Accuracy', 'N/A')
            acc_str = f"{acc:.2f}%" if isinstance(acc, float) else str(acc)
            pdf.cell(0, 8, f"- Precisión WFO (Walk-Forward): {acc_str}", new_x="LMARGIN", new_y="NEXT")
            
            metrics = bt_results.get('metrics', {})
            total_ret = metrics.get('Retorno Total (%)', metrics.get('Total Return [%]', 'N/A'))
            anual_ret = metrics.get('Retorno Anualizado (%)', metrics.get('Ann. Return [%]', 'N/A'))
            drawdown  = metrics.get('Max Drawdown (%)', metrics.get('Max Drawdown [%]', 'N/A'))
            win_rate  = metrics.get('Win Rate (%)', metrics.get('Win Rate [%]', 'N/A'))
            
            pdf.cell(0, 8, f"- Retorno Total IA: {total_ret:.2f}%" if isinstance(total_ret, float) else f"- Retorno Total IA: {total_ret}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 8, f"- Retorno Anualizado IA: {anual_ret:.2f}%" if isinstance(anual_ret, float) else f"- Retorno Anualizado IA: {anual_ret}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 8, f"- Max Drawdown: {drawdown:.2f}%" if isinstance(drawdown, float) else f"- Max Drawdown: {drawdown}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 8, f"- Win Rate: {win_rate:.2f}%" if isinstance(win_rate, float) else f"- Win Rate: {win_rate}", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


# --- INTERFAZ STREAMLIT ---

st.title("📊 Data Investment Engine v1.0")
st.markdown("Plataforma de ingeniería de datos financieros")

# Sidebar
st.sidebar.header("Configuración")
# Mantener tickers base en sesión para poder agregar dinámicamente
if 'base_tickers' not in st.session_state:
    st.session_state.base_tickers = "AAPL, BTC-USD, GC=F, MSFT, IWDA.AS"

# A1: Función de validación de tickers
def _sanitize_ticker(t: str):
    """Solo acepta caracteres válidos para un ticker (letras, dígitos, guión, punto, igual)."""
    t = t.strip().upper()
    if re.match(r'^[A-Z0-9\-\.\=]{1,20}$', t):
        return t
    return None

tickers_input = st.sidebar.text_input("Lista de Tickers", st.session_state.base_tickers)
st.session_state.base_tickers = tickers_input  # Actualiza si el user escribe manual

periodo = st.sidebar.selectbox("Rango Temporal", ["1mo", "6mo", "1y", "2y", "5y"], index=2)
# A1: Filtrar tickers inválidos antes de cualquier operación
tickers_raw = [t.strip() for t in tickers_input.split(",")]
tickers = [r for r in (_sanitize_ticker(t) for t in tickers_raw) if r]
if not tickers:
    st.error("❌ Ninguno de los tickers introducidos es válido. Usa formato como: AAPL, BTC-USD, GC=F")
    st.stop()


# L2: Cache de datos de mercado para evitar re-descarga en cada interacción del usuario
@st.cache_data(ttl=3600)
def _load_market_data(tickers_tuple: tuple, period: str):
    """Descarga y cachea datos de mercado durante 1 hora."""
    eng = FinanceEngine(list(tickers_tuple))
    data_closes = eng.extract_data(period=period)
    data_full = eng.extract_full_data(period=period)
    return data_closes, data_full


# Inicializar motor
engine = FinanceEngine(tickers)
data, full_data = _load_market_data(tuple(tickers), periodo)

if data is not None and full_data:
    # ASERCIÓN DE DATOS: Pasamos los datos del caché al motor interno
    engine.data = data 
    
    # Procesar métricas
    returns, vol, corr = engine.transform_data()

    # 1. KPIs Superiores
    cols = st.columns(len(tickers))
    for i, ticker in enumerate(tickers):
        with cols[i]:
            last_p = data[ticker].iloc[-1]
            prev_p = data[ticker].iloc[-2]
            delta = ((last_p / prev_p) - 1) * 100
            st.metric(label=ticker, value=f"{last_p:.2f}", delta=f"{delta:.2f}%")

    # 2. Gráfico Principal (Rendimiento Acumulado)
    st.subheader("📈 Rendimiento Relativo (Base 100)")
    data_norm = (data / data.iloc[0]) * 100
    fig_main = px.line(data_norm, labels={'value': 'Crecimiento %', 'Date': 'Fecha'})
    st.plotly_chart(fig_main, width="stretch")

    # 3. Dos columnas para Riesgo y Correlación
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("⚡ Volatilidad Anualizada")
        fig_vol = px.bar(vol, color=vol.values, color_continuous_scale='Reds')
        st.plotly_chart(fig_vol, width="stretch")

    with col_b:
        st.subheader("🔗 Matriz de Correlación")
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, width="stretch")

    # --- ACCIONES PRO (SIDEBAR) ---
    st.sidebar.markdown("---")
    st.sidebar.header("🚀 Funciones Pro")

    if st.sidebar.button("🔔 Enviar Alerta Telegram"):
        if send_telegram_alert(tickers, vol, price_data=data):
            st.sidebar.success("Resumen enviado a Telegram ✅")
        else:
            st.sidebar.error("Error en API Telegram")

    # Generamos las figuras y las mostramos en Streamlit
    data_norm = (data / data.iloc[0]) * 100
    fig_main = px.line(data_norm, title="Rendimiento")
    st.plotly_chart(fig_main, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        fig_vol = px.bar(vol, title="Volatilidad")
        st.plotly_chart(fig_vol)
    with col2:
        fig_corr = px.imshow(corr, text_auto=True, title="Correlación")
        st.plotly_chart(fig_corr)

    # --- BOTÓN DE DESCARGA ---
    # --- LÓGICA PARA GENERAR LOS INFORMES DE IA ---
    # Creamos un diccionario para guardar los análisis breves para el PDF
    if 'ai_cache' not in st.session_state:
        st.session_state.ai_cache = {}

    if st.sidebar.button("🤖 Generar Análisis IA para Informe PDF"):
        with st.sidebar:
            with st.spinner("Analizando noticias de todos los activos..."):
                for t in tickers:
                    st.session_state.ai_cache[t] = get_ai_analysis(t, is_brief=True)
                st.success("Análisis completados. Ya puedes descargar el PDF.")

    # --- ACTUALIZACIÓN DEL BOTÓN DE DESCARGA ---
    if st.session_state.ai_cache:
        try:
            # Pasamos el diccionario ai_cache a la función del PDF
            full_pdf_bytes = create_full_pdf(
                data, vol, corr, tickers,
                fig_main, fig_vol, fig_corr,
                st.session_state.ai_cache,
                bt_results=st.session_state.get('bt_results'),
                markowitz_res=st.session_state.get('markowitz_res')
            )

            st.sidebar.download_button(
                label="📥 Descargar Reporte con IA (PDF)",
                data=full_pdf_bytes,
                file_name="reporte_ia_financiero.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.sidebar.error(f"Error al generar PDF: {e}")
    else:
        st.sidebar.warning("Pulsa el botón de arriba para incluir el análisis de IA en el PDF.")

    # 4. Tabla de datos técnica
    with st.expander("Inspeccionar Data Lake (Parquet Format)"):
        st.dataframe(data.tail(10), width="stretch")

    # --- NUEVA SECCIÓN: RADAR DE OPORTUNIDADES (FINVIZ) ---
    st.divider()
    st.header("🌍 Radar de Oportunidades Global (Finviz & Correlación)")
    st.markdown("Busca nuevos activos en el mercado global que matemáticamente te ayuden a **diversificar y mejorar** tu portafolio actual.")

    col_screener, col_s_opts = st.columns([1, 3])
    with col_screener:
        st.subheader("Filtros de Búsqueda")
        s_market_cap = st.selectbox("Market Cap", ["Any", "Mega ($200bln and more)", "+Large (over $10bln)", "+Mid (over $2bln)", "+Small (over $300mln)"])
        s_sector = st.selectbox("Sector", ["Any", "Technology", "Healthcare", "Financial", "Energy", "Consumer Defensive"])
        s_index = st.selectbox("Index", ["Any", "S&P 500", "DJIA"])
        btn_search = st.button("🔍 Escanear Mercado", use_container_width=True)
        
    with col_s_opts:
        if btn_search:
            with st.spinner("Conectando con Finviz y analizando correlaciones con tu cartera..."):
                screener = AssetScreener(current_tickers=tickers)
                res_df, msg = screener.find_opportunities(
                    market_cap=s_market_cap,
                    sector=s_sector,
                    index=s_index
                )
                
                st.session_state.screener_results = res_df
                st.session_state.screener_msg = msg
                
        if 'screener_results' in st.session_state:
            st.info(st.session_state.screener_msg)
            df_res = st.session_state.screener_results
            
            if not df_res.empty:
                st.dataframe(df_res.style.format({
                    'Correlación vs Cartera': "{:.2f}",
                    'Retorno (6m)': "{:.2%}",
                    'Riesgo (Volatilidad)': "{:.2%}"
                }))
                
                st.markdown("Añadir candidatos seleccionados a tu cartera principal:")
                add_tickers = st.multiselect("Selecciona los Tickers que quieres analizar en el Dashboard:", df_res['Ticker'].tolist())
                
                if st.button("➕ Añadir a mi Portafolio"):
                    new_list = st.session_state.base_tickers + ", " + ", ".join(add_tickers)
                    st.session_state.base_tickers = new_list
                    st.success("Activos añadidos con éxito. ¡Refresca la página o haz click en cualquier botón del menú lateral para recargar todo el sistema con tus nuevos activos!")

    # --- NUEVA SECCIÓN: PREDICCIÓN Y BACKTESTING ---
    st.divider()
    st.header("🔮 IA Predictiva (Machine Learning) y Backtesting")
    st.markdown("Entrena modelos de Machine Learning en *tiempo real* para pronosticar la dirección del próximo movimiento y simula su rendimiento.")

    col_model, col_bt = st.columns([1, 2])

    with col_model:
        st.subheader("Configuración del Modelo")
        target_ticker = st.selectbox("Activo a modelar:", tickers, key="ml_ticker")
        horizon = st.slider("Horizonte de Predicción (Días):", 1, 10, 5, help="Predice si el precio subirá o bajará de aquí a X días.")
        model_type = st.radio("Algoritmo", ["XGBoost", "Random Forest", "LightGBM", "CatBoost", "Ensemble"])
        comission = st.number_input("Comisión por Operación (%)", value=0.1, step=0.01) / 100.0
        
        run_backtest = st.button("🚀 Entrenar y Simular (VectorBT)", use_container_width=True)

    with col_bt:
        if run_backtest:
            with st.spinner(f"1. Calculando Indicadores Globales para todos los activos..."):
                global_ml = MLEngine(full_data)
                global_ml.create_features_and_target(target_horizon=horizon)
            
            with st.spinner(f"2. Entrenando {model_type} Global (Time Series Split)..."):
                if model_type == 'Ensemble':
                    accs = []
                    for m in ['XGBoost', 'LightGBM', 'CatBoost']:
                        res = global_ml.train_and_evaluate(model_name=m, n_splits=5, ticker="GLOBAL")
                        accs.append(res['accuracy'])
                    eval_results = {'accuracy': np.mean(accs), 'report': res['report']}
                else:
                    eval_results = global_ml.train_and_evaluate(model_name=model_type, n_splits=5, ticker="GLOBAL")
                
            with st.spinner(f"3. Generando señales para {target_ticker}..."):
                ml_sys = MLEngine(full_data[target_ticker])
                ml_sys.create_features_and_target(target_horizon=horizon, ticker_name=target_ticker)
                
                # Inyectar el modelo global
                if model_type == 'Ensemble':
                    for m in ['XGBoost', 'LightGBM', 'CatBoost']:
                        ml_sys.trained_models[m] = global_ml.trained_models[m]
                    if hasattr(global_ml, 'selected_features'):
                        ml_sys.selected_features = global_ml.selected_features
                else:
                    ml_sys.trained_models[model_type] = global_ml.trained_models[model_type]
                    if hasattr(global_ml, 'selected_features'):
                        ml_sys.selected_features = global_ml.selected_features
                    
                ml_sys.selected_features_list = global_ml.selected_features_list
                
                signals = ml_sys.generate_signals(model_name=model_type)

            with st.spinner("4. Ejecutando Backtester (Cálculo Vectorial)..."):
                # Ejecutar Backtest
                bt = Backtester(ml_sys.df_processed['Close'], signals, comission_pct=comission)
                metrics = bt.get_metrics()
                bt_fig = bt.get_plotly_chart()
                
            st.success("✅ Simulación completada.")
            
            # --- MOSTRAR RESULTADOS ---
            # Guardamos los resultados iterativos en sesión para la recarga
            st.session_state.bt_results = {
                'fig': bt_fig,
                'accuracy': eval_results['accuracy'],
                'report': {**eval_results['report'], 'WFO_Accuracy': eval_results['accuracy'] * 100},
                'metrics': metrics,
                'model_type': model_type
            }
            
        if 'bt_results' in st.session_state:
            res = st.session_state.bt_results
            tabs = st.tabs(["Gráfico de Rendimiento", "Métricas del Modelo", "Estadísticas del Backtest"])
            
            with tabs[0]:
                st.plotly_chart(res['fig'], use_container_width=True)
                
            with tabs[1]:
                st.markdown("#### Precisión del Modelo (Walk-Forward Optimization)")
                st.metric(label=f"Avg WFO Accuracy ({res['model_type']})", value=f"{res['accuracy']:.2%}")
                st.markdown("*(Nota: En finanzas, una precisión WFO superior al 53-55% ya es excepcionalmente buena debido a la naturaleza aleatoria del mercado).*")
                
                st.markdown("#### Matriz de Clasificación (Último Fold)")
                st.dataframe(pd.DataFrame(res['report']).transpose().style.format("{:.2f}"))

            with tabs[2]:
                st.markdown("#### Métricas Financieras (VectorBT)")
                
                # Mostrar métricas en formato grid usando columnas
                m_cols = st.columns(3)
                idx = 0
                for k, v in res['metrics'].items():
                    with m_cols[idx % 3]:
                        # Formateo dependiendo del tipo de dato
                        val = float(v) if isinstance(v, (int, float)) else 0
                        if pd.isna(val):
                            val = 0
                            
                        if "%" in k or k == "Win Rate [%]":
                            st.metric(k, f"{val:.2f}%")
                        elif "Sharpe" in k:
                            st.metric(k, f"{val:.2f}")
                        else:
                            st.metric(k, f"{int(val)}")
                    idx += 1
                        

    # --- NUEVA SECCIÓN: GESTIÓN DE PORTAFOLIO (MARKOWITZ) ---
    st.divider()
    st.header("⚖️ Gestión Cuantitativa de Cartera (Frontera de Markowitz)")
    st.markdown("El optimizador matemático calcula **exactamente qué porcentaje de tu capital** deberías invertir en cada uno de los activos seleccionados para maximizar tus retornos ajustados al riesgo (Sharpe Ratio).")

    if st.button("📊 Calcular Distribución Óptima de Capital"):
        with st.spinner("Resolviendo modelo matemático de varianza media de Markowitz..."):
            port_opt = PortfolioOptimizer(data)
            opt_results = port_opt.optimize_max_sharpe()
            
            w_df = pd.DataFrame.from_dict(opt_results['weights'], orient='index', columns=['Peso Asignado'])
            w_df = w_df[w_df['Peso Asignado'] > 0.001] # Filtrar activos con 0% de peso
            
            # Gráfico Torta de Plotly
            fig_pie = px.pie(
                values=w_df['Peso Asignado'], 
                names=w_df.index, 
                title="Composición de Portafolio Ideal (Max Sharpe)",
                hole=0.4 # Estilo "Donut"
            )
            fig_pie.update_traces(textinfo='percent+label', textfont_size=12)

            w_df_display = w_df.copy()
            w_df_display['Peso Asignado'] = (w_df_display['Peso Asignado'] * 100).map("{:.2f}%".format)
            
            # Guardamos los resultados de Markowitz
            st.session_state.markowitz_res = {
                'fig': fig_pie,
                'opt_results': opt_results,
                'w_df_display': w_df_display
            }
            
    if 'markowitz_res' in st.session_state:
        m_res = st.session_state.markowitz_res
        
        # Mostrar la UI de Markowitz
        col_weights, col_metrics = st.columns([1, 1])
        
        with col_weights:
            st.plotly_chart(m_res['fig'], use_container_width=True)
            
        with col_metrics:
            st.subheader("Métricas Teóricas Anualizadas")
            st.metric("Retorno Esperado del Portafolio", f"{m_res['opt_results']['expected_return'] * 100:.2f}%")
            st.metric("Volatilidad Esperada (Riesgo)", f"{m_res['opt_results']['expected_volatility'] * 100:.2f}%")
            st.metric("Ratio de Sharpe Máximo Alcanzado", f"{m_res['opt_results']['sharpe_ratio']:.3f}")
            
            st.markdown("---")
            st.markdown("### Tabla de Pesos Óptimos")
            
            # Mostrar tabla limpia
            st.dataframe(m_res['w_df_display'], width=300)

    # --- SECCIÓN DE INTELIGENCIA ARTIFICIAL (NOTICIAS GROQ) ---
    st.divider()
    st.header("🧠 AI Market Insights (Análisis Semántico de Noticias)")

    col_ia, col_info = st.columns([1, 2])

    with col_ia:
        selected_ticker = st.selectbox("Selecciona un activo para analizar en profundidad:", tickers)
        analyze_btn = st.button("Generar Análisis Detallado")

    with col_info:
        if analyze_btn:
            with st.spinner(f"La IA Semántica está procesando las últimas noticias de {selected_ticker}..."):
                reporte_largo = get_ai_analysis(selected_ticker, is_brief=False)
                
                if 'ai_reports_long' not in st.session_state:
                    st.session_state.ai_reports_long = {}
                
                st.session_state.ai_reports_long[selected_ticker] = reporte_largo
                st.session_state.last_analyzed_ticker = selected_ticker
                
        if 'ai_reports_long' in st.session_state and 'last_analyzed_ticker' in st.session_state:
            last_t = st.session_state.last_analyzed_ticker
            st.markdown(f"### Informe Detallado: {last_t}")
            st.info(st.session_state.ai_reports_long[last_t])

else:
    st.error("Error al conectar con la API de datos. Revisa los tickers.")

# ─────────────────────────────────────────────────────────────────
# 🏦 CONTROL DE TRADING EN VIVO (IBKR)
# ─────────────────────────────────────────────────────────────────
st.divider()
st.header("🏦 Control de Trading en Vivo (Interactive Brokers)")

# Importaciones locales aquí para no interrumpir la carga si TWS no está activo
from core.broker_executor import BrokerExecutor, IS_LIVE
from core.risk_manager import RiskManager
from core.trading_runner import run_trading_session, JOURNAL_PATH
import os

# --- Indicador de modo ---
mode_label = "🔴 **LIVE — DINERO REAL**" if IS_LIVE else "🟡 **PAPER TRADING — Simulación Segura**"
st.markdown(f"### Modo Activo: {mode_label}")
if IS_LIVE:
    st.warning("⚠️ Estás en modo LIVE. Las órdenes se ejecutan con dinero real en tu cuenta de IBKR.")
else:
    st.info("ℹ️ Modo Paper. Para activar Live, añade `LIVE_TRADING=true` al fichero `.env` y reinicia.")

col_trading_1, col_trading_2 = st.columns([2, 1])

with col_trading_1:
    st.subheader("📊 Cuenta y Posiciones")

    if st.button("🔄 Actualizar Estado de la Cuenta"):
        with st.spinner("Conectando con IBKR..."):
            broker = BrokerExecutor()
            if broker.connect():
                account_info = broker.get_account_balance()
                positions_info = broker.get_open_positions()
                broker.disconnect()
                st.session_state['ibkr_account'] = account_info
                st.session_state['ibkr_positions'] = positions_info
                st.success("Conectado y datos actualizados ✅")
            else:
                st.error("❌ No se pudo conectar a Interactive Brokers.")
                st.info("""
                **Sigue estos pasos en TWS:**
                1. Ve a `File` -> `Global Configuration`
                2. Busca `API` -> `Settings` en el menú izquierdo
                3. ✅ Marca: **'Enable ActiveX and Socket Clients'**
                4. ✅ Desmarca: **'Read-Only API'** (para poder operar)
                5. Verifica el puerto: `7497` (Paper) o `7496` (Live)
                """)

    if 'ibkr_account' in st.session_state:
        acc = st.session_state['ibkr_account']
        c1, c2, c3 = st.columns(3)
        c1.metric("Capital Disponible", f"{acc.get('AvailableFunds', 0):.2f} €")
        c2.metric("Valor Neto (NAV)", f"{acc.get('NetLiquidation', 0):.2f} €")
        c3.metric("PnL Hoy", f"{acc.get('RealizedPnL', 0):.2f} €",
                  delta_color="normal" if acc.get('RealizedPnL', 0) >= 0 else "inverse")

    if 'ibkr_positions' in st.session_state:
        positions_df = pd.DataFrame(st.session_state['ibkr_positions'])
        if not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.caption("Sin posiciones abiertas actualmente.")

    # Parámetros de la sesión de trading
    st.subheader("⚙️ Configuración de Sesión")
    trading_tickers = st.multiselect(
        "Activos a operar",
        options=tickers if data is not None else ["AAPL", "MSFT"],
        default=tickers[:2] if data is not None and len(tickers) >= 2 else []
    )
    trading_model = st.radio("Modelo ML", ["XGBoost", "Random Forest", "LightGBM", "CatBoost", "Ensemble"], horizontal=True)
    force_retrain = st.checkbox("🔄 Forzar Reentrenamiento (Ignorar modelos guardados)", value=False)

    # Mostrar resumen de guardarraíles activos
    with st.expander("🛡️ Guardarraíles de Riesgo Activos"):
        risk = RiskManager()
        capital_est = st.session_state.get('ibkr_account', {}).get('AvailableFunds', 300.0)
        risk_summary = risk.get_risk_summary(capital_est)
        for k, v in risk_summary.items():
            st.markdown(f"- **{k}**: {v}")

with col_trading_2:
    st.subheader("🎮 Acciones")

    # --- Botón de Ejecución ---
    if st.button("▶️ Ejecutar Sesión de Trading", type="primary", use_container_width=True):
        if not trading_tickers:
            st.error("Selecciona al menos un activo para operar.")
        else:
            with st.spinner(f"Ejecutando sesión en {broker.mode if 'broker' in dir() else 'PAPER'}..."):
                result = run_trading_session(
                    trading_tickers,
                    model_type=trading_model,
                    force_retrain=force_retrain
                )
                st.session_state['trading_result'] = result

            if result.get('status') == 'ok':
                st.success("✅ Sesión completada. Revisa el journal.")
            else:
                st.error(f"Error: {result.get('msg', 'Desconocido')}")

    st.markdown("---")

    # --- Kill Switch (C3: Protegido con contraseña) ---
    st.markdown("### 🛑 Kill Switch")
    st.caption("Cancela TODAS las órdenes pendientes inmediatamente.")
    ks_password = st.text_input(
        "🔐 Contraseña Kill Switch",
        type="password",
        key="ks_pwd",
        help="Configura KILL_SWITCH_PASSWORD en el fichero .env"
    )
    if st.button("🛑 ACTIVAR KILL SWITCH", type="secondary", use_container_width=True):
        expected_ks_pwd = os.getenv("KILL_SWITCH_PASSWORD", "")
        if not expected_ks_pwd:
            st.error("❌ KILL_SWITCH_PASSWORD no está configurado en .env. Añádelo para activar el Kill Switch.")
        elif ks_password != expected_ks_pwd:
            st.error("❌ Contraseña incorrecta. No se cancelaron órdenes.")
        else:
            with st.spinner("Cancelando todas las órdenes..."):
                broker_ks = BrokerExecutor()
                if broker_ks.connect():
                    ok = broker_ks.cancel_all_orders()
                    broker_ks.disconnect()
                    if ok:
                        st.success("🛑 Kill Switch activado. Todas las órdenes canceladas.")
                    else:
                        st.error("Error al cancelar órdenes.")
                else:
                    st.error("No se pudo conectar a IBKR.")

# --- Audit Journal Viewer ---
st.subheader("📋 Registro de Auditoría (Trading Journal)")
if os.path.isfile(JOURNAL_PATH):
    df_journal = pd.read_csv(JOURNAL_PATH)
    # Mostrar las últimas 20 entradas en orden cronológico inverso
    st.dataframe(df_journal.tail(20).iloc[::-1], use_container_width=True)
    with open(JOURNAL_PATH, 'rb') as f:
        st.download_button("⬇️ Descargar Journal CSV", f, file_name="trading_journal.csv")
else:
    st.caption("Sin historial de operaciones todavía. Ejecuta una sesión de trading para comenzar.")



