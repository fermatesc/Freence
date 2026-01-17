import os
import io
import requests

import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
from dotenv import load_dotenv
import plotly.io as pio

from finance_ingestor import FinanceEngine
from daily_bot import get_ai_analysis

load_dotenv()

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Data Investment Hub", layout="wide")


# --- FUNCIONES DE UTILIDAD ---

def send_telegram_alert(tickers, vol_data):
    """Envía alerta de volatilidad por Telegram"""
    top_vol = vol_data.idxmax()
    message = f"⚠️ Alerta de Cartera\nActivo más volátil: {top_vol} ({vol_data[top_vol]:.2%})"

    url = f"https://api.telegram.org/bot{os.getenv('BOT_TOKEN')}/sendMessage"
    data = {"chat_id": os.getenv('BOT_ID'), "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data)
        return True
    except:
        return False


def create_full_pdf(data, vol, corr, tickers, fig_main, fig_vol, fig_corr):
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

    return bytes(pdf.output())


# --- INTERFAZ STREAMLIT ---

st.title("📊 Data Investment Engine v1.0")
st.markdown("Plataforma de ingeniería de datos financieros")

# Sidebar
st.sidebar.header("Configuración")
tickers_input = st.sidebar.text_input("Lista de Tickers", "AAPL, BTC-USD, GC=F, MSFT, IWDA.AS")
periodo = st.sidebar.selectbox("Rango Temporal", ["1mo", "6mo", "1y", "2y", "5y"], index=2)
tickers = [t.strip() for t in tickers_input.split(",")]

# Inicializar motor
engine = FinanceEngine(tickers)
data = engine.extract_data(period=periodo)

if data is not None:
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
        if send_telegram_alert(tickers, vol):
            st.sidebar.success("Alerta enviada")
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
                st.session_state.ai_cache
            )

            st.download_button(
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

    # --- SECCIÓN DE INTELIGENCIA ARTIFICIAL ---
    st.divider()
    st.header("🧠 AI Market Insights (Análisis Extendido)")

    col_ia, col_info = st.columns([1, 2])

    with col_ia:
        selected_ticker = st.selectbox("Selecciona un activo para analizar en profundidad:", tickers)
        analyze_btn = st.button("Generar Análisis Detallado")

    if analyze_btn:
        with st.spinner(f"La IA está procesando las últimas noticias de {selected_ticker}..."):
            reporte_largo = get_ai_analysis(selected_ticker, is_brief=False)
            st.markdown(f"### Informe Detallado: {selected_ticker}")
            st.info(reporte_largo)

else:
    st.error("Error al conectar con la API de datos. Revisa los tickers.")


