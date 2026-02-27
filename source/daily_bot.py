import os
import requests
import feedparser
from groq import Groq
from finance_ingestor import FinanceEngine
from portfolio_optimizer import PortfolioOptimizer
from dotenv import load_dotenv

load_dotenv()


def get_ai_analysis(ticker, is_brief=True):
    """Obtiene noticias y genera análisis con Groq"""
    try:
        # 1. Buscar noticias (RSS gratuito)
        url = f"https://news.google.com/rss/search?q={ticker}+stock+when:1d&hl=es&gl=ES&ceid=ES:es"
        feed = feedparser.parse(url)
        titulares = [entry.title for entry in feed.entries[:5]]
        contexto = "\n".join(titulares) if titulares else "Sin noticias recientes."

        # 2. Configurar Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        longitud = "un párrafo breve" if is_brief else "tres puntos detallados (sentimiento, catalizadores y riesgo)"

        prompt = f"""
        Analiza estos titulares para {ticker}:
        {contexto}

        Responde en español. Proporciona {longitud}. 
        Sé profesional y directo.
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Análisis no disponible actualmente. ({e})"


def get_markowitz_summary(tickers):
    """Ejecuta la optimización de Markowitz y devuelve el resumen formateado para Telegram."""
    try:
        engine = FinanceEngine(tickers)
        data = engine.extract_data(period="2y")  # 2 años para Markowitz robusto

        if data is None or data.empty:
            return "❌ Sin datos suficientes para cálculo cuantitativo."

        optimizer = PortfolioOptimizer(data)
        weights, exp_return, volatility, sharpe = optimizer.optimize()

        if weights is None:
            return "❌ El optimizador no pudo converger con estos activos."

        # Formatear tabla de pesos
        lines = ["📊 *Distribución Óptima Markowitz (Sharpe Max)*"]
        for ticker, weight in zip(tickers, weights):
            bar = "█" * int(weight * 20)  # Barra visual proporcional
            lines.append(f"  • *{ticker}*: `{weight:.1%}` {bar}")

        lines.append(f"\n📈 Retorno Esperado: `{exp_return:.2f}%`")
        lines.append(f"📉 Riesgo (Volatilidad): `{volatility:.2f}%`")
        lines.append(f"⚡ Sharpe Ratio: `{sharpe:.2f}`")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Error en análisis cuantitativo: {str(e)}"


def run_daily_report(tickers):
    engine = FinanceEngine(tickers)
    data = engine.extract_data(period="5d")
    returns, vol, _ = engine.transform_data()

    mensaje = "🤖 *AI Financial Assistant*\n\n"

    # --- BLOQUE 1: Precios y IA por ticker ---
    for t in tickers:
        try:
            last_p = data[t].iloc[-1]
            change = ((last_p / data[t].iloc[-2]) - 1) * 100
            emoji = "🟢" if change >= 0 else "🔴"

            # Obtener análisis breve para Telegram
            analisis_ia = get_ai_analysis(t, is_brief=True)

            mensaje += f"{emoji} *{t}*: `{last_p:.2f}` ({change:.2f}%)\n"
            mensaje += f"🧠 *IA:* {analisis_ia}\n\n"
        except Exception:
            mensaje += f"⚠️ Sin datos para {t}\n\n"

    # --- BLOQUE 2: Recomendación Cuantitativa Markowitz ---
    mensaje += "─────────────────────\n"
    mensaje += get_markowitz_summary(tickers)
    mensaje += "\n\n⚠️ _Rebalancear si los pesos actuales difieren >5% de lo óptimo._\n"

    url = f"https://api.telegram.org/bot{os.getenv('BOT_TOKEN')}/sendMessage"
    requests.post(url, data={"chat_id": os.getenv("BOT_ID"), "text": mensaje, "parse_mode": "Markdown"})
    print("✅ Reporte diario enviado a Telegram.")


if __name__ == "__main__":
    tickers = ["AAPL", "BTC-USD", "GC=F", "MSFT"]
    run_daily_report(tickers)