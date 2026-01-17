import os
import requests
import feedparser
from groq import Groq
from finance_ingestor import FinanceEngine
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


def run_daily_report():
    tickers = ["AAPL", "BTC-USD", "GC=F", "MSFT"]
    engine = FinanceEngine(tickers)
    data = engine.extract_data(period="5d")
    returns, vol, _ = engine.transform_data()

    mensaje = "🤖 *AI Financial Assistant*\n\n"

    for t in tickers:
        last_p = data[t].iloc[-1]
        change = ((last_p / data[t].iloc[-2]) - 1) * 100
        emoji = "🟢" if change >= 0 else "🔴"

        # Obtener análisis breve para Telegram
        analisis_ia = get_ai_analysis(t, is_brief=True)

        mensaje += f"{emoji} *{t}*: `{last_p:.2f}` ({change:.2f}%)\n"
        mensaje += f"🧠 *IA:* {analisis_ia}\n\n"

    # Enviar a Telegram
    token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("BOT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"})


if __name__ == "__main__":
    run_daily_report()