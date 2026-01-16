import os
import requests
import pandas as pd
from finance_ingestor import FinanceEngine
from dotenv import load_dotenv

# Cargar variables (en local usa .env, en GitHub Actions usa los Secrets)
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_msg(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, data=payload)
    return response.json()


def generate_summary():
    # 1. Configuración de activos (puedes cambiarlos aquí)
    tickers = ["AAPL", "BTC-USD", "GC=F", "MSFT", "IWDA.AS"]
    engine = FinanceEngine(tickers)

    # 2. Obtener datos
    data = engine.extract_data(period="1mo")
    if data is None:
        return "❌ Error al extraer datos financieros."

    returns, vol, corr = engine.transform_data()

    # 3. Construir el mensaje
    mensaje = "🚀 *Resumen Diario de Inversión*\n"
    mensaje += f"📅 Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n"
    mensaje += "--------------------------------\n\n"

    for t in tickers:
        last_price = data[t].iloc[-1]
        prev_price = data[t].iloc[-2]
        change = ((last_price / prev_price) - 1) * 100
        vola = vol[t]

        emoji = "🟢" if change >= 0 else "🔴"
        mensaje += f"{emoji} *{t}*\n"
        mensaje += f"   • Precio: `{last_price:.2f}`\n"
        mensaje += f"   • Var: *{change:.2f}%*\n"
        mensaje += f"   • Vol: `{vola:.2%}`\n\n"

    # Alerta especial de volatilidad
    high_vol = vol[vol > 0.40].index.tolist()  # Ejemplo: activos con >40% vola
    if high_vol:
        mensaje += f"⚠️ *Atención:* Alta volatilidad en: {', '.join(high_vol)}\n"

    mensaje += "\n🔗 [Ver Dashboard Completo](https://tu-app.streamlit.app)"

    return mensaje


if __name__ == "__main__":
    print("Iniciando bot diario...")
    summary = generate_summary()
    result = send_telegram_msg(summary)

    if result.get("ok"):
        print("✅ Resumen enviado con éxito a Telegram.")
    else:
        print(f"❌ Fallo al enviar: {result}")