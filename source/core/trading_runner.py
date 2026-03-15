"""
trading_runner.py
-----------------
Orquestador del pipeline de trading automatizado.

Flujo de una sesión:
1. ML Engine genera señales para cada activo
2. RiskManager valida cada señal
3. BrokerExecutor ejecuta las órdenes aprobadas
4. Todo se registra en trading_journal.csv
5. Notificación por Telegram

Por defecto opera en PAPER TRADING.
Para Live: LIVE_TRADING=true en el .env
"""

import os
import sys
import csv
import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Ensure the 'source' directory is in the Python path so it can be run from the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.finance_ingestor import FinanceEngine
from ml.ml_engine import MLEngine
from core.broker_executor import BrokerExecutor
from core.risk_manager import RiskManager

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ruta del journal de auditoría
JOURNAL_PATH = os.path.join(os.path.dirname(__file__), "trading_journal.csv")

# Campos del CSV de auditoría
JOURNAL_FIELDS = [
    "timestamp", "mode", "ticker", "signal", "price",
    "qty", "stop_loss", "capital_before", "daily_pnl",
    "model_accuracy", "risk_approved", "risk_reason",
    "order_status", "order_id", "notes"
]


# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────

def _send_telegram(message: str):
    """Envía un mensaje de texto al bot de Telegram configurado."""
    token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("BOT_ID")
    if not token or not chat_id:
        log.warning("Telegram no configurado. Skipping notification.")
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        log.error(f"Error enviando Telegram: {e}")


def _rotate_journal_if_needed():
    """M2: Rota el journal CSV si supera 10 MB para evitar crecimiento ilimitado."""
    max_bytes = 10 * 1024 * 1024  # 10 MB
    if os.path.isfile(JOURNAL_PATH) and os.path.getsize(JOURNAL_PATH) > max_bytes:
        suffix = datetime.now().strftime("%Y%m")
        backup = JOURNAL_PATH.replace(".csv", f"_{suffix}.csv")
        try:
            os.rename(JOURNAL_PATH, backup)
            log.info(f"Journal rotado: {backup}")
        except Exception as e:
            log.error(f"No se pudo rotar el journal: {e}")


def _append_to_journal(record: dict):
    """Añade una entrada al CSV de auditoría. Crea el fichero si no existe."""
    _rotate_journal_if_needed()  # M2: rotar si es necesario
    file_exists = os.path.isfile(JOURNAL_PATH)
    with open(JOURNAL_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


# ─────────────────────────────────────────────
# Pipeline Principal
# ─────────────────────────────────────────────

def run_trading_session(
    tickers: list,
    model_type: str = "AUTO",
    target_horizon: int = 5,
    n_wfo_splits: int = 5
) -> dict:
    """
    Ejecuta una sesión completa de trading para la lista de activos.

    Args:
        tickers: Lista de símbolos (ej. ['AAPL', 'MSFT'])
        model_type: 'XGBoost' o 'Random Forest'
        target_horizon: Días a predecir
        n_wfo_splits: Splits WFO para validación

    Returns:
        Diccionario con el resumen de la sesión
    """
    session_start = datetime.now()
    broker = BrokerExecutor()
    risk = RiskManager()
    session_log = []
    mode = broker.mode

    log.info(f"{'='*60}")
    log.info(f"🚀 INICIANDO SESIÓN DE TRADING — {mode}")
    log.info(f"   Activos: {tickers}")
    log.info(f"{'='*60}")

    # --- PASO 1: Conectar al broker ---
    if not broker.connect():
        msg = "❌ No se pudo conectar a IBKR. ¿Está TWS/IB Gateway abierto?"
        log.error(msg)
        _send_telegram(f"🤖 *Trading Runner*\n{msg}")
        return {'status': 'error', 'msg': msg}

    # --- PASO 2: Estado de la cuenta ---
    account = broker.get_account_balance()
    capital = account.get('AvailableFunds', account.get('TotalCashValue', 0.0))
    daily_pnl = account.get('RealizedPnL', 0.0)
    open_positions = broker.get_open_positions()

    log.info(f"💰 Capital disponible: {capital:.2f}€ | PnL hoy: {daily_pnl:.2f}€")
    log.info(f"📊 Posiciones abiertas: {len(open_positions)}")

    # --- PASO 3: Cargar datos históricos ---
    engine = FinanceEngine(tickers)
    full_data = engine.extract_full_data(period="10y")

    if not full_data:
        broker.disconnect()
        return {'status': 'error', 'msg': 'No se pudieron cargar datos históricos.'}

    # No entrenamos Modelo Global aquí. Runners strictly use pre-trained models.
    log.info("\n--- El Trading Runner usará los modelos entrenados en background ---")

    results_by_ticker = {}

    # --- PASO 4: Analizar cada activo usando el Modelo Global ---
    for ticker in tickers:
        log.info(f"\n--- Analizando {ticker} ---")
        record = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "ticker": ticker,
            "capital_before": capital,
            "daily_pnl": daily_pnl
        }

        try:
            # Para evaluar el activo, extraemos sus features localmente
            if ticker not in full_data:
                log.warning(f"  Saltando {ticker}: Sin datos suficientes.")
                continue
            ml = MLEngine(full_data[ticker])
            ml.create_features_and_target(target_horizon=target_horizon, ticker_name=ticker)
            
            # Selección de Modelo (Local fallback Global)
            selected_model_name = model_type
            model_accuracy = 0.0
            model_status = "loaded"
            
            if model_type == 'AUTO':
                best_acc = 0.0
                best_mod = None
                for m in ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest']:
                    meta = ml.load_persistence(ticker, m)
                    if meta and float(meta.get('accuracy', 0)) > float(best_acc):
                        best_acc = float(meta['accuracy'])
                        best_mod = m
                
                # Si no hay local, buscar global
                if not best_mod:
                    log.warning(f"No hay modelo local para {ticker}, buscando globales...")
                    for m in ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest']:
                         meta = ml.load_persistence("GLOBAL", m)
                         if meta and float(meta.get('accuracy', 0)) > float(best_acc):
                             best_acc = float(meta['accuracy'])
                             best_mod = m
                             # Copiar al scope local para uso
                             ml.trained_models[m] = ml.trained_models.get(m) # Ya cargado por load_persistence bajo "GLOBAL" key si existe
                
                if not best_mod:
                     log.error(f"No hay modelo para operar en {ticker}.")
                     continue
                
                selected_model_name = best_mod
                model_accuracy = best_acc
                
            else:
                # Carga modelo especifico forzado
                meta = ml.load_persistence(ticker, selected_model_name)
                if not meta:
                    meta = ml.load_persistence("GLOBAL", selected_model_name)
                if meta:
                    model_accuracy = float(meta.get('accuracy', 0))
                else:
                    log.error(f"Modelo forzado {selected_model_name} no existe para {ticker}.")
                    continue

            # Generar señales
            signals = ml.generate_signals(model_name=selected_model_name)
            latest_signal = int(signals.iloc[-1])  # 1=BUY, 0=SELL/NEUTRAL

            signal_str = "BUY" if latest_signal == 1 else "NEUTRAL"
            log.info(f"  Modelo seleccionado: {selected_model_name} | Señal para {ticker}: {signal_str}")

            record['signal'] = signal_str
            record['model_accuracy'] = f"{model_accuracy:.4f}"
            record['notes'] = f"Model: {selected_model_name}"

            # 4b. Validación de riesgo
            risk_ok, risk_reason = risk.validate_trade(
                ticker=ticker,
                capital=capital,
                model_accuracy=model_accuracy,
                open_positions=open_positions,
                daily_pnl=daily_pnl
            )
            record['risk_approved'] = risk_ok
            record['risk_reason'] = risk_reason

            if not risk_ok:
                log.warning(f"  ⛔ Bloqueado: {risk_reason}")
                record['order_status'] = 'BLOCKED'
                record['notes'] = risk_reason
                results_by_ticker[ticker] = {'signal': signal_str, 'action': 'BLOCKED', 'reason': risk_reason, 'accuracy': model_accuracy}
                _append_to_journal(record)
                continue

            # 4c. Decidir acción
            current_price = float(full_data[ticker]['Close'].iloc[-1])
            record['price'] = f"{current_price:.2f}"

            in_position = any(p['ticker'] == ticker and p['qty'] > 0 for p in open_positions)

            if latest_signal == 1 and not in_position:
                # BUY: calcular tamaño y colocar orden
                qty = risk.calculate_position_size(capital, current_price, model_accuracy=model_accuracy)
                if qty == 0:
                    record['order_status'] = 'SKIPPED'
                    record['notes'] = 'Tamaño de posición = 0 (capital insuficiente para el precio)'
                    results_by_ticker[ticker] = {'signal': signal_str, 'action': 'SKIPPED', 'reason': 'Size=0', 'accuracy': model_accuracy}
                    _append_to_journal(record)
                    continue

                order_result = broker.place_bracket_order(ticker, 'BUY', qty, risk.stop_loss_pct)
                record['qty'] = qty
                record['stop_loss'] = order_result.get('stop_loss', 'N/A')
                record['order_status'] = order_result.get('status', 'error').upper()
                record['order_id'] = order_result.get('entry_order_id', 'N/A')
                results_by_ticker[ticker] = {**order_result, 'accuracy': model_accuracy}

            elif latest_signal == 0 and in_position:
                # SELL: cerrar posición existente
                close_result = broker.close_position(ticker)
                record['qty'] = close_result.get('qty', 0)
                record['order_status'] = close_result.get('status', 'error').upper()
                results_by_ticker[ticker] = {**close_result, 'accuracy': model_accuracy}

            else:
                log.info(f"  → Sin acción para {ticker} (señal={signal_str}, en_posición={in_position})")
                record['order_status'] = 'NO_ACTION'
                results_by_ticker[ticker] = {'signal': signal_str, 'action': 'NO_ACTION', 'accuracy': model_accuracy}

        except Exception as e:
            log.error(f"Error procesando {ticker}: {e}", exc_info=True)
            record['order_status'] = 'ERROR'
            record['notes'] = str(e)
            results_by_ticker[ticker] = {'status': 'error', 'msg': str(e)}

        session_log.append(record)
        _append_to_journal(record)

    # --- PASO 5: Notificación Telegram ---
    broker.disconnect()
    session_duration = (datetime.now() - session_start).seconds

    msg_lines = [f"🤖 *Sesión de Trading Completada* — {mode}\n"]
    for t, res in results_by_ticker.items():
        action = res.get('action', res.get('status', '?')).upper()
        acc = res.get('accuracy', 0)
        msg_lines.append(f"  • *{t}*: `{action}` | AccWFO: `{acc:.2%}`")

    msg_lines.append(f"\n💰 Capital: `{capital:.2f}€` | PnL: `{daily_pnl:.2f}€`")
    msg_lines.append(f"⏱️ Duración: `{session_duration}s` | Log: `trading_journal.csv`")
    _send_telegram("\n".join(msg_lines))

    log.info(f"\n✅ Sesión completada en {session_duration}s")
    return {
        'status': 'ok',
        'mode': mode,
        'results': results_by_ticker,
        'capital': capital,
        'daily_pnl': daily_pnl,
        'session_log': session_log
    }


if __name__ == "__main__":
    # Ejecutar manualmente para pruebas
    tickers = ["AAPL", "MSFT"]
    result = run_trading_session(tickers)
    print(result)
