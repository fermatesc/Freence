"""
feedback_loop.py
----------------
Analiza el trading_journal.csv y comprueba la exactitud de las decisiones tomadas
varios días atrás vs el precio actual. 
Aplica Fine-Tuning o fuerza un re-entrenamiento exhaustivo en función de los errores.
"""
import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.ml_engine import MLEngine
from data.finance_ingestor import FinanceEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

JOURNAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "trading_journal.csv")
MAX_ERRORS_BEFORE_RETRAIN = 3

def run_feedback_loop(target_horizon: int = 5):
    """
    Lee el journal e identifica predicciones de hace `target_horizon` días.
    Comprueba si fueron acertadas o no comparadas con el precio actual.
    """
    if not os.path.exists(JOURNAL_PATH):
        log.warning("No existe el journal para analizar.")
        return

    df_journal = pd.read_csv(JOURNAL_PATH)
    if df_journal.empty:
        log.info("Journal vacío.")
        return
        
    df_journal['timestamp'] = pd.to_datetime(df_journal['timestamp'], errors='coerce')
    df_journal.dropna(subset=['timestamp'], inplace=True)
    
    # Filtramos las órdenes
    df_trades = df_journal[df_journal['order_status'].isin(['FILLED', 'SUBMITTED', 'NO_ACTION'])].copy()
    
    now = datetime.now()
    cutoff_start = now - timedelta(days=target_horizon + 2)
    cutoff_end = now - timedelta(days=target_horizon - 1)
    
    trades_to_verify = df_trades[(df_trades['timestamp'] >= cutoff_start) & (df_trades['timestamp'] <= cutoff_end)]
    
    if trades_to_verify.empty:
        log.info("No hay trades en el periodo objetivo para evaluar.")
        return
        
    log.info(f"Verificando {len(trades_to_verify)} predicciones generadas hace ~{target_horizon} días...")
    
    tickers = trades_to_verify['ticker'].unique().tolist()
    finance_engine = FinanceEngine(tickers)
    market_data = finance_engine.extract_full_data(period="1mo")
    
    for _, row in trades_to_verify.iterrows():
        ticker = row['ticker']
        pred_signal = row['signal']
        try:
            exec_price = float(row['price'])
        except (ValueError, TypeError):
            continue
        
        if ticker not in market_data or market_data[ticker].empty:
            continue
            
        df = market_data[ticker]
        current_price = float(df['Close'].iloc[-1])
        pct_change = (current_price - exec_price) / exec_price
        
        # Determinar resultado real
        is_error = False
        if pred_signal == 'BUY' and pct_change < -0.01:
            is_error = True
        elif pred_signal != 'BUY' and pct_change > 0.02:
            is_error = True
            
        if is_error:
            log.warning(f"❌ FALLO detectado en {ticker}: Predijo {pred_signal} a {exec_price:.2f}, actual {current_price:.2f} ({pct_change:.2%})")
            
            # Revisar si hay errores consecutivos en el journal reciente para este ticker
            recent_trades = df_trades[df_trades['ticker'] == ticker].sort_values('timestamp', ascending=False).head(5)
            # Simplificamos: si ya sabemos que este último falló, podríamos asumir 1 fallo.
            # Verificamos si fine-tunear es suficiente.
            
            ml = MLEngine(df)
            ml.create_features_and_target(target_horizon=target_horizon, ticker_name=ticker)
            
            df_valid = ml.features[ml.target.notna()]
            target_valid = ml.target[ml.target.notna()]
            
            if len(df_valid) == 0:
                continue
                
            X_new = df_valid.iloc[-5:] # Tomar últimos 5 días válidos
            y_new = target_valid.iloc[-5:]
            
            model_to_tune = 'XGBoost'
            success = ml.fine_tune(model_name=model_to_tune, ticker=ticker, X_new=X_new, y_new=y_new, error_weight=3.0)
            if success:
                 log.info(f"🤖 Fine-tuning dinámico aplicado a {ticker} ({model_to_tune})")
            
        else:
            log.info(f"✅ Predicción CORRECTA para {ticker} ({pct_change:.2%})")

if __name__ == "__main__":
    run_feedback_loop()
