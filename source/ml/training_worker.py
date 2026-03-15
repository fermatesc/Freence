"""
training_worker.py
------------------
Proceso en background encargado de escanear el mercado, descargar datos
y entrenar múltiples modelos de forma exhaustiva (Grid Search).
Guarda los modelos y sus métricas en disco para que el runner las consuma.
"""
import os
import sys
import logging
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.finance_ingestor import FinanceEngine
from ml.ml_engine import MLEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def run_worker(tickers: list, target_horizon: int = 5):
    """
    Ejecuta un ciclo de entrenamiento exhaustivo para la lista de activos.
    """
    log.info(f"🚀 INICIANDO TRAINING WORKER para {len(tickers)} activos...")
    
    engine = FinanceEngine(tickers)
    full_data = engine.extract_full_data(period="5y") # 5 años para mayor robustez
    
    if not full_data:
        log.error("No se pudieron cargar datos históricos.")
        return

    models_to_train = ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest']

    for ticker in tickers:
        if ticker not in full_data or full_data[ticker].empty:
            log.warning(f"  Saltando {ticker}: Sin datos.")
            continue
            
        log.info(f"\n[{ticker}] --- Iniciando Entrenamiento Exhaustivo ---")
        df = full_data[ticker]
        ml = MLEngine(df)
        ml.create_features_and_target(target_horizon=target_horizon, ticker_name=ticker)
        
        best_acc = 0.0
        best_model = None

        for model_name in models_to_train:
            log.info(f"[{ticker}] Entrenando {model_name} con Grid Search...")
            try:
                res = ml.train_and_evaluate(model_name=model_name, n_splits=5, ticker=ticker, search_type='grid')
                acc = res['accuracy']
                log.info(f"[{ticker}] {model_name} WFO Accuracy: {acc:.2%}")
                
                if float(acc) > float(best_acc):
                    best_acc = acc
                    best_model = model_name
            except Exception as e:
                log.error(f"[{ticker}] Error entrenando {model_name}: {e}")
                
        log.info(f"[{ticker}] ✅ Modelos listos. Mejor modelo histórico: {best_model} ({best_acc:.2%})")

if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    # Para producción, este proceso puede correr infinitamente o ser lanzado por un cron
    while True:
        run_worker(test_tickers)
        log.info("Ciclo terminado. Durmiendo 24h...")
        time.sleep(86400) # 24 horas
