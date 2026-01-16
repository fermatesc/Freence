import logging
import os

import yfinance as yf
import pandas as pd
import numpy as np

# Configuración de logs para parecer un pro
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FinanceEngine:
    def __init__(self, tickers):
        self.tickers = tickers
        self.data = None

    # Sustituye la parte de descarga por esta:
    def extract_data(self, period="1y") -> pd.DataFrame:
        try:
            logging.info(f"Extrayendo datos para: {self.tickers}")
            raw_data = yf.download(self.tickers, period=period)

            # Si descargamos varios tickers, yfinance devuelve un MultiIndex.
            # Intentamos sacar 'Adj Close', si no, 'Close'.
            if 'Adj Close' in raw_data.columns:
                self.data = raw_data['Adj Close']
            else:
                self.data = raw_data['Close']

            # Manejo de missing values (importante en finanzas)
            self.data = self.data.ffill().dropna()
            return self.data
        except Exception as e:
            logging.error(f"Error en la extracción: {e}")
            return None

    def transform_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Cálculos financieros clave usando lógica vectorial de Pandas"""
        logging.info("Calculando métricas financieras...")

        # 1. Retornos diarios (logarítmicos para mejor tratamiento estadístico)
        daily_returns = np.log(self.data / self.data.shift(1)).dropna()

        # 2. Volatilidad anualizada (Desviación estándar * raíz de días de mercado)
        volatility = daily_returns.std() * np.sqrt(252)

        # 3. Matriz de correlación
        correlation = daily_returns.corr()

        return daily_returns, volatility, correlation

    def load_to_parquet(self, df: pd.DataFrame, filename: str) -> None:
        """Persistencia eficiente"""
        directory = "data_output"
        if not os.path.exists(directory):
            os.makedirs(directory)

        path = os.path.join(directory, f"{filename}.parquet")
        df.to_parquet(path)
        logging.info(f"Datos guardados en {path}")


# --- EJECUCIÓN DEL FLUJO ---
# if __name__ == "__main__":
#     # Activos: Tecnología, Cripto, Oro y un Índice Mundial
#     my_assets = ["AAPL", "BTC-USD", "GC=F", "IWDA.AS"]
#
#     engine = FinanceEngine(my_assets)
#
#     # 1. Extraer
#     raw_df = engine.extract_data()
#
#     if raw_df is not None:
#         # 2. Transformar
#         returns, vol, corr = engine.transform_data()
#
#         # 3. Cargar (Guardamos los retornos para futuros modelos de IA)
#         engine.load_to_parquet(returns, "daily_returns")
#
#         print("\n--- RESUMEN DE VOLATILIDAD ---")
#         print(vol)
#         print("\n--- MATRIZ DE CORRELACIÓN ---")
#         print(corr)
