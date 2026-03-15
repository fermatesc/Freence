import logging
import os

import yfinance as yf
import pandas as pd
import numpy as np

# Configuración de logs para parecer un pro
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
yf.set_tz_cache_location(os.getcwd())

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

    def extract_full_data(self, period="1y", interval="1d", include_macro=True) -> dict:
        """
        Extrae todos los datos (OHLCV) de los activos y añade de forma opcional
        indicadores macro (SPY, VIX) sincronizados por fecha.
        Soporta datos intradía cambiando 'interval' (ej. '1m', '5m', '1h').
        Retorna un diccionario: { "Ticker": pd.DataFrame(...) }
        """
        logging.info(f"Extrayendo datos FULL para ML: {self.tickers} (Interval: {interval})")
        
        macro_df = pd.DataFrame()
        if include_macro:
            try:
                spy_raw = yf.download("SPY", period=period, interval=interval, progress=False)
                if not spy_raw.empty:
                    if isinstance(spy_raw.columns, pd.MultiIndex):
                        spy_col = 'Adj Close' if 'Adj Close' in spy_raw.columns.levels[0] else 'Close'
                        macro_df['SPY_Close'] = spy_raw[spy_col]['SPY']
                    else:
                        spy_col = 'Adj Close' if 'Adj Close' in spy_raw.columns else 'Close'
                        macro_df['SPY_Close'] = spy_raw[spy_col]
                else:
                    logging.warning("No se pudo descargar SPY")
                
                vix_raw = yf.download("^VIX", period=period, interval=interval, progress=False)
                if not vix_raw.empty:
                    if isinstance(vix_raw.columns, pd.MultiIndex):
                        macro_df['VIX_Close'] = vix_raw['Close']['^VIX']
                    else:
                        macro_df['VIX_Close'] = vix_raw['Close']
                else:
                    logging.warning("No se pudo descargar ^VIX. Rellenando con 20.0 (mediana histórica) temporalmente.")
                    macro_df['VIX_Close'] = 20.0
            except Exception as e:
                logging.warning(f"Error parseando datos macro (SPY, VIX): {e}")
                if 'VIX_Close' not in macro_df.columns:
                    macro_df['VIX_Close'] = 20.0

        try:
            raw_data = yf.download(self.tickers, period=period, interval=interval, progress=False)
            raw_data = raw_data.ffill().dropna(how='all')
        except Exception as e:
            logging.error(f"Error en la extracción FULL: {e}")
            return {}

        full_data_dict = {}
        if isinstance(raw_data.columns, pd.MultiIndex):
            for t in self.tickers:
                try:
                    df_t = raw_data.xs(t, axis=1, level=1).copy()
                    if not macro_df.empty:
                        df_t = df_t.join(macro_df, how='left')
                    df_t = df_t.ffill().dropna()
                    if not df_t.empty:
                        full_data_dict[t] = df_t
                except KeyError:
                    logging.warning(f"Datos no encontrados para {t}")
        else:
            if len(self.tickers) == 1:
                t = self.tickers[0]
                df_t = raw_data.copy()
                if not macro_df.empty:
                    df_t = df_t.join(macro_df, how='left')
                df_t = df_t.ffill().dropna()
                if not df_t.empty:
                    full_data_dict[t] = df_t

        return full_data_dict

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
