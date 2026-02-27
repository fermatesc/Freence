import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from finvizfinance.screener.overview import Overview
from datetime import datetime, timedelta
import warnings

# Ignorar warnings de yfinance
warnings.filterwarnings("ignore", category=FutureWarning)

@st.cache_data(ttl=86400, show_spinner=False)
def cached_download(tickers, start_date_str, end_date_str):
    """Descarga datos de YFinance usando el caché nativo de Streamlit (Válido 24h)"""
    return yf.download(
        tickers, 
        start=start_date_str, 
        end=end_date_str, 
        progress=False
    )['Close']

class AssetScreener:
    def __init__(self, current_tickers):
        """
        Inicializa el buscador de activos.
        param current_tickers: Lista de tickers que el usuario ya tiene en su portafolio.
        """
        self.current_tickers = current_tickers
        self.current_data = None
        self._load_current_portfolio_data()

    def _load_current_portfolio_data(self):
        """Descarga el histórico reciente de los activos actuales para la base de correlación."""
        if not self.current_tickers:
            return
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180) # 6 meses de datos
        
        try:
            self.current_data = cached_download(
                tuple(self.current_tickers), 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            # Si solo es 1 ticker, yf devuelve Series, lo forzamos a DF
            if isinstance(self.current_data, pd.Series):
                self.current_data = self.current_data.to_frame(name=self.current_tickers[0])
                
        except Exception as e:
            print(f"Error descargando datos base: {e}")
            self.current_data = pd.DataFrame()

    def find_opportunities(self, market_cap="Any", sector="Any", index="Any", max_results=20):
        """
        1. Extrae candidatos de Finviz según filtros.
        2. Calcula correlación con el portafolio actual.
        3. Devuelve los mejores candidatos para diversificar.
        """
        # --- 1. Scraping Finviz ---
        foverview = Overview()
        filters_dict = {}
        
        # Mapeo de filtros UI al formato de Finviz
        if market_cap != "Any":
            filters_dict['Market Cap.'] = market_cap
            
        if sector != "Any":
            filters_dict['Sector'] = sector
            
        if index != "Any":
            filters_dict['Index'] = index
            
        # Filtros base de calidad (Volumen > 1M para liquidez, EPS growth > 0 para empresas sanas)
        filters_dict['Average Volume'] = 'Over 1M'
        
        try:
            if filters_dict:
                foverview.set_filter(filters_dict=filters_dict)
                
            # Finviz scraper
            df_finviz = foverview.screener_view()
            
            if df_finviz.empty:
                return pd.DataFrame(), "No se encontraron activos con estos filtros en Finviz."
                
            # Extraer listado inicial (Limitamos a max_results para no saturar YFinance)
            candidates = df_finviz['Ticker'].head(max_results).tolist()
            companies_info = df_finviz[['Ticker', 'Company', 'Sector', 'Industry']].head(max_results)
            
        except Exception as e:
            return pd.DataFrame(), f"Error conectando con Finviz: {str(e)}"

        # --- 2. Análisis Cuantitativo YFinance (Correlación & Retorno) ---
        if self.current_data is None or self.current_data.empty:
            # Si no hay portfolio previo, solo devolvemos los de Finviz
            return companies_info, "Muestra inicial de Finviz (Sin portfolio base para correlacionar)"

        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        try:
            # Descargar precios de los candidatos de Finviz
            cand_data = cached_download(
                tuple(candidates), 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            # Limpiar NAs
            cand_data = cand_data.dropna(axis=1)
            valid_candidates = cand_data.columns.tolist()
            
            if not valid_candidates:
                return pd.DataFrame(), "No se pudieron descargar precios históricos de los candidatos."

            results = []
            
            # Calcular rendimientos diarios del portafolio actual (promedio simple)
            curr_returns = self.current_data.pct_change().dropna()
            port_returns_daily = curr_returns.mean(axis=1)

            # Analizar cada candidato viable
            for ticker in valid_candidates:
                if ticker in self.current_tickers:
                    continue # Saltar si ya lo tenemos
                    
                t_returns = cand_data[ticker].pct_change().dropna()
                
                # Alinear fechas
                aligned_returns = pd.concat([port_returns_daily, t_returns], axis=1, join='inner')
                aligned_returns.columns = ['Portfolio', 'Candidate']
                
                # Calcular Correlación
                correlation = aligned_returns['Portfolio'].corr(aligned_returns['Candidate'])
                
                # Calcular Retorno Acumulado Histórico (Momentum a 6 meses)
                cum_return = (cand_data[ticker].iloc[-1] / cand_data[ticker].iloc[0]) - 1
                
                # Riesgo propio (Volatilidad)
                volatility = t_returns.std() * np.sqrt(252)
                
                results.append({
                    'Ticker': ticker,
                    'Correlación vs Cartera': correlation,
                    'Retorno (6m)': cum_return,
                    'Riesgo (Volatilidad)': volatility
                })

            df_results = pd.DataFrame(results)
            
            if df_results.empty:
                 return pd.DataFrame(), "No hay resultados válidos tras el cruce matemático."
                 
            # --- 3. Puntuación y Ranking (Scoring Algorítmico) ---
            # Ideal: Correlación baja o negativa, Alto retorno.
            # Ordenar primero por menor correlación, luego mayor retorno
            df_results = df_results.sort_values(by=['Correlación vs Cartera', 'Retorno (6m)'], ascending=[True, False])
            
            # Hacer Merge con el nombre de la empresa e industria de Finviz
            final_df = pd.merge(df_results, companies_info, on='Ticker', how='left')
            
            # Reordenar columnas para UI
            cols_order = ['Ticker', 'Company', 'Sector', 'Correlación vs Cartera', 'Retorno (6m)', 'Riesgo (Volatilidad)']
            final_df = final_df[cols_order]
            
            return final_df.head(10), "Análisis Completado Exitosamente."
            
        except Exception as e:
            return pd.DataFrame(), f"Error en el análisis matemático: {str(e)}"
