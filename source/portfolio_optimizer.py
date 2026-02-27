import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, df_prices: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Inicializa el optimizador con un DataFrame donde las columnas son los Tickers 
        y las filas son los precios diarios de cierre.
        """
        self.prices = df_prices.dropna()
        # Calcular retornos diarios logarítmicos
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        self.risk_free_rate = risk_free_rate
        
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        self.trading_days = 252 # Días de trading en un año
        self.num_assets = len(self.returns.columns)

    def portfolio_performance(self, weights):
        """
        Calcula el retorno anualizado y la volatilidad anualizada para un conjunto de pesos dados.
        """
        # Retorno esperado: W * R_media * 252
        returns = np.sum(self.mean_returns * weights) * self.trading_days
        
        # Volatilidad esperada: sqrt(W^T * Cov * W) * sqrt(252)
        variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        volatility = np.sqrt(variance) * np.sqrt(self.trading_days)
        
        return returns, volatility

    def negative_sharpe_ratio(self, weights):
        """
        Función a minimizar. Maximizamos el Sharpe minimizando el Sharpe negativo.
        """
        p_returns, p_volatility = self.portfolio_performance(weights)
        sharpe_ratio = (p_returns - self.risk_free_rate) / p_volatility
        return -sharpe_ratio

    def optimize_max_sharpe(self):
        """
        Encuentra los pesos óptimos que maximizan el Sharpe Ratio (Frontera Eficiente Markowitz).
        """
        # 1. Pesos iniciales iguales
        args = ()
        initial_guess = self.num_assets * [1. / self.num_assets,]
        
        # 2. Límites: Ningún activo puede tener peso negativo (No shorting), máximo 100% de la cartera
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        
        # 3. Restricciones: La suma de los pesos debe ser 1 (100% invertido)
        # CHECK MATH: Nos aseguramos matemáticamente de la distribución total del capital
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # 4. Optimizar minimizando el Negative Sharpe Ratio
        optimized = minimize(
            self.negative_sharpe_ratio, 
            initial_guess,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        opt_weights = optimized.x
        
        # Limpiar decimales microscópicos de SciPy (ej. 1e-17) y forzar suma 1
        opt_weights = np.round(opt_weights, 4)
        
        # Ajuste de compensación de redondeo
        if np.sum(opt_weights) != 1.0:
            diff = 1.0 - np.sum(opt_weights)
            # Acumulamos el error de redondeo en el activo con mayor peso
            max_idx = np.argmax(opt_weights)
            opt_weights[max_idx] += diff
            
        opt_returns, opt_vol = self.portfolio_performance(opt_weights)
        sharpe = (opt_returns - self.risk_free_rate) / opt_vol

        return {
            'weights': dict(zip(self.returns.columns, opt_weights)),
            'expected_return': opt_returns,
            'expected_volatility': opt_vol,
            'sharpe_ratio': sharpe
        }
