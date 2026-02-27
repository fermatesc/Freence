import vectorbt as vbt
import pandas as pd
import plotly.graph_objects as go

class Backtester:
    def __init__(self, price_series: pd.Series, signals: pd.Series, comission_pct: float = 0.001):
        """
        price_series: Serie de pandas con los precios de cierre.
        signals: Serie de pandas con señales (1 compra, 0 venta) alineada con los precios.
        comission_pct: Comisión por operación en decimal (ej. 0.001 = 0.1%).
        """
        # Alinear series por el índice de las señales (que omite los primeros NaNs del feature engineering)
        aligned_data = pd.concat([price_series, signals], axis=1).dropna()
        self.price = aligned_data.iloc[:, 0]
        self.entries = aligned_data.iloc[:, 1] == 1
        self.exits = aligned_data.iloc[:, 1] == 0
        
        self.comission = comission_pct
        self.portfolio = None

    def run_backtest(self):
        """
        Ejecuta la simulación VectorBT.
        Asume señales "close-to-close" (señal generada al cierre, ejecutada al siguiente precio).
        """
        self.portfolio = vbt.Portfolio.from_signals(
            close=self.price,
            entries=self.entries,
            exits=self.exits,
            fees=self.comission,
            init_cash=10000, # Capital inicial
            freq='D' # Frecuencia diaria
        )
        return self.portfolio

    def get_metrics(self):
        """Devuelve un diccionario con las principales métricas del backtest."""
        if self.portfolio is None:
            self.run_backtest()
            
        stats = self.portfolio.stats()
        
        # Filtramos un poco las métricas más legibles
        metrics = {
            'Retorno Total (%)': stats.get('Total Return [%]', 0),
            'Retorno Anualizado (%)': stats.get('Ann. Return [%]', 0),
            'Max Drawdown (%)': stats.get('Max Drawdown [%]', 0),
            'Win Rate (%)': stats.get('Win Rate [%]', 0),
            'Sharpe Ratio': stats.get('Sharpe Ratio', 0),
            'Operaciones Totales': stats.get('Total Trades', 0)
        }
        return metrics

    def get_plotly_chart(self):
        """Retorna una figura de Plotly con la evolución del capital."""
        if self.portfolio is None:
            self.run_backtest()
        
        # vectorbt tiene su propia función plot, pero a menudo extraemos el valor
        # del portafolio para tener más control visual en Streamlit
        value = self.portfolio.value()
        
        # Calculamos un Buy&Hold base para comparar usando el array subyacente de pandas
        bnh_shares = 10000 / self.price.iloc[0]
        # Restamos comisiones a la entrada de BnH
        bnh_value = (self.price * bnh_shares) * (1 - self.comission)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=value.index,
            y=value.values,
            mode='lines',
            name='Estrategia ML',
            line=dict(color='#00ff88', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=bnh_value.index,
            y=bnh_value.values,
            mode='lines',
            name='Buy & Hold (Referencia)',
            line=dict(color='#cccccc', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='Curva de Capital (Equity Curve) - Backtest',
            xaxis_title='Fecha',
            yaxis_title='Capital ($)',
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
