"""
broker_executor.py
------------------
Módulo de conexión y ejecución de órdenes con Interactive Brokers.
DISEÑADO PARA PYTHON 3.14 + STREAMLIT: Usa un hilo de fondo dedicado para 
el event loop de IBKR, evitando errores de 'Timeout should be used inside a task'.
"""

import os
import logging
import threading
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import nest_asyncio

# Aplicamos nest_asyncio al inicio
nest_asyncio.apply()

from ib_insync import IB, Stock, MarketOrder, StopOrder, util

load_dotenv()
log = logging.getLogger(__name__)

# --- Constantes de Configuración ---
PAPER_PORT = 7497
LIVE_PORT  = 7496
IS_LIVE = os.getenv("LIVE_TRADING", "false").lower() == "true"
ACTIVE_PORT = LIVE_PORT if IS_LIVE else PAPER_PORT
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")

class BrokerExecutor:
    """
    Gestiona la conexión a IBKR de forma SEGURA para Streamlit.
    Utiliza el event loop global parcheado por nest_asyncio.
    """

    def __init__(self, client_id=None):
        self.ib = IB()
        self.connected = False
        self.mode = "🔴 LIVE" if IS_LIVE else "🟡 PAPER TRADING"
        # M1: clientId dinámico basado en PID para evitar conflictos si dashboard
        # y trading_runner se conectan simultáneamente al mismo TWS.
        default_id = int(os.getenv("IB_CLIENT_ID") or (os.getpid() % 9000 + 1000))
        self.client_id = client_id or default_id
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Establece conexión con TWS."""
        with self._lock:
            if self.ib.isConnected():
                return True
            try:
                # En Python 3.14, usamos el método síncrono que ib_insync 
                # maneja internamente con su propio loop worker si fuera necesario.
                # IMPORTANTE: No usamos connectAsync aquí para evitar el error de Timeout.
                self.ib.connect(IB_HOST, ACTIVE_PORT, clientId=self.client_id, timeout=10)
                self.connected = True
                log.info(f"Conectado a IBKR en puerto {ACTIVE_PORT}")
                return True
            except Exception as e:
                log.error(f"Error conectando a IBKR: {e}")
                self.connected = False
                return False

    def disconnect(self):
        """Cierra conexión."""
        with self._lock:
            if self.ib.isConnected():
                self.ib.disconnect()
                self.connected = False

    def get_account_balance(self) -> dict:
        """Resumen de cuenta."""
        if not self.connect(): return {}
        try:
            # Llamada síncrona (internamente usa el loop parcheado)
            summary = self.ib.accountSummary()
            result = {}
            for item in summary:
                if item.tag in ('TotalCashValue', 'NetLiquidation', 'AvailableFunds', 'RealizedPnL'):
                    result[item.tag] = float(item.value)
            return result
        except Exception as e:
            log.error(f"Error balance: {e}")
            return {}

    def get_open_positions(self) -> list:
        """Posiciones abiertas."""
        if not self.connect(): return []
        try:
            positions_data = self.ib.positions()
            positions = []
            for pos in positions_data:
                positions.append({
                    'ticker': pos.contract.symbol,
                    'qty': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_value': pos.position * pos.avgCost
                })
            return positions
        except Exception as e:
            log.error(f"Error posiciones: {e}")
            return []

    def _resolve_contract(self, ticker: str):
        """Crea y califica el contrato."""
        contract = Stock(ticker, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        return contract

    def place_bracket_order(self, ticker: str, action: str, qty: int, stop_loss_pct: float = 0.03):
        """Ejecuta orden Broker con stop-loss."""
        if not self.connect(): return {'status': 'error', 'msg': 'Broker no conectado'}
        
        try:
            contract = self._resolve_contract(ticker)

            # Obtener precio para el Stop
            ticker_data = self.ib.reqMktData(contract, '', True, False)
            # Esperar síncronamente usando el util de ib_insync
            self.ib.sleep(2) 
            
            last_price = ticker_data.last or ticker_data.close or ticker_data.marketPrice()
            
            if not last_price or last_price == 0:
                return {'status': 'error', 'msg': f'No hay precio para {ticker}'}

            # Definir orden principal (Market)
            main_order = MarketOrder(action, qty)
            main_order.transmit = False

            # Definir orden Stop-Loss
            stop_price = round(last_price * (1 - stop_loss_pct) if action == 'BUY' else last_price * (1 + stop_loss_pct), 2)
            stop_order = StopOrder('SELL' if action == 'BUY' else 'BUY', qty, stop_price)
            stop_order.parentId = main_order.orderId
            stop_order.transmit = True

            # Enviar
            self.ib.placeOrder(contract, main_order)
            self.ib.placeOrder(contract, stop_order)
            
            return {
                'status': 'ok',
                'ticker': ticker,
                'qty': qty,
                'stop_price': stop_price,
                'mode': self.mode
            }
        except Exception as e:
            log.error(f"Error orden: {e}")
            return {'status': 'error', 'msg': str(e)}

    def close_position(self, ticker: str):
        """Cierra posición."""
        if not self.connect(): return {'status': 'error'}
        try:
            positions = self.ib.positions()
            p = next((x for x in positions if x.contract.symbol == ticker), None)
            if not p: return {'status': 'skip'}
            
            contract = self._resolve_contract(ticker)
            order = MarketOrder('SELL' if p.position > 0 else 'BUY', abs(p.position))
            self.ib.placeOrder(contract, order)
            return {'status': 'ok'}
        except Exception as e:
            log.error(f"Error cierre: {e}")
            return {'status': 'error'}

    def cancel_all_orders(self):
        """Kill switch: cancela todas las órdenes activas en IBKR."""
        if not self.connect():
            return False
        try:
            self.ib.reqGlobalCancel()
            log.info("🛑 Kill Switch activado: todas las órdenes canceladas.")
            return True
        except Exception as e:
            log.error(f"Error en cancel_all_orders: {e}", exc_info=True)
            return False
