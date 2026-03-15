"""
risk_manager.py
---------------
Motor de gestión de riesgo de nivel institucional.

Todos los parámetros pueden configurarse vía .env.
El gestor valida CADA decisión antes de enviarla al broker.
Si cualquier guardarrail falla, la orden se bloquea y se registra el motivo.
"""

import os
import json
import logging
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# Ruta para persistir el estado del circuit breaker entre reinicios (A4)
CB_STATE_PATH = os.path.join(os.path.dirname(__file__), ".circuit_state.json")


class RiskManager:
    """
    Valida cada decisión de trading contra todos los guardarraíles configurados.
    Todos los parámetros son configurables via variables de entorno o constructor.
    """

    def __init__(
        self,
        max_risk_pct: float = None,
        stop_loss_pct: float = None,
        max_daily_loss_pct: float = None,
        max_positions: int = None,
        min_model_accuracy: float = None,
        min_capital: float = None
    ):
        # --- Parámetros de Riesgo (de .env o valores por defecto conservadores) ---
        self.max_risk_pct        = max_risk_pct        or float(os.getenv("MAX_RISK_PCT",        "0.02"))   # 2% del capital por operación
        self.stop_loss_pct       = stop_loss_pct       or float(os.getenv("STOP_LOSS_PCT",       "0.03"))   # 3% de caída = activar stop
        self.max_daily_loss_pct  = max_daily_loss_pct  or float(os.getenv("MAX_DAILY_LOSS_PCT",  "0.05"))   # -5% diario = Circuit Breaker
        self.max_positions       = max_positions       or int(os.getenv("MAX_POSITIONS",         "3"))      # Máx. posiciones simultáneas
        self.min_model_accuracy  = min_model_accuracy  or float(os.getenv("MIN_MODEL_ACCURACY",  "0.56"))   # Mínimo de WFO accuracy aceptado
        self.min_capital         = min_capital         or float(os.getenv("MIN_CAPITAL",         "50.0"))   # Capital mínimo para operar

        # --- Estado diario (se persiste en disco para sobrevivir reinicios) ---
        self._daily_loss_tracker = {}
        self._circuit_broken = False
        self._circuit_reason = ""

        # Cargar estado persistido (A4): recupera el circuit breaker si el proceso reinició hoy
        self._load_circuit_state()

    # ─────────────────────────────────────────────
    # PÚBLICA: Punto de entrada principal
    # ─────────────────────────────────────────────

    def validate_trade(
        self,
        ticker: str,
        capital: float,
        model_accuracy: float,
        open_positions: list,
        daily_pnl: float = 0.0
    ) -> tuple[bool, str]:
        """
        Valida una potencial operación contra todos los guardarraíles.

        Returns:
            (True, "OK")   → La operación puede ejecutarse
            (False, razón) → Bloqueada, con el motivo específico
        """
        checks = [
            self._check_circuit_breaker(capital, daily_pnl),
            self._check_min_capital(capital),
            self._check_model_accuracy(model_accuracy),
            self._check_max_positions(open_positions, ticker),
        ]

        for passed, reason in checks:
            if not passed:
                log.warning(f"🛡️ Trade BLOQUEADO [{ticker}]: {reason}")
                return False, reason

        log.info(f"✅ Trade APROBADO [{ticker}] — Todos los guardarraíles superados.")
        return True, "OK"

    def calculate_position_size(self, capital: float, price: float, model_accuracy: float = None, win_loss_ratio: float = 1.5) -> int:
        """
        Kelly Criterion / Fixed Fractional Position Sizing:
        Calcula el tamaño de la posición basándose en el nivel de certidumbre del modelo (Kelly)
        amortiguado de forma conservadora (Half-Kelly).
        
        Nº de acciones = (Capital * Riesgo_Dinámico) / (Precio * Stop_Loss_pct)
        """
        if price <= 0 or capital < self.min_capital:
            return 0

        # Fractional Kelly (Half-Kelly)
        if model_accuracy is not None and model_accuracy > 0.5:
            # Fómula de Kelly: W - [(1-W)/R]
            kelly_f = model_accuracy - ((1 - model_accuracy) / win_loss_ratio)
            half_kelly = max(0.01, kelly_f / 2)
            
            # Restringido por el riesgo máximo fijo de seguridad
            dynamic_risk_pct = min(self.max_risk_pct, half_kelly)
            log.info(f"🧠 Asignación Cuantitativa: Kelly_F={kelly_f:.2%}, Half={half_kelly:.2%}, Aplicado={dynamic_risk_pct:.2%}")
        else:
            dynamic_risk_pct = self.max_risk_pct

        risk_amount = capital * dynamic_risk_pct
        shares = risk_amount / (price * self.stop_loss_pct)
        shares_int = max(1, int(shares))

        # Rebalanceo: Diversificar para no superar más del 40% del capital real total en 1 activo
        max_position_value = capital * 0.40
        if shares_int * price > max_position_value:
            shares_int = max(1, int(max_position_value / price))

        log.info(
            f"📐 Position Size: {shares_int} acciones | "
            f"Capital en riesgo: {risk_amount:.2f}€ | "
            f"Precio: {price:.2f} | Stop: {self.stop_loss_pct:.1%}"
        )
        return shares_int

    def get_risk_summary(self, capital: float) -> dict:
        """Devuelve un resumen de los parámetros de riesgo actuales para la UI."""
        return {
            'Modo': '🔴 LIVE' if os.getenv("LIVE_TRADING", "false").lower() == "true" else '🟡 PAPER',
            'Capital Disponible': f"{capital:.2f} €",
            'Riesgo por Operación': f"{self.max_risk_pct:.1%} = {capital * self.max_risk_pct:.2f} €",
            'Stop-Loss por Defecto': f"{self.stop_loss_pct:.1%}",
            'Circuit Breaker Diario': f"{self.max_daily_loss_pct:.1%} = {capital * self.max_daily_loss_pct:.2f} €",
            'Precisión Mín. Modelo': f"{self.min_model_accuracy:.1%}",
            'Posiciones Máx.': str(self.max_positions),
            'Circuit Activo': '⚠️ SÍ — ' + self._circuit_reason if self._circuit_broken else '✅ NO'
        }

    # ─────────────────────────────────────────────
    # PRIVADAS: Guardarraíles individuales
    # ─────────────────────────────────────────────

    def _check_circuit_breaker(self, capital: float, daily_pnl: float) -> tuple[bool, str]:
        """Bloquea el sistema si las pérdidas del día superan el límite."""
        if self._circuit_broken:
            return False, f"Circuit Breaker activo: {self._circuit_reason}"

        max_daily_loss = capital * self.max_daily_loss_pct
        if daily_pnl < 0 and abs(daily_pnl) >= max_daily_loss:
            self._circuit_broken = True
            self._circuit_reason = f"Pérdida diaria de {daily_pnl:.2f}€ supera el límite de -{max_daily_loss:.2f}€"
            self._persist_circuit_state()  # A4: persistir antes de retornar
            return False, self._circuit_reason

        return True, "OK"

    # ─────────────────────────────────────────────
    # Persistencia del Circuit Breaker (A4 - Security Fix)
    # ─────────────────────────────────────────────

    def _persist_circuit_state(self):
        """Guarda el estado del circuit breaker en disco para sobrevivir reinicios."""
        state = {
            'broken': self._circuit_broken,
            'reason': self._circuit_reason,
            'date': date.today().isoformat()
        }
        try:
            with open(CB_STATE_PATH, 'w') as f:
                json.dump(state, f)
            log.info(f"⚠️ Circuit Breaker persistido en disco: {self._circuit_reason}")
        except Exception as e:
            log.error(f"No se pudo persistir el circuit breaker: {e}")

    def _load_circuit_state(self):
        """Recupera el estado del circuit breaker de disco (solo si es del día de hoy)."""
        if not os.path.exists(CB_STATE_PATH):
            return
        try:
            with open(CB_STATE_PATH, 'r') as f:
                s = json.load(f)
            if s.get('date') == date.today().isoformat():
                self._circuit_broken = s.get('broken', False)
                self._circuit_reason = s.get('reason', '')
                if self._circuit_broken:
                    log.warning(
                        f"⚠️ Circuit Breaker restaurado desde disco — sigue activo hoy: {self._circuit_reason}"
                    )
            else:
                # Estado de otro día: eliminar fichero obsoleto
                os.remove(CB_STATE_PATH)
        except Exception as e:
            log.error(f"Error al cargar estado del circuit breaker: {e}")

    def _check_min_capital(self, capital: float) -> tuple[bool, str]:
        """Bloquea si el capital es demasiado bajo para operar de forma segura."""
        if capital < self.min_capital:
            return False, f"Capital insuficiente: {capital:.2f}€ < mínimo {self.min_capital:.2f}€"
        return True, "OK"

    def _check_model_accuracy(self, accuracy: float) -> tuple[bool, str]:
        """Bloquea si el modelo no supera el umbral mínimo de precisión WFO."""
        if accuracy < self.min_model_accuracy:
            return False, (
                f"Precisión del modelo ({accuracy:.2%}) insuficiente. "
                f"Mínimo requerido: {self.min_model_accuracy:.2%}"
            )
        return True, "OK"

    def _check_max_positions(self, open_positions: list, ticker: str) -> tuple[bool, str]:
        """Bloquea si ya tenemos el máximo de posiciones abiertas."""
        active = [p for p in open_positions if p.get('qty', 0) != 0]

        # Si ya tenemos una posición en este ticker, no bloqueamos (es una señal de cierre)
        already_in = any(p['ticker'] == ticker for p in active)
        if already_in:
            return True, "OK"  # Es una operación sobre posición existente

        if len(active) >= self.max_positions:
            return False, (
                f"Máximo de posiciones alcanzado: {len(active)}/{self.max_positions}. "
                f"Cierra una posición antes de abrir {ticker}."
            )
        return True, "OK"

    def reset_daily_circuit(self):
        """Reinicia el circuit breaker al inicio de cada jornada y elimina el estado persistido."""
        self._circuit_broken = False
        self._circuit_reason = ""
        if os.path.exists(CB_STATE_PATH):
            try:
                os.remove(CB_STATE_PATH)
            except Exception as e:
                log.error(f"No se pudo eliminar el fichero de circuit breaker: {e}")
        log.info("🔄 Circuit Breaker reiniciado para nueva jornada.")
