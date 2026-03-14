import os
import joblib
import json
import hashlib
import logging
import pandas as pd
import numpy as np
import ta
import asyncio
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel

log = logging.getLogger(__name__)

class MLEngine:
    """
    Motor de Machine Learning con soporte para:
    - Persistencia (Joblib + JSON metadata)
    - Smart Retraining (Histeresis + Edad)
    - Feature Engineering Avanzado (Bollinger, ATR, OBV, ADX)
    """
    
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

    def __init__(self, df: pd.DataFrame = None):
        """
        Inicializa el motor. 
        df es opcional si solo se va a cargar un modelo existente.
        """
        if df is not None:
            if isinstance(df, pd.Series):
                self.df = df.to_frame(name='Close')
            else:
                self.df = df.copy()
                if 'Close' not in self.df.columns and 'Adj Close' in self.df.columns:
                     self.df.rename(columns={'Adj Close': 'Close'}, inplace=True)
                elif len(self.df.columns) == 1:
                    self.df.columns = ['Close']
        else:
            self.df = None

        self.features = None
        self.target = None
        self.trained_models = {}
        
        if not os.path.exists(self.MODELS_DIR):
            os.makedirs(self.MODELS_DIR, exist_ok=True)

    def create_features_and_target(self, target_horizon: int = 5):
        """
        Genera un set robusto de indicadores técnicos.
        """
        if self.df is None:
            raise ValueError("No hay datos cargados para generar features.")
            
        df = self.df.copy()

        # --- 1. INDICADORES BÁSICOS (TA) ---
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # --- 2. VOLATILIDAD AVANZADA ---
        # Bandas de Bollinger
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High_Dist'] = (bb.bollinger_hband() - df['Close']) / df['Close']
        df['BB_Low_Dist'] = (df['Close'] - bb.bollinger_lband()) / df['Close']
        df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        
        # ATR (Average True Range) - Usamos una estimación si no hay High/Low
        if 'High' in df.columns and 'Low' in df.columns:
            df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
        else:
            df['ATR'] = df['Close'].rolling(window=14).std()
        df['ATR_Pct'] = df['ATR'] / df['Close']
        
        # --- 3. VOLUMEN (Si existe) ---
        if 'Volume' in df.columns:
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
            df['OBV_ROC'] = df['OBV'].pct_change(5)
            
        # --- 4. TENDENCIA Y MOMENTUM ---
        df['SMA_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()
        
        df['Dist_SMA_20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        df['Dist_SMA_200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']

        # Retornos pasados
        for lag in [1, 2, 3, 5, 10]:
            df[f'Ret_Lag_{lag}'] = df['Close'].pct_change(lag)

        # Calendario
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month

        # --- 5. TARGET DINÁMICO (Basado en Volatilidad - ATR) ---
        # Subimos la valla a 0.75 * ATR para ser más selectivos y buscar señales claras.
        target_mult = 0.75 
        df['Target_Threshold'] = df['ATR'] * target_mult / df['Close']
        df[f'Fut_Ret_{target_horizon}'] = df['Close'].pct_change(target_horizon).shift(-target_horizon)
        
        # Clase 1 si el retorno supera la "valla" de volatilidad
        df['Target'] = (df[f'Fut_Ret_{target_horizon}'] > df['Target_Threshold']).astype(int)

        df.dropna(inplace=True)

        features_cols = sorted([c for c in df.columns if c not in ['Target', f'Fut_Ret_{target_horizon}', 'Close', 'High', 'Low', 'Volume', 'Adj Close', 'OBV', 'Target_Threshold']])
        self.features = df[features_cols]
        self.target = df['Target']
        self.df_processed = df

        return self.features, self.target

    def _get_model_paths(self, ticker: str, model_name: str):
        base = f"{model_name}_{ticker.replace('-','_')}"
        return os.path.join(self.MODELS_DIR, f"{base}.joblib"), os.path.join(self.MODELS_DIR, f"{base}_meta.json")

    # ─────────────────────────────────────────────
    # Integridad de modelos (C2 - Security Fix)
    # ─────────────────────────────────────────────

    def _get_model_hash(self, model_path: str) -> str:
        """Calcula el SHA-256 de un fichero de modelo."""
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _verify_model_hash(self, model_path: str, meta: dict) -> bool:
        """Verifica la integridad del modelo comparando el hash guardado."""
        expected_hash = meta.get('extra', {}).get('model_sha256')
        if not expected_hash:
            log.warning("Modelo legacy sin hash guardado. Se recomienda forzar reentrenamiento.")
            return True  # Compatibilidad hacia atrás — solo advertencia
        actual_hash = self._get_model_hash(model_path)
        if actual_hash != expected_hash:
            log.error(
                f"❌ INTEGRIDAD COMPROMETIDA: el hash del modelo no coincide.\n"
                f"  Esperado: {expected_hash}\n"
                f"  Real:     {actual_hash}\n"
                f"  Fichero:  {model_path}"
            )
            return False
        log.info(f"✅ Hash del modelo verificado correctamente para {model_path}")
        return True

    def save_persistence(self, ticker: str, model_name: str, accuracy: float, metadata: dict = None):
        """Guarda el modelo y sus metadatos con hash SHA-256 de integridad."""
        model_path, meta_path = self._get_model_paths(ticker, model_name)
        joblib.dump(self.trained_models[model_name], model_path)

        # Calcular y guardar hash SHA-256 del modelo serializado
        model_sha256 = self._get_model_hash(model_path)

        extra = metadata or {}
        extra['model_sha256'] = model_sha256

        info = {
            'ticker': ticker,
            'model_name': model_name,
            'accuracy': accuracy,
            'train_date': datetime.now().isoformat(),
            'n_samples': len(self.features),
            'extra': extra
        }
        with open(meta_path, 'w') as f:
            json.dump(info, f, indent=4)
        log.info(f"Modelo persistido para {ticker} en {model_path} (SHA-256: {model_sha256[:12]}...)")

    def load_persistence(self, ticker: str, model_name: str) -> dict:
        """Carga un modelo y sus metadatos — verifica integridad SHA-256 antes de deserializar."""
        model_path, meta_path = self._get_model_paths(ticker, model_name)
        if os.path.exists(model_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)

                # ── VERIFICACIÓN DE INTEGRIDAD (C2) ──────────────────────────
                # Se verifica el hash ANTES de llamar a joblib.load() para
                # evitar deserialización de ficheros manipulados/comprometidos.
                if not self._verify_model_hash(model_path, meta):
                    log.error(f"Modelo de {ticker} rechazado por integridad comprometida. Forzando reentrenamiento.")
                    return None  # Forzar reentrenamiento limpio

                self.trained_models[model_name] = joblib.load(model_path)

                # Restaurar lista de features seleccionadas (con fallback para modelos viejos)
                saved_features = meta.get('extra', {}).get('features', [])
                if not saved_features:
                    log.warning(f"Modelo legacy detectado para {ticker}. Reentrenamiento recomendado.")
                    self.selected_features_list = None
                else:
                    self.selected_features_list = sorted(saved_features)
                return meta
            except Exception as e:
                log.error(f"Error cargando persistencia de {ticker}: {e}", exc_info=True)
        return None

    def train_and_evaluate(self, model_name: str = 'XGBoost', n_splits: int = 5, ticker: str = "UNKNOWN"):
        """
        Entrena el modelo con búsqueda de hiperparámetros (RandomizedSearch) 
        y selección de características.
        """
        if self.features is None or self.target is None:
            self.create_features_and_target()

        # 1. Búsqueda de Hiperparámetros (RandomizedSearch)
        # Solo lo hacemos si es un entrenamiento nuevo o necesario
        if model_name == 'XGBoost':
            # Calcular balance de clases para scale_pos_weight
            pos_ratio = (self.target == 0).sum() / (self.target == 1).sum() if (self.target == 1).sum() > 0 else 1
            
            base_model = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=pos_ratio)
            param_grid = {
                'n_estimators': [100, 200, 400],
                'max_depth': [3, 4, 6, 8],
                'learning_rate': [0.01, 0.03, 0.07, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2]
            }
        else:
            base_model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }

        # 0. Cálculo de Pesos Temporales (Exponential Decay)
        # Priorizamos datos de los últimos 2 años (1.0) y bajamos hasta 0.5 para el pasado.
        n_samples = len(self.features)
        # 504 días ~ 2 años de trading
        decay_period = 504 
        weights = np.ones(n_samples)
        for i in range(n_samples):
            # i=0 es el dato más antiguo, i=n-1 es el más reciente
            dist_from_now = n_samples - 1 - i
            if dist_from_now > decay_period:
                # Decaimiento suave hasta 0.5
                weights[i] = max(0.5, 1.0 - (dist_from_now - decay_period) / (n_samples * 2))
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # Realizamos una búsqueda ligera (n_iter=10) para no relentizar demasiado
        search = RandomizedSearchCV(
            base_model, param_distributions=param_grid, 
            n_iter=10, cv=tscv, scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        # Pasar pesos al fit de la búsqueda
        search.fit(self.features, self.target, sample_weight=weights)
        best_model = search.best_estimator_
        log.info(f"Mejores parámetros para {ticker}: {search.best_params_}")

        # 2. Selección de Características (Eliminar ruido)
        selector = SelectFromModel(best_model, prefit=True, threshold='median')
        feature_mask = selector.get_support()
        selected_features = self.features.columns[feature_mask].tolist()
        log.info(f"Features seleccionadas para {ticker}: {len(selected_features)}/{len(self.features.columns)}")
        
        # Guardamos la lista de features para futuras predicciones
        self.selected_features_list = selected_features
        X_reduced = self.features[selected_features]

        # 3. Evaluación Final (WFO) sobre features reducidas
        fold_accuracies = []
        last_y_test, last_y_pred = None, None

        for train_index, test_index in tscv.split(X_reduced):
            X_train, X_test = X_reduced.iloc[train_index], X_reduced.iloc[test_index]
            y_train, y_test = self.target.iloc[train_index], self.target.iloc[test_index]
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            fold_accuracies.append(accuracy_score(y_test, y_pred))
            last_y_test, last_y_pred = y_test, y_pred

        avg_acc = np.mean(fold_accuracies)
        
        # Entrenamiento final con todo el set reducido y pesos
        reduced_weights = weights # El peso es por fila, no cambia al reducir columnas
        best_model.fit(X_reduced, self.target, sample_weight=reduced_weights)
        self.trained_models[model_name] = best_model
        
        # Persistencia automática con metadatos de búsqueda
        self.save_persistence(ticker, model_name, avg_acc, metadata={'features': selected_features, 'params': search.best_params_})

        return {
            'model': best_model,
            'accuracy': avg_acc,
            'report': classification_report(last_y_test, last_y_pred, output_dict=True)
        }

    def load_or_train(self, ticker: str, model_name: str = 'XGBoost', force_retrain: bool = False):
        """
        Lógica inteligente de persistencia: Shadow Training + Histeresis + Edad.
        """
        meta = self.load_persistence(ticker, model_name)
        
        if meta and not force_retrain:
            train_date = datetime.fromisoformat(meta['train_date'])
            age_days = (datetime.now() - train_date).days
            old_acc = meta['accuracy']
            
            # 1. Si es muy reciente (<7 días), lo usamos tal cual
            # REFINAMIENTO: Si es un modelo legacy (sin features), forzamos reentrenamiento 
            # para evitar el error de mismatch.
            if age_days < 7 and self.selected_features_list is not None:
                log.info(f"Cargado modelo reciente ({age_days}d) para {ticker}. Acc: {old_acc:.2%}")
                return {'status': 'loaded', 'accuracy': old_acc, 'meta': meta}
            
            # 2. Si es viejo, entrenamos un candidato (Shadow)
            log.info(f"Modelo de {ticker} es viejo ({age_days}d). Entrenando candidato...")
            res_new = self.train_and_evaluate(model_name, ticker=ticker)
            new_acc = res_new['accuracy']
            
            # 3. Histeresis: Solo guardamos si mejora significativamente (>2%)
            if new_acc > old_acc + 0.02:
                log.info(f"Candidato mejora a previa ({new_acc:.2%} vs {old_acc:.2%}). Sustituyendo.")
                return {**res_new, 'status': 'updated'}
            else:
                log.info(f"Candidato no mejora lo suficiente ({new_acc:.2%} vs {old_acc:.2%}). Mantenemos el anterior.")
                self.load_persistence(ticker, model_name) # Recargar el viejo en trained_models
                return {'status': 'kept_old', 'accuracy': old_acc, 'meta': meta}
        
        # Si no existe o forzamos reentrenamiento
        log.info(f"Entrenamiento inicial/forzado para {ticker}...")
        res = self.train_and_evaluate(model_name, ticker=ticker)
        return {**res, 'status': 'new'}

    def generate_signals(self, model_name: str = 'XGBoost'):
        """Genera señales usando las features seleccionadas."""
        if model_name not in self.trained_models:
             raise ValueError("Modelo no cargado/entrenado.")
        
        # Obtener las features que el modelo espera
        # Si no hay lista (modelo legacy), usamos todas las actuales (esto fallará si cambiaron, lo cual es correcto)
        features_to_use = getattr(self, 'selected_features_list', None)
        if features_to_use is None:
            features_to_use = self.features.columns.tolist()
        
        # Asegurar orden consistente
        features_to_use = sorted(features_to_use)
        
        X_pred = self.features[features_to_use]
        return pd.Series(self.trained_models[model_name].predict(X_pred), 
                         index=self.features.index, name='Signal')
