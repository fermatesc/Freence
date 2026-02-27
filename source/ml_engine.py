import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report


class MLEngine:
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa con el DataFrame de precios histéricos.
        Se espera que el index del DataFrame sea DatetimeIndex y contenga al menos la columna de precio
        o que df sea una serie si solo pasamos Close/Adj Close.
        """
        # Si recibimos una serie (solo precios de cierre), la convertimos a DataFrame
        if isinstance(df, pd.Series):
            self.df = df.to_frame(name='Close')
        else:
            self.df = df.copy()
            # Asegurar que tenemos una columna llamada 'Close' si hay un DataFrame multivariado
            if 'Close' not in self.df.columns and 'Adj Close' in self.df.columns:
                 self.df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            elif len(self.df.columns) == 1:
                self.df.columns = ['Close']

        self.features = None
        self.target = None
        self.models = {
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        }
        self.trained_models = {}

    def create_features_and_target(self, target_horizon: int = 5):
        """
        Genera indicadores técnicos y establece el Target para la clasificación.
        target_horizon: Días en el futuro a predecir.
        """
        df = self.df.copy()

        # 1. Indicadores Técnicos (Usando ta)
        # Momentum
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        # Trend
        macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        
        # Volatilidad (Requiere High, Low, Close idealmente, pero estimamos con Close si no hay más)
        # Si solo tenemos Close, usaremos desviaciones estándar móviles como proxi de volatilidad
        df['Vol_20'] = df['Close'].rolling(window=20).std()
        
        # Medias Móviles
        df['SMA_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
        
        # Distancia a las Medias Móviles (normalizada)
        df['Dist_SMA_20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

        # Retornos pasados (Lags)
        for lag in [1, 2, 3, 5, 10]:
            df[f'Ret_Lag_{lag}'] = df['Close'].pct_change(lag)

        # 2. Variables de Calendario
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month

        # 3. Definición del Target (Variable Objetivo)
        # Calculamos el retorno futuro a N días
        df[f'Fut_Ret_{target_horizon}'] = df['Close'].pct_change(target_horizon).shift(-target_horizon)
        # Clasificación binaria: 1 si sube el precio (más que comisiones base), 0 si baja o se mantiene
        # Añadimos un pequeño umbral de 0.001 (0.1%) para superar mínimamente comisiones
        df['Target'] = (df[f'Fut_Ret_{target_horizon}'] > 0.001).astype(int)

        # Limpiar NaNs provocados por indicadores y shifts
        df.dropna(inplace=True)

        # Guardar en la instancia
        features_cols = [c for c in df.columns if c not in ['Target', f'Fut_Ret_{target_horizon}', 'Close']]
        self.features = df[features_cols]
        self.target = df['Target']
        self.df_processed = df

        return self.features, self.target

    def train_and_evaluate(self, model_name: str = 'XGBoost', n_splits: int = 5):
        """
        Entrena y evalúa usando Walk-Forward Optimization (TimeSeriesSplit) para 
        asegurar una métrica de precisión realista, sin data leakage.
        """
        if self.features is None or self.target is None:
            raise ValueError("Debes llamar a create_features_and_target() primero.")

        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Modelo {model_name} no soperte. Usa: {list(self.models.keys())}")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_accuracies = []
        
        # Guardaremos las métricas del último fold para mantener la estructura de reporte
        last_y_test = None
        last_y_pred = None

        print(f"--- Evaluando {model_name} con {n_splits}-Fold WFO ---")
        
        for fold, (train_index, test_index) in enumerate(tscv.split(self.features)):
            # CHECK MATH: Nos aseguramos de que el final del set de entrenamiento 
            # sea ESTRICTAMENTE ANTERIOR al principio del set de prueba para evitar fugas.
            assert max(train_index) < min(test_index), "¡Data Leakage Detectado en TimeSeriesSplit!"
            
            X_train, X_test = self.features.iloc[train_index], self.features.iloc[test_index]
            y_train, y_test = self.target.iloc[train_index], self.target.iloc[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(acc)
            # print(f"Fold {fold+1}: Exactitud = {acc:.2%}")
            
            # Reasignar para conservar el último
            last_y_test = y_test
            last_y_pred = y_pred

        # Finalmente, reentrenamos el modelo con la totalidad de los datos (lo usaremos para generar señales futuras)
        model.fit(self.features, self.target)
        self.trained_models[model_name] = model

        avg_acc = np.mean(fold_accuracies)
        report = classification_report(last_y_test, last_y_pred, output_dict=True)

        print(f"Precisión Promedio Walk-Forward ({n_splits} splits): {avg_acc:.2%}")
        
        return {
            'model': model,
            'accuracy': avg_acc, # Usamos el WFO Average Accuracy como métrica global
            'report': report,    # Mantenemos el reporte del último fold para desglosar el F1
            'y_test': last_y_test,  
            'y_pred': last_y_pred,
            'test_index': self.features.index[-len(last_y_pred):]
        }

    def generate_signals(self, model_name: str = 'XGBoost'):
        """
        Genera todas las señales para el data frame procesado usando el modelo entrenado.
        Devuelve una Serie de pandas con las predicciones.
        """
        if model_name not in self.trained_models:
             raise ValueError(f"El modelo {model_name} no ha sido entrenado. Llama a train_and_evaluate() primero.")
             
        model = self.trained_models[model_name]
        predictions = model.predict(self.features)
        
        # Crear una serie de señales (1 = Comprar/Mantener, 0 = Vender/Neutral)
        signals = pd.Series(predictions, index=self.features.index, name='Signal')
        return signals

