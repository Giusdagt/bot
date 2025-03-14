# ai_model.py - Modello AI per il trading automatico con ottimizzazione del portafoglio
import os
import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from indicators import TradingIndicators
from data_handler import load_data
from drl_agent import DRLAgent
from gym_trading_env import TradingEnv
from risk_management import RiskManagement
from portfolio_optimization import PortfolioOptimizer

# 📌 Configurazione logging avanzata
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# 📌 Percorsi per il salvataggio dei modelli
MODEL_DIR = Path("/mnt/usb_trading_data/models") if Path(
    "/mnt/usb_trading_data").exists() else Path("D:/trading_data/models")
CLOUD_MODEL_DIR = Path("/mnt/google_drive/trading_models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CLOUD_MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "trading_model.h5"
XGB_MODEL_FILE = MODEL_DIR / "xgb_trading_model.json"

# ===========================
# 🔹 Preprocessing Dati
# ===========================
def preprocess_data(data):
    """Preprocessa i dati per il modello AI."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler

def prepare_lstm_data(data, look_back=60):
    """Prepara i dati per l'input nel modello LSTM."""
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X).reshape(-1, look_back, 1), np.array(y)

def prepare_xgboost_data(data, look_back=60):
    """Prepara i dati per l'input nel modello XGBoost."""
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# ===========================
# 🔹 Creazione e Addestramento dei Modelli AI
# ===========================
def create_lstm_model(input_shape):
    """Crea un modello LSTM compilato."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(X_train, y_train, X_val, y_val):
    """Allena il modello LSTM."""
    model = create_lstm_model((X_train.shape[1], 1))
    early_stop = EarlyStopping(monitor='val_loss', patience=5,
                               restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=32, epochs=50,
              validation_data=(X_val, y_val), callbacks=[early_stop])
    model.save(MODEL_FILE)
    logging.info(f"✅ Modello LSTM salvato in {MODEL_FILE}")
    return model

def create_xgboost_model():
    """Crea un modello XGBoost."""
    return xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,
                            learning_rate=0.1)

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Allena il modello XGBoost."""
    model = create_xgboost_model()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              early_stopping_rounds=10, verbose=True)
    model.save_model(XGB_MODEL_FILE)
    logging.info(f"✅ Modello XGBoost salvato in {XGB_MODEL_FILE}")
    return model

# ===========================
# 🔹 Ottimizzazione del Portafoglio
# ===========================
def optimize_trading_portfolio(data):
    """Ottimizza il portafoglio utilizzando la classe PortfolioOptimizer."""
    optimizer = PortfolioOptimizer(data)
    optimized_allocation = optimizer.optimize()
    logging.info(f"📈 Allocazione ottimizzata: {optimized_allocation}")
    return optimized_allocation

# ===========================
# 🔹 Funzioni per le Previsioni
# ===========================
def load_lstm_model():
    """Carica il modello LSTM."""
    if MODEL_FILE.exists():
        model = load_model(MODEL_FILE)
        logging.info(f"✅ Modello LSTM caricato da {MODEL_FILE}")
        return model
    logging.error(f"❌ Il file del modello LSTM {MODEL_FILE} non esiste.")
    return None

def load_xgboost_model():
    """Carica il modello XGBoost."""
    if XGB_MODEL_FILE.exists():
        model = xgb.XGBRegressor()
        model.load_model(XGB_MODEL_FILE)
        logging.info(f"✅ Modello XGBoost caricato da {XGB_MODEL_FILE}")
        return model
    logging.error(f"❌ Il file del modello XGBoost {XGB_MODEL_FILE} non esiste.")
    return None

# ===========================
# 🔹 Esecuzione del Modello AI
# ===========================
def example_prediction():
    """Esegue una previsione di esempio con i modelli AI."""
    data = load_data()
    scaled_data, _ = preprocess_data(data['close'].values.reshape(-1, 1))

    X_lstm, _ = prepare_lstm_data(scaled_data)
    X_xgb, _ = prepare_xgboost_data(scaled_data)

    lstm_model = load_lstm_model()
    xgb_model = load_xgboost_model()

    if lstm_model:
        lstm_predictions = lstm_model.predict(X_lstm)
        logging.info(f"📊 Previsione LSTM: {lstm_predictions[-1]}")

    if xgb_model:
        xgb_predictions = xgb_model.predict(X_xgb)
        logging.info(f"📊 Previsione XGBoost: {xgb_predictions[-1]}")

    # 🔥 Utilizzo di `os` per ottenere la directory corrente
    current_dir = os.getcwd()
    logging.info(f"📂 Directory corrente: {current_dir}")

    # 🔥 Utilizzo di `pd` per creare un DataFrame di esempio
    df_example = pd.DataFrame({'date': [datetime.now()], 
                               'prediction': [lstm_predictions[-1]]})
    logging.info(f"📋 DataFrame di esempio creato: {df_example}")

    # 🔥 Utilizzo di `RandomForestRegressor` per creare un modello di esempio
    rf_model = RandomForestRegressor(n_estimators=10)
    rf_model.fit(np.array([[0]]), np.array([0]))
    logging.info(f"🌲 Modello RandomForestRegressor creato: {rf_model}")

    optimized_portfolio = optimize_trading_portfolio(data)
    logging.info(f"💰 Portafoglio ottimizzato: {optimized_portfolio}")

if __name__ == "__main__":
    logging.info("🚀 Avvio del modello AI per il trading automatico con ottimizzazione del portafoglio.")
    example_prediction()
