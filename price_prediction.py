import logging
import numpy as np
import polars as pl
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Percorsi per il modello e i dati
MODEL_DIR = Path("/mnt/usb_trading_data/models") if Path("/mnt/usb_trading_data").exists() else Path("D:/trading_data/models")
MODEL_FILE = MODEL_DIR / "lstm_price_prediction.h5"
DATA_FILE = MODEL_DIR / "price_history.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Normalizzatore per i dati di input
scaler = MinMaxScaler(feature_range=(0, 1))

def load_price_data():
    """Carica i dati storici dei prezzi con compressione avanzata."""
    if DATA_FILE.exists():
        logging.info("📥 Caricamento storico prezzi da Parquet...")
        return pl.read_parquet(DATA_FILE)
    logging.warning("⚠️ Nessun dato storico trovato. Fornire dati validi.")
    return None

def save_price_data(price_data):
    """Salva i dati di prezzo in formato Parquet compresso."""
    pl.DataFrame(price_data).write_parquet(DATA_FILE, compression="zstd")
    logging.info("💾 Dati di prezzo salvati con compressione avanzata.")

def create_lstm_model(input_shape):
    """Crea e restituisce un modello LSTM ottimizzato."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(price_data):
    """Allena il modello LSTM sulla serie storica dei prezzi."""
    logging.info("🔄 Avvio dell'addestramento LSTM...")
    
    price_series = np.array(price_data["close"]).reshape(-1, 1)
    price_series = scaler.fit_transform(price_series)
    X, y = [], []
    for i in range(60, len(price_series)):
        X.append(price_series[i-60:i])
        y.append(price_series[i])
    
    X, y = np.array(X), np.array(y)
    model = create_lstm_model((X.shape[1], 1))
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])
    model.save(MODEL_FILE)
    logging.info("✅ Modello LSTM addestrato e salvato con successo.")

def predict_future_prices(recent_prices):
    """Prevede i prezzi futuri utilizzando il modello LSTM."""
    if not MODEL_FILE.exists():
        logging.error("⚠️ Nessun modello trovato. Addestrare prima il modello.")
        return None
    
    model = load_model(MODEL_FILE)
    scaled_input = scaler.transform(np.array(recent_prices).reshape(-1, 1))
    X_pred = np.array([scaled_input[-60:]])
    prediction = model.predict(X_pred)
    return scaler.inverse_transform(prediction)[0][0]  # Restituisce il valore predetto

if __name__ == "__main__":
    historical_prices = load_price_data()
    if historical_prices is not None:
        train_lstm_model(historical_prices)
        last_60_prices = historical_prices["close"][-60:].to_list()
        predicted_price = predict_future_prices(last_60_prices)
        logging.info(f"📈 Prezzo futuro previsto: {predicted_price:.5f}")
