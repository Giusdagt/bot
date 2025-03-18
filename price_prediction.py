import logging
from pathlib import Path
import numpy as np
import polars as pl
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from data_handler import get_normalized_market_data

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Percorsi dei modelli e dei dati
MODEL_DIR = Path("D:/trading_data/models")
MODEL_FILE = MODEL_DIR / "lstm_model.h5"
MEMORY_FILE = MODEL_DIR / "lstm_memory.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Configurazione avanzata dell'LSTM
SEQUENCE_LENGTH = 50
BATCH_SIZE = 32


class PricePredictionModel:
    """
    Modello LSTM per la previsione dei prezzi con ottimizzazione avanzata.
    - Compressione ultra-efficiente della memoria.
    - Allenamento continuo senza accumulo di dati inutili.
    - Salvataggio adattivo per evitare spreco di risorse.
    """

    def __init__(self, asset="EURUSD"):
        self.asset = asset
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.memory = self.load_memory()
        self.model = self.load_or_create_model()

    def load_memory(self):
        """Carica/inizializza/memoria compressa per l'allenamento continuo."""
        if MEMORY_FILE.exists():
            logging.info("📥 Caricamento memoria LSTM da Parquet...")
            return pl.read_parquet(MEMORY_FILE)["compressed_memory"].to_numpy()
        else:
            logging.info("🔄 Creazione nuova memoria LSTM...")
            return np.zeros((SEQUENCE_LENGTH, 1), dtype=np.float32)

    def save_memory(self, new_data):
        """Aggiorna la memoria senza accumulare dati inutili."""
        compressed_memory = np.mean(new_data, axis=0, keepdims=True)
        df = pl.DataFrame({"compressed_memory": compressed_memory.flatten()})
        df.write_parquet(MEMORY_FILE, compression="zstd")
        logging.info("💾 Memoria LSTM aggiornata e compressa con Zstd.")

    def load_or_create_model(self):
        """Carica il modello LSTM esistente o ne crea uno nuovo."""
        if MODEL_FILE.exists():
            logging.info("📥 Caricamento modello LSTM esistente...")
            return load_model(MODEL_FILE)
        else:
            logging.info("🔧 Creazione nuovo modello LSTM...")
            return self.build_lstm_model()

    def build_lstm_model(self):
        """Costruisce un modello LSTM ottimizzato."""
        model = Sequential([
            LSTM(
                64, activation="tanh", return_sequences=True, dtype="float16"
            ),
            Dropout(0.2),
            LSTM(
                32, activation="tanh", return_sequences=False, dtype="float16"
            ),
            Dense(1, activation="linear", dtype="float16")
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def preprocess_data(self, raw_data):
        """Pre-elabora i dati e normalizza per il training."""
        raw_data = np.array(raw_data).reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(raw_data)
        return scaled_data

    def train_model(self, new_data):
        """Allena il modello LSTM in modo intelligente senza accumulo di dati inutili."""
        data = self.preprocess_data(new_data)
        X, y = [], []

        for i in range(len(data) - SEQUENCE_LENGTH):
            X.append(data[i:i+SEQUENCE_LENGTH])
            y.append(data[i+SEQUENCE_LENGTH])

        X, y = np.array(X), np.array(y)

        # 🔥 Se il modello esiste già, carica i pesi per NON perdere dati precedenti
        if MODEL_FILE.exists():
            logging.info("📥 Caricamento pesi esistenti nel modello LSTM...")
            self.model.load_weights(MODEL_FILE)

        # ✅ Configurazione `EarlyStopping` per un allenamento ultra-efficiente
        early_stop = EarlyStopping(
            monitor="loss", patience=3, restore_best_weights=True
        )

        # 🔥 Allenamento ottimizzato
        self.model.fit(
            X, y, epochs=10, batch_size=BATCH_SIZE, verbose=1, callbacks=[early_stop]
        )

        # ✅ Salvataggio ottimizzato dei pesi (senza riscrivere tutto il modello)
        self.model.save_weights(MODEL_FILE, overwrite=True)

        # ✅ Aggiorna la memoria compressa senza accumulo
        self.save_memory(new_data)

        logging.info("✅ Modello LSTM allenato e ottimizzato con `EarlyStopping`.")

    def predict_price(self):
        """Prevede il prezzo futuro basandosi sugli ultimi dati di mercato."""
        raw_data = get_normalized_market_data(self.asset)["close"].to_numpy()
        data = self.preprocess_data(raw_data)
        last_sequence = data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)

        prediction = self.model.predict(last_sequence)[0][0]
        predicted_price = self.scaler.inverse_transform([[prediction]])[0][0]

        logging.info(
            f"📊 Prezzo previsto per {self.asset}: {predicted_price:.5f}"
        )
        return predicted_price


if __name__ == "__main__":
    predictor = PricePredictionModel()
    market_data = get_normalized_market_data(predictor.asset)["close"].to_numpy()
    predictor.train_model(market_data)
    future_price = predictor.predict_price()
