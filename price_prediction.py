"""
Modulo per la previsione dei prezzi tramite LSTM.
- Utilizza data_handler.py per recuperare storici normalizzati con indicatori
- Supporta più di 300 asset contemporaneamente.
- Gli asset possono essere caricati da preset_asset.json
(se attivo in `data_loader.py`)
oppure selezionati dinamicamente.
"""


import logging
from pathlib import Path
import numpy as np
import polars as pl
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from data_handler import get_normalized_market_data, get_available_assets

MODEL_DIR = Path("D:/trading_data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = MODEL_DIR / "lstm_memory.parquet"
SEQUENCE_LENGTH = 50
BATCH_SIZE = 32


class PricePredictionModel:
    """
    Modello LSTM ultra-ottimizzato per previsioni di prezzo:
    - Allenamento continuo senza accumulo di dati inutili.
    - Compressione avanzata della memoria.
    - Salvataggio intelligente (non cresce su disco nel tempo).
    """

    def __init__(self):
        """
        Inizializza il modello di previsione dei prezzi:
        - Crea uno scaler per normalizzare i dati.
        - Carica la tabella di memoria compressa per tutti gli asset.
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.memory_df = self.load_memory_table()

    def get_model_file(self, asset):
        """
        Restituisce il percorso del file modello per un asset specifico.
        """
        return MODEL_DIR / f"lstm_model_{asset}.h5"

    def load_memory_table(self):
        """
        Carica la tabella completa di memoria compressa per tutti gli asset.
        Se il file non esiste, inizializza una tabella vuota.
        """
        if MEMORY_FILE.exists():
            return pl.read_parquet(MEMORY_FILE)
        return pl.DataFrame({"asset": [], "compressed_memory": []})

    def load_memory(self, asset):
        """
        Recupera la memoria compressa per un asset specifico.
        Se non esiste, restituisce un array vuoto iniziale (tutto a zero).
        """
        try:
            row = self.memory_df.filter(pl.col("asset") == asset)
            if row.is_empty():
                return np.zeros((SEQUENCE_LENGTH, 1), dtype=np.float32)
            return np.frombuffer(
                row["compressed_memory"][0],
                dtype=np.float32
            ).reshape(SEQUENCE_LENGTH, 1)
        except Exception:
            return np.zeros((SEQUENCE_LENGTH, 1), dtype=np.float32)

    def save_memory(self, asset, new_data):
        """
        Salva i nuovi dati compressi per un asset:
        - Comprimi i dati come media.
        - Aggiorna solo se ci sono variazioni effettive
        rispetto alla memoria esistente.
        - Scrive in un file Parquet ultra compresso.
        """
        
        compressed =(
            np.mean(new_data, axis=0, keepdims=True).astype(np.float32)
        )
        existing = self.memory_df.filter(pl.col("asset") == asset)

        if not existing.is_empty():
            if np.array_equal(
                np.frombuffer(
                    existing["compressed_memory"][0], dtype=np.float32
                ),
                compressed
            ):
                return
            self.memory_df = self.memory_df.filter(pl.col("asset") != asset)

        new_row = pl.DataFrame({
            "asset": [asset],
            "compressed_memory": [compressed.tobytes()]
        })
        self.memory_df = pl.concat([self.memory_df, new_row])
        self.memory_df.write_parquet(MEMORY_FILE, compression="zstd")

    def build_lstm_model(self):
        """
        Costruisce un modello LSTM leggero e ottimizzato:
        - Architettura con due layer LSTM e un Dense finale.
        - Precisione ridotta (float16) per consumi ridotti.
        - Compilato con ottimizzatore 'adam' e perdita 'mean_squared_error'.
        """
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

    def load_or_create_model(self, asset):
        """
        Carica il modello salvato per l'asset, se esiste.
        Altrimenti crea un nuovo modello LSTM da zero.
        """
        model_file = self.get_model_file(asset)
        if model_file.exists():
            return load_model(model_file)
        return self.build_lstm_model()

    def preprocess_data(self, raw_data):
        """
        Normalizza i dati grezzi usando MinMaxScaler.
        Converte i dati in una forma adatta all'LSTM.
        """
        raw_data = np.array(raw_data).reshape(-1, 1)
        return self.scaler.fit_transform(raw_data)

    def train_model(self, asset, raw_data):
        """
        Allena il modello per un asset specifico:
        - Prepara i dati di input (sequenze e target).
        - Applica early stopping per evitare overfitting.
        - Salva il modello e la memoria compressa.
        """
        if len(raw_data) <= SEQUENCE_LENGTH:
            return

        model = self.load_or_create_model(asset)
        memory = self.load_memory(asset)
        data = self.preprocess_data(raw_data)

        x, y = [], []
        for i in range(len(data) - SEQUENCE_LENGTH):
            x.append(data[i:i+SEQUENCE_LENGTH])
            y.append(data[i+SEQUENCE_LENGTH])
        x, y = np.array(x), np.array(y)

        early_stop = EarlyStopping(
            monitor="loss", patience=3, restore_best_weights=True
        )
        model.fit(
            x, y, epochs=10, batch_size=BATCH_SIZE, verbose=1, callbacks=[early_stop]
        )
        model.save(self.get_model_file(asset))
        self.save_memory(asset, raw_data[-SEQUENCE_LENGTH:])

    def predict_price(self, asset):
        """
        Prevede il prossimo prezzo per un asset:
        - Preleva gli ultimi dati normalizzati.
        - Calcola la previsione usando il modello LSTM.
        - Inverte la normalizzazione e restituisce il prezzo stimato.
        """
        model = self.load_or_create_model(asset)
        raw_data = get_normalized_market_data(asset)["close"].to_numpy()

        if len(raw_data) < SEQUENCE_LENGTH:
            logging.warning(f"⚠️ Dati insufficienti per {asset}")
            return None

        data = self.preprocess_data(raw_data)
        last_sequence = data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
        prediction = model.predict(last_sequence)[0][0]
        predicted_price = self.scaler.inverse_transform([[prediction]])[0][0]

        logging.info(f"📊 Prezzo previsto per {asset}: {predicted_price:.5f}")
        return predicted_price


if __name__ == "__main__":
    """
    Esegue il training e la previsione su tutti gli asset disponibili:
    - Recupera gli asset da `get_available_assets`.
    - Addestra il modello per ciascun asset.
    - Prevede il prezzo futuro.
    """
    model = PricePredictionModel()
    all_assets = get_available_assets()  # Utilizza gli asset disponibili
    for asset in all_assets:
        market_data = get_normalized_market_data(asset)["close"].to_numpy()
        model.train_model(asset, market_data)
        model.predict_price(asset)
