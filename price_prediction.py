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
from market_fingerprint import get_embedding_for_symbol
from smart_features import apply_all_market_structure_signals

MODEL_DIR = Path("D:/trading_data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = MODEL_DIR / "lstm_memory.parquet"
SEQUENCE_LENGTH = 50
BATCH_SIZE = 32


class PricePredictionModel:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.memory_df = self.load_memory_table()

    def get_model_file(self, asset):
        return MODEL_DIR / f"lstm_model_{asset}.h5"

    def load_memory_table(self):
        if MEMORY_FILE.exists():
            return pl.read_parquet(MEMORY_FILE)
        return pl.DataFrame({"asset": [], "compressed_memory": []})

    def load_memory(self, asset):
        try:
            row = self.memory_df.filter(pl.col("asset") == asset)
            if row.is_empty():
                return np.zeros((SEQUENCE_LENGTH, 1), dtype=np.float32)
            return np.frombuffer(
                row[0]["compressed_memory"],
                dtype=np.float32
            ).reshape(
                SEQUENCE_LENGTH,
                1
            )
        except (KeyError, ValueError, TypeError):
            return np.zeros((SEQUENCE_LENGTH, 1), dtype=np.float32)

    def save_memory(self, asset, new_data):
        compressed = (
            np.mean(new_data, axis=0, keepdims=True).astype(np.float32)
        )
        existing = self.memory_df.filter(pl.col("asset") == asset)
        if not existing.is_empty():
            if np.array_equal(
                np.frombuffer(
                    existing[0]["compressed_memory"], dtype=np.float32
                ),
                compressed
            ):
                return
            self.memory_df = self.memory_df.filter(pl.col("asset") != asset)

        new_row = pl.DataFrame(
            {"asset": [asset], "compressed_memory": [compressed.tobytes()]}
        )
        self.memory_df = pl.concat([self.memory_df, new_row])
        self.memory_df.write_parquet(MEMORY_FILE, compression="zstd")

    def build_lstm_model(self):
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
        model.compile(
            optimizer="adam", loss="mean_squared_error"
        )
        return model

    def load_or_create_model(self, asset):
        model_file = self.get_model_file(asset)
        if model_file.exists():
            return load_model(model_file)
        return self.build_lstm_model()

    def preprocess_data(self, raw_data):
        raw_data = np.array(raw_data).reshape(-1, 1)
        return self.scaler.fit_transform(raw_data)

    def train_model(self, asset, raw_data):
        if len(raw_data) <= SEQUENCE_LENGTH:
            return

        local_model = self.load_or_create_model(asset)
        memory = self.load_memory(asset)

        df = pl.DataFrame({"close": raw_data})
        df = apply_all_market_structure_signals(df)

        last_row = df[-1]
        signal_score = (
            int(last_row["ILQ_Zone"]) +
            int(last_row["fakeout_up"]) +
            int(last_row["fakeout_down"]) +
            int(last_row["volatility_squeeze"]) +
            int(last_row["micro_pattern_hft"])
        )

        emb_m1 = get_embedding_for_symbol(asset, "1m")
        emb_m5 = get_embedding_for_symbol(asset, "5m")
        emb_m15 = get_embedding_for_symbol(asset, "15m")
        emb_m30 = get_embedding_for_symbol(asset, "30m")
        emb_1h = get_embedding_for_symbol(asset, "1h")
        emb_4h = get_embedding_for_symbol(asset, "4h")
        emb_1d = get_embedding_for_symbol(asset, "1d")

        extra_features = np.concatenate(
            [[signal_score], emb_m1, emb_m5, emb_m15, emb_m30,
             emb_1h, emb_4h, emb_1d]
        )

        data = self.preprocess_data(raw_data)
        x, y = [], []
        for i in range(len(data) - SEQUENCE_LENGTH):
            x.append(data[i:i+SEQUENCE_LENGTH])
            y.append(data[i+SEQUENCE_LENGTH])

        x = np.array(x)
        y = np.array(y)
        memory_tiled = np.tile(memory, (len(x), 1, 1))
        context_tiled = (
            np.tile(extra_features, (len(x), 1)).reshape(len(x), 1, -1)
        )
        full_input = np.concatenate([x, memory_tiled, context_tiled], axis=2)

        early_stop = EarlyStopping(
            monitor="loss", patience=3,
            restore_best_weights=True
        )
        local_model.fit(
            full_input, y, epochs=10,
            batch_size=BATCH_SIZE,
            verbose=1, callbacks=[early_stop]
        )

        local_model.save(self.get_model_file(asset))
        self.save_memory(asset, raw_data[-SEQUENCE_LENGTH:])

    def predict_price(
        self, asset: str, full_state: np.ndarray = None
    ) -> float:
        local_model = self.load_or_create_model(asset)

        if full_state is not None:
            try:
                full_state = (
                    np.array(full_state, dtype=np.float32).reshape(1, -1, 1)
                )
                prediction = local_model.predict(full_state, verbose=0)[0][0]
                return float(prediction)
            except Exception as e:
                logging.error(
                    f"❌ Errore della previsione con full_state x {asset}: {e}"
                )
                return None

        try:
            raw_data = get_normalized_market_data(asset)["close"].to_numpy()
            if len(raw_data) < SEQUENCE_LENGTH:
                logging.warning("⚠️ Dati insufficienti per %s", asset)
                return None

            data = self.preprocess_data(raw_data)
            last_sequence = (
                data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
            )
            prediction = local_model.predict(last_sequence, verbose=0)[0][0]
            predicted_price = (
                self.scaler.inverse_transform([[prediction]])[0][0]
            )

            logging.info(
                "📊 Prezzo previsto per %s: %.5f", asset, predicted_price
            )
            return float(predicted_price)

        except Exception as e:
            logging.error(f"❌ Errore fallback per {asset}: {e}")
            return None

    def build_full_state(self, asset) -> np.ndarray:
        df = pl.DataFrame(get_normalized_market_data(asset))
        if df.is_empty() or df.shape[0] == 0:
            return None

        df = apply_all_market_structure_signals(df)
        last_row = df[-1]

        signal_score = (
            int(last_row["ILQ_Zone"]) +
            int(last_row["fakeout_up"]) +
            int(last_row["fakeout_down"]) +
            int(last_row["volatility_squeeze"]) +
            int(last_row["micro_pattern_hft"])
        )

        emb_m1 = get_embedding_for_symbol(asset, "1m")
        emb_m5 = get_embedding_for_symbol(asset, "5m")
        emb_m15 = get_embedding_for_symbol(asset, "15m")
        emb_m30 = get_embedding_for_symbol(asset, "30m")
        emb_1h = get_embedding_for_symbol(asset, "1h")
        emb_4h = get_embedding_for_symbol(asset, "4h")
        emb_1d = get_embedding_for_symbol(asset, "1d")

        market_data_array = (
            df.select(pl.col(pl.NUMERIC_DTYPES)).to_numpy().flatten()
        )

        full_state = np.concatenate([
            market_data_array,
            [signal_score],
            emb_m1, emb_m5, emb_m15, emb_m30,
            emb_1h, emb_4h, emb_1d
        ])

        return np.clip(full_state, -1, 1)


if __name__ == "__main__":
    model = PricePredictionModel()
    all_assets = get_available_assets()

    for asset in all_assets:
        state = model.build_full_state(asset)
        if state is None:
            continue

        raw_close = (
            pl.DataFrame(get_normalized_market_data(asset))["close"].to_numpy()
        )
        if len(raw_close) > SEQUENCE_LENGTH:
            model.train_model(asset, raw_close)
            model.predict_price(asset=asset, full_state=state)
