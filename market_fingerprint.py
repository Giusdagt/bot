"""
market_fingerprint.py
Questo modulo gestisce la compressione dei dati di
mercato in vettori numerici
rappresentativi e l'aggiornamento/ripristino degli embedding
per simboli e timeframe specifici.
Funzionalità principali:
- compress_to_vector: Comprimi un DataFrame in un vettore numerico.
- update_embedding_in_processed_file:
Aggiorna il file processato con un embedding.
- get_embedding_for_symbol:
Recupera l'embedding per un dato simbolo e timeframe.
"""
from pathlib import Path
import hashlib
import polars as pl
import numpy as np

MODEL_DIR = (
    Path("/mnt/usb_trading_data/models") if
    Path("/mnt/usb_trading_data").exists() else
    Path("D:/trading_data/models")
)
PROCESSED_DATA_PATH = MODEL_DIR / "processed_data.zstd.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_FILE = PROCESSED_DATA_PATH


def compress_to_vector(df: pl.DataFrame, length: int = 32) -> np.ndarray:
    """
    Comprimi un DataFrame di candele in un vettore numerico rappresentativo.
    """
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).to_numpy()
    flattened = numeric_cols.flatten()
    hash_digest = hashlib.sha256(flattened.tobytes()).digest()
    raw_vector = np.frombuffer(hash_digest[:length * 4], dtype=np.float32)
    norm = np.linalg.norm(raw_vector)
    return raw_vector / norm if norm > 0 else raw_vector


def update_embedding_in_processed_file(
    symbol: str, new_df: pl.DataFrame, timeframe: str = "1m", length: int = 32
):
    """
    Aggiorna il file 'processed_data.zstd.parquet' con una colonna embedding.
    Salva 1 sola riga per simbolo + timeframe → ultra leggero e preciso.
    """
    if not PROCESSED_DATA_PATH.exists():
        print("❌ File processed_data.zstd.parquet non trovato.")
        return

    try:
        df = pl.read_parquet(PROCESSED_DATA_PATH)
        compressed_vector = compress_to_vector(new_df, length=length)

        # Rimuove righe duplicate (stesso symbol + timeframe)
        df = df.filter(
            ~(
                (pl.col("symbol") == symbol) &
                (pl.col("timeframe") == timeframe)
            )
        )

        latest_row = new_df[-1].with_columns([
            pl.Series(name="symbol", values=[symbol]),
            pl.Series(name="timeframe", values=[timeframe]),
            pl.Series(name="embedding", values=[compressed_vector.tobytes()])
        ])

        updated_df = pl.concat([df, latest_row])
        updated_df.write_parquet(
            PROCESSED_DATA_PATH, compression="zstd", mode="overwrite"
        )
        print(f"✅ Embedding aggiornato per {symbol} [{timeframe}]")

    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"❌ Errore durante aggiornamento embedding: {e}")


def get_embedding_for_symbol(
    symbol: str, timeframe: str = "1m", length: int = 32
) -> np.ndarray:
    """
    Recupera l'embedding vettoriale salvato per un dato simbolo e timeframe.
    Restituisce un array numpy normalizzato (32 valori).
    """
    try:
        if not EMBEDDING_FILE.exists():
            return np.zeros(length, dtype=np.float32)

        df = pl.read_parquet(EMBEDDING_FILE)
        row = df.filter(
            (pl.col("symbol") == symbol) & (pl.col("timeframe") == timeframe)
        )

        if row.is_empty():
            return np.zeros(length, dtype=np.float32)

        raw = row[0]["embedding"]
        vector = np.frombuffer(raw, dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"❌ Errore durante il recupero dell'embedding: {e}")
        return np.zeros(length, dtype=np.float32)
