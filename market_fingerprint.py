from pathlib import Path
import polars as pl
import numpy as np
import hashlib

# Percorso del file principale di data_handler
MODEL_DIR = Path("/mnt/usb_trading_data/models") if Path("/mnt/usb_trading_data").exists() else Path("D:/trading_data/models")
PROCESSED_DATA_PATH = MODEL_DIR / "processed_data.zstd.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

def update_embedding_in_processed_file(symbol: str, new_df: pl.DataFrame, length: int = 32):
    """
    Aggiorna il file 'processed_data.zstd.parquet' con una colonna embedding.
    Salva 1 sola riga per simbolo → leggero e potente.
    """
    if not PROCESSED_DATA_PATH.exists():
        print("❌ File processed_data.zstd.parquet non trovato.")
        return

    try:
        df = pl.read_parquet(PROCESSED_DATA_PATH)
        compressed_vector = compress_to_vector(new_df, length=length)

        df = df.filter(pl.col("symbol") != symbol)
        latest_row = new_df[-1].with_columns([
            pl.Series(name="embedding", values=[compressed_vector.tobytes()])
        ])

        updated_df = pl.concat([df, latest_row])
        updated_df.write_parquet(PROCESSED_DATA_PATH, compression="zstd", mode="overwrite")
        print(f"✅ Embedding aggiornato per {symbol} nel file processato.")
    except Exception as e:
        print(f"❌ Errore durante aggiornamento embedding: {e}")
