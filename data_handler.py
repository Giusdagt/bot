"""
data_handler.py
Modulo definitivo per la gestione autonoma, intelligente e ottimizzata
per la normalizzazione e gestione avanzata dei dati storici e realtime.
Ottimizzato per IA, Deep Reinforcement Learning (DRL) e scalping
con MetaTrader5.
"""

import os
import logging
import hashlib
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import polars as pl
import MetaTrader5 as mt5
from sklearn.preprocessing import MinMaxScaler
from column_definitions import required_columns
from indicators import (
    calculate_historical_indicators,
    calculate_scalping_indicators
)
from data_loader import (
    load_auto_symbol_mapping,
    standardize_symbol,
    USE_PRESET_ASSETS,
    load_preset_assets
)
from data_api_module import main as fetch_new_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SAVE_DIRECTORY = (
    "/mnt/usb_trading_data/processed_data"
    if os.path.exists("/mnt/usb_trading_data")
    else "D:/trading_data"
)

RAW_DATA_PATH = "market_data.zstd.parquet"
PROCESSED_DATA_PATH = os.path.join(
    SAVE_DIRECTORY, "processed_data.zstd.parquet"
)
CLOUD_SYNC_PATH = "/mnt/google_drive/trading_sync/processed_data.zstd.parquet"

os.makedirs(SAVE_DIRECTORY, exist_ok=True)
os.makedirs(os.path.dirname(CLOUD_SYNC_PATH), exist_ok=True)

executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 8)


def file_hash(filepath):
    """Calcola l'hash del file per rilevare modifiche."""
    if not os.path.exists(filepath):
        return None
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def sync_to_cloud():
    """Sincronizzazione con Google Drive solo se necessario."""
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            return
        existing_hash = file_hash(CLOUD_SYNC_PATH)
        new_hash = file_hash(PROCESSED_DATA_PATH)
        if existing_hash == new_hash:
            logging.info("☁️ Nessuna modifica, skip sincronizzazione.")
            return
        shutil.copy2(PROCESSED_DATA_PATH, CLOUD_SYNC_PATH)
        logging.info("☁️ Sincronizzazione cloud completata.")
    except (OSError, IOError) as e:
        logging.error("❌ Errore sincronizzazione cloud: %s", e)


def save_and_sync(df):
    """Salvataggio intelligente con verifica delle modifiche."""
    try:
        if df.is_empty():
            logging.warning("⚠️ Tentativo salvataggio, un DataFrame vuoto.")
            return
        df.write_parquet(PROCESSED_DATA_PATH, compression="zstd")
        logging.info("✅ Dati elaborati salvati con successo.")
        executor.submit(sync_to_cloud)
    except (OSError, IOError, ValueError) as e:
        logging.error("❌ Errore durante il salvataggio dati: %s", e)


def ensure_all_columns(df):
    """Garantisce che il DataFrame contenga tutte le colonne richieste."""
    for col in required_columns:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    return df


def normalize_data(df):
    """Normalizzazione avanzata con selezione dinamica delle feature per IA."""
    if df.is_empty():
        return df
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.select(numeric_cols).to_numpy())
    df = df.with_columns(
        [pl.Series(col, scaled_data[:, idx]) for idx, col in enumerate(
            numeric_cols)]
    )
    return df


def process_historical_data():
    """Elabora i dati storici, calcola indicatori avanzati e li normalizza."""
    try:
        if not os.path.exists(RAW_DATA_PATH):
            logging.warning("⚠️ Dati grezzi non trovato, avvio fetch.")
            fetch_new_data()
        df = pl.read_parquet(RAW_DATA_PATH)
        if df.is_empty():
            logging.warning("⚠️ File dati grezzi vuoto, nessun dato da processare.")
            return
        df = calculate_historical_indicators(df)
        df = ensure_all_columns(df)
        df = normalize_data(df)
        save_and_sync(df)
    except (OSError, IOError, ValueError) as e:
        logging.error("❌ Errore elaborazione dati storici: %s", e)


def fetch_mt5_data(symbol):
    """Recupera dati di scalping in tempo reale da MetaTrader5."""
    try:
        if not mt5.initialize():
            logging.error("❌Errore inizializzazione MT5: %s", mt5.last_error())
            return None

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
        if rates is None or len(rates) == 0:
            logging.warning(f"⚠️Nessun dato realtime disponibile per {symbol}")
            return None

        df = pl.DataFrame(rates)
        df = calculate_scalping_indicators(df)
        df = ensure_all_columns(df)
        df = normalize_data(df)

        return df

    except Exception as e:
        logging.error(f"❌ Errore nel recupero dati MT5 per {symbol}: {e}")
        return None


def get_realtime_data(symbols):
    """Ottiene dati in tempo reale da MT5 e aggiorna il database."""
    try:
        for symbol in symbols:
            logging.info(f"📡 Recupero dati real-time per {symbol}")
            df = fetch_mt5_data(symbol)
            if df is None:
                continue

            save_and_sync(df)
            logging.info(f"✅ Dati real-time per {symbol} aggiornati.")
    except Exception as e:
        logging.error(f"❌ Errore nel recupero dei dati real-time: {e}")


def get_normalized_market_data(symbol):
    """Recupera dati normalizzati per un singolo simbolo in modo efficiente"""
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            logging.warning("⚠️ File dati processati non trovato.")
            return None

        df = pl.scan_parquet(PROCESSED_DATA_PATH).filter(
            pl.col("symbol") == symbol
        ).collect()

        if df.is_empty():
            logging.warning(f"⚠️ Nessun dato trovato {symbol}, avvio fetch.")
            fetch_new_data()
            return None

        latest_data = df[-1]  # Prende l'ultimo valore disponibile
        return latest_data.to_dict()

    except Exception as e:
    logging.error(f"❌ Errore durante il recupero dei dati " 
                  f"normalizzati per {symbol}: {e}")
    return None


if __name__ == "__main__":
    auto_mapping = load_auto_symbol_mapping()
    process_historical_data()
    realtime_symbols = (
        sum(load_preset_assets().values(), [])
        if USE_PRESET_ASSETS else
        list(auto_mapping.values())
    )
    realtime_symbols = [
        standardize_symbol(s, auto_mapping) for s in realtime_symbols
    ]
    asyncio.run(get_realtime_data(realtime_symbols))
