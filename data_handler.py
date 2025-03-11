"""
data_handler.py
Modulo definitivo per la gestione autonoma, intelligente e ultra-ottimizzata
per la normalizzazione e gestione avanzata dei dati storici e realtime.
Ottimizzato per IA, Deep Reinforcement Learning (DRL) e scalping con MetaTrader5.
"""

import os
import logging
import hashlib
import shutil
import asyncio
import polars as pl
import MetaTrader5 as mt5
from concurrent.futures import ThreadPoolExecutor
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
PROCESSED_DATA_PATH = os.path.join(SAVE_DIRECTORY, "processed_data.zstd.parquet")
CLOUD_SYNC_PATH = "/mnt/google_drive/trading_sync/processed_data.zstd.parquet"

executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 8)
scalping_buffer = []
BUFFER_SIZE = 100


def file_hash(filepath):
    """Calcola l'hash del file per rilevare modifiche."""
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
        existing_hash = file_hash(CLOUD_SYNC_PATH) if os.path.exists(CLOUD_SYNC_PATH) else None
        new_hash = file_hash(PROCESSED_DATA_PATH)
        if existing_hash == new_hash:
            logging.info("☁️ Nessuna modifica, skip sincronizzazione.")
            return
        shutil.copy2(PROCESSED_DATA_PATH, CLOUD_SYNC_PATH)
        logging.info("☁️ Sincronizzazione cloud completata.")
    except Exception as e:
        logging.error("❌ Errore sincronizzazione cloud: %s", e)


def save_and_sync(df):
    """Salvataggio ultra-intelligente con verifica delle modifiche."""
    new_hash = hashlib.md5(df.write_csv().encode()).hexdigest()
    if os.path.exists(PROCESSED_DATA_PATH):
        old_hash = file_hash(PROCESSED_DATA_PATH)
        if old_hash == new_hash:
            logging.info("🔄 Nessuna modifica, salvataggio non necessario.")
            return
    df.write_parquet(PROCESSED_DATA_PATH, compression="zstd")
    logging.info("✅ Dati elaborati salvati con successo.")
    executor.submit(sync_to_cloud)


def normalize_data(df):
    """Normalizzazione avanzata con selezione dinamica delle feature per IA."""
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.select(numeric_cols).to_numpy())
    df = df.with_columns(
        [pl.Series(col, scaled_data[:, idx]) for idx, col in enumerate(numeric_cols)]
    )
    return df


def process_historical_data():
    """Elabora i dati storici, calcola indicatori avanzati e li normalizza."""
    try:
        df = pl.read_parquet(RAW_DATA_PATH)
        df = calculate_historical_indicators(df)
        df = normalize_data(df)
        save_and_sync(df)
    except Exception as e:
        logging.error("❌ Errore elaborazione dati storici: %s", e)


async def get_realtime_data(symbols):
    """Ottiene i dati realtime da MetaTrader5 e calcola indicatori."""
    if not mt5.initialize():
        logging.error("❌ Errore inizializzazione MT5: %s", mt5.last_error())
        return
    tasks = [asyncio.to_thread(fetch_mt5_data, symbol) for symbol in symbols]
    await asyncio.gather(*tasks)
    mt5.shutdown()


def fetch_mt5_data(symbol):
    """Scarica dati in parallelo da MT5."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
    if rates is None or len(rates) == 0:
        logging.warning("⚠️ Nessun dato realtime per %s", symbol)
        return
    df = pl.DataFrame(rates)
    df = calculate_scalping_indicators(df)
    df = normalize_data(df)
    scalping_buffer.append(df)
    if len(scalping_buffer) >= BUFFER_SIZE:
        df_batch = pl.concat(scalping_buffer)
        save_and_sync(df_batch)
        scalping_buffer.clear()
        logging.info("✅ Dati scalping aggiornati, batch di %d messaggi", BUFFER_SIZE)


def fetch_and_process_data():
    """Scarica i dati grezzi solo se necessario e li elabora."""
    if not os.path.exists(RAW_DATA_PATH):
        logging.info("⚠️ Dati grezzi mancanti, avvio scaricamento...")
        executor.submit(fetch_new_data)
    process_historical_data()


if __name__ == "__main__":
    auto_mapping = load_auto_symbol_mapping()
    fetch_and_process_data()
    realtime_symbols = (
        sum(load_preset_assets().values(), [])
        if USE_PRESET_ASSETS else
        list(auto_mapping.values())
    )
    realtime_symbols = [standardize_symbol(s, auto_mapping) for s in realtime_symbols]
    asyncio.run(get_realtime_data(realtime_symbols))
