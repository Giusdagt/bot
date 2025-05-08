"""
data_handler.py
Modulo definitivo per la gestione autonoma, intelligente e ottimizzata
per la normalizzazione e gestione avanzata dei dati storici e realtime.
Ottimizzato per IA, Deep Reinforcement Learning (DRL) e scalping
con MetaTrader5.
"""

import os
import sys
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
from smart_features import apply_all_advanced_features, detect_strategy_type
from ai_features import get_features_by_strategy_type
from market_fingerprint import update_embedding_in_processed_file

print("data_handler.py caricato ✅")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Disabilita l'inizializzazione MetaTrader5 durante i test o in modalità mock
if "pytest" not in sys.modules and "MOCK_MT5" not in os.environ:
    if not mt5.initialize():
        logging.error("❌ Errore inizializzazione MT5: %s", mt5.last_error())
        sys.exit()
else:
    logging.info("⚠️ Inizializzazione MT5 disabilitata (modalità test/mock).")

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
    strategy_type = detect_strategy_type(df)
    numeric_cols = get_features_by_strategy_type(strategy_type)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.select(numeric_cols).to_numpy())
    df = df.with_columns(
        [pl.Series(col, scaled_data[:, idx]) for idx, col in enumerate(
            numeric_cols)]
    )
    df = apply_all_advanced_features(df)
    return df


async def process_historical_data():
    """Elabora i dati storici, calcola indicatori avanzati e li normalizza."""
    try:
        if not os.path.exists(RAW_DATA_PATH):
            logging.warning("⚠️ Grezzi non trovato, avvio fetch.")
        await fetch_new_data()
        df = pl.read_parquet(RAW_DATA_PATH)
        if df.is_empty():
            logging.warning("⚠️ Dati grezzi vuoto, nessun dato da processare.")
            return
        df = calculate_historical_indicators(df)
        df = apply_all_advanced_features(df)
        df = ensure_all_columns(df)
        df = normalize_data(df)
        save_and_sync(df)
    except (OSError, IOError, ValueError) as e:
        logging.error("❌ Errore elaborazione dati storici: %s", e)


def fetch_mt5_data(symbol, timeframe="1m"):
    """
    Recupera i dati di mercato da MetaTrader5 per un simbolo e timeframe.
    Args:
    symbol (str): Il simbolo di mercato da analizzare.
    timeframe (str): Il timeframe (es. "1m", "5m", "1h").
    Returns:
    pl.DataFrame: Dati elaborati, normalizzati e arricchiti con indicatori.
    """
    try:
        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }

        tf_mt5 = tf_map[timeframe]
        rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, 1)
        if rates is None or len(rates) == 0:
            return None

        df = pl.DataFrame(rates)
        df = df.with_columns(pl.lit(timeframe).alias("timeframe"))
        df = calculate_scalping_indicators(df)
        df = apply_all_advanced_features(df)
        df = ensure_all_columns(df)
        df = normalize_data(df)
        return df

    except (OSError, IOError, ValueError) as e:
        logging.error("❌ Errore nel recupero dati MT5 per %s: %s", symbol, e)
        return None


def get_multi_timeframe_data(symbol, timeframes):
    """
    Restituisce un dizionario con i dati di mercato
    per ciascun timeframe specificato.
    Sceglie automaticamente se utilizzare dati normalizzati o
    recuperare dati diretti.
    """
    result = {}
    for tf in timeframes:
        try:
            # Prova a recuperare i dati normalizzati
            normalized_data = get_normalized_market_data(f"{symbol}_{tf}")
            if normalized_data is not None:
                result[tf] = normalized_data
            else:
                # Se i dati non sono disponibili, recupera i dati diretti
                result[tf] = fetch_mt5_data(symbol, timeframe=tf)
        except Exception as e:
            # Log dell'errore per eventuali problemi nel recupero dei dati
            logging.error(
                "Errore nel recupero dei dati per %s con timeframe %s: %s",
                symbol, tf, e
            )
            result[tf] = None
    return result


async def get_realtime_data(symbols):
    """Ottiene dati in tempo reale da MT5 e aggiorna il database."""
    try:
        for symbol in symbols:
            for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
                logging.info("📡 Dati realtime %s | TF %s", symbol, tf)
                df = fetch_mt5_data(symbol, timeframe=tf)
                if df is None:
                    continue
                update_embedding_in_processed_file(symbol, df)
                save_and_sync(df)

            logging.info("✅ Dati realtime per %s aggiornati.", symbol)
    except (OSError, IOError, ValueError) as e:
        logging.error("❌ Errore nel recupero dei dati realtime: %s", e)


def get_normalized_market_data(symbol):
    """Recupera dati normalizzati per un singolo simbolo in modo efficiente"""
    try:
        if not os.path.exists(PROCESSED_DATA_PATH):
            logging.warning("⚠️ File dati processati non trovato.")
            return None

        df = pl.scan_parquet(PROCESSED_DATA_PATH).filter(
            pl.col("symbol") == symbol
        ).collect()

        if df is None or df.is_empty():
            logging.warning("⚠️ Nessun dato trovato %s, avvio fetch.", symbol)
            fetch_new_data()
            return None

        # Verifica il numero di righe e restituisce di conseguenza
        if df.shape[0] == 1:
            # Se il DataFrame ha una sola riga, restituiscila come dizionario
            return df[-1].to_dict()

        # Altrimenti, restituisci il DataFrame completo
        return df

    except (OSError, IOError, ValueError) as e:
        logging.error("❌ Errore durante il recupero dei dati "
                      "normalizzati per %s: %s", symbol, e)
        return None


async def main():
    """
    Funzione principale per inizializzare MT5,
    elaborare dati storici e recuperare dati in tempo reale.
    Questa funzione carica la mappatura automatica dei simboli,
    elabora i dati storici e recupera i dati in tempo reale per
    i simboli specificati utilizzando MetaTrader5.
    """
    try:
        auto_mapping = load_auto_symbol_mapping()
        await process_historical_data()
        realtime_symbols = (
            sum(load_preset_assets().values(), [])
            if USE_PRESET_ASSETS else
            list(auto_mapping.values())
        )
        realtime_symbols = [
            standardize_symbol(s, auto_mapping) for s in realtime_symbols
        ]
        await get_realtime_data(realtime_symbols)
    finally:
        mt5.shutdown()  # Assicura che la connessione venga chiusa alla fine


def get_available_assets():
    """
    Restituisce tutti gli asset disponibili, da preset o in modo dinamico.
    Nessuna limitazione su USD, EUR o altro.
    """
    if USE_PRESET_ASSETS:
        assets = load_preset_assets()
        return sum(assets.values(), [])

    mapping = load_auto_symbol_mapping()
    return list(mapping.values())


if __name__ == "__main__":
    asyncio.run(main())
