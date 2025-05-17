"""
data_api_module.py
Modulo avanzato per il download e la gestione dei dati di mercato.
"""

import asyncio
import logging
import os
import stat
import sys
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import polars as pl
import yfinance as yf
import MetaTrader5 as mt5
from polars.exceptions import ComputeError
from data_loader import (
    load_auto_symbol_mapping,
    standardize_symbol,
    load_preset_assets,
)
from column_definitions import required_columns

# Configurazione globale
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_PATH = os.path.join(SCRIPT_DIR, "market_data.zstd.parquet")
CLOUD_SYNC_PATH = "/mnt/google_drive/trading_sync/market_data.zstd.parquet"
ENABLE_CLOUD_SYNC = False
USE_MARKET_APIS = False
ENABLE_YFINANCE = False   # Attiva/disattiva yfinance
ENABLE_MT5 = True   # Attiva/disattiva MT5
CHUNK_SIZE = 10000        # Numero massimo di righe per ogni chunk
start_date = (datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d")
DAYS_HISTORY = 60
executor = ThreadPoolExecutor(max_workers=8)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Utility
def ensure_permissions(file_path):
    """Garantisce che il file abbia i permessi di lettura e scrittura."""
    try:
        if not os.path.exists(file_path):
            with open(file_path, 'w'):
                pass
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        logging.info("✅ Permessi garantiti per il file: %s", file_path)
    except Exception as e:
        logging.error(
            "❌ Errore nell'impostare i permessi per %s: %s", file_path, e
        )


def ensure_all_columns(df):
    """Garantisce che il DataFrame contenga tutte le colonne richieste."""
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    for col in missing_columns:
        df = df.with_columns(pl.lit(0).alias(col))
    return df.fill_nan(0).fill_null(0)


def delay_request(seconds=2):
    """Introduce un ritardo tra le richieste."""
    logging.info(
        "⏳ Attendo %d secondi prima della prossima richiesta...", seconds
    )
    time.sleep(seconds)


def retry_request(func, *args, retries=3, delay=2, **kwargs):
    """
    Riprova una funzione in caso di errore,
    con un ritardo esponenziale tra i tentativi.
    """
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"⚠️ Errore tentativo {attempt + 1}: {e}")
            time.sleep(delay * (2 ** attempt))
    return None


def download_data_with_yfinance(symbols, start_date=start_date):
    """Scarica dati storici da Yahoo Finance usando batch di ticker."""
    if start_date is None:
        start_date = (
            datetime.today() - timedelta(days=60)
        ).strftime("%Y-%m-%d")
    data = []
    try:
        yf_symbols = [standardize_symbol(
            s, load_auto_symbol_mapping(),
            provider="yfinance") for s in symbols]
        combined_data = None

        for attempt in range(3):
            try:
                combined_data = yf.download(
                    yf_symbols,
                    start=start_date,
                    end=datetime.today().strftime("%Y-%m-%d"),
                    progress=False,
                    group_by=None
                )
                if combined_data.empty or combined_data.isna().all().all():
                    logging.warning(
                        "⚠️ Nessun dato valido ricevuto dal batch di yfinance."
                    )
                    return []

                combined_data = combined_data.reset_index()
                if isinstance(combined_data.columns, pd.MultiIndex):
                    combined_data.columns = [
                        "_".join(
                            [str(i) for i in col if i]
                        ) for col in combined_data.columns.values
                    ]
                else:
                    combined_data.columns = [
                        str(col) for col in combined_data.columns
                    ]
                combined_data["symbol"] = (
                    yf_symbols[0] if len(yf_symbols) == 1 else "MULTIPLE"
                )
                data.extend(combined_data.to_dict(orient="records"))
                logging.info("✅ Dati batch scaricati con successo.")
                break
            except Exception as e:
                if "Rate limited" in str(e):
                    logging.warning(
                        "⚠️ Rate limit raggiunto. Attendo 60sec prima di riprovare"
                    )
                    time.sleep(60)
                else:
                    logging.error(f"❌ Errore durante il download batch: {e}")
                    return []
    except Exception as e:
        logging.error(f"❌ Errore durante il download batch: {e}")
    return data


def download_data_with_mt5(symbols, days=60, timeframes=None):
    """
    Scarica dati storici da MT5 per gli ultimi N giorni e per più timeframe.
    """
    if not mt5.initialize():
        logging.error("❌ Impossibile inizializzare MT5: %s", mt5.last_error())
        return []

    if timeframes is None:
        timeframes = [mt5.TIMEFRAME_D1]  # Default solo daily

    tf_map = {
        "1d": mt5.TIMEFRAME_D1,
        "h4": mt5.TIMEFRAME_H4,
        "h1": mt5.TIMEFRAME_H1,
    }

    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    data = []

    for tf_name in timeframes:
        tf = tf_map[tf_name]
        for symbol in symbols:
            rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
            if rates is None or len(rates) == 0:
                logging.warning(
                    f"⚠️ Nessun dato per {symbol} timeframe {tf_name} da MT5."
                )
                continue
            df = pd.DataFrame(rates)
            df["symbol"] = symbol
            df["timeframe"] = tf_name
            df.rename(columns={
                "time": "timestamp",
                "open": f"{symbol}_Open",
                "high": f"{symbol}_High",
                "low": f"{symbol}_Low",
                "close": f"{symbol}_Close",
                "tick_volume": f"{symbol}_Volume"
            }, inplace=True)
            data.extend(df.to_dict(orient="records"))
            time.sleep(1)
    mt5.shutdown()
    return data


def save_and_sync(data):
    """Salva i dati grezzi e sincronizza se richiesto."""
    if not data:
        logging.warning("⚠️ Nessun dato da salvare. Skip del salvataggio.")
        return

    ensure_permissions(STORAGE_PATH)
    try:
        df_new = pl.DataFrame(data)
        df_new = ensure_all_columns(df_new)
        if df_new.is_empty():
            logging.warning(
                "⚠️ DataFrame vuoto dopo la pulizia. Skip del salvataggio."
            )
            return

        if os.path.exists( STORAGE_PATH ) and os.path.getsize(STORAGE_PATH) > 0:
            existing_df = pl.read_parquet(STORAGE_PATH)
            df_final = pl.concat(
                [existing_df, df_new]
            ).unique(subset=["timestamp", "symbol"])
        else:
            df_final = df_new

            df_final = df_final.sort(["timestamp", "symbol"])

        df_final.write_parquet(STORAGE_PATH, compression="zstd")
        logging.info("✅ Dati grezzi salvati: %s", STORAGE_PATH)

        if ENABLE_CLOUD_SYNC:
            sync_to_cloud()
    except (OSError, IOError, ComputeError) as e:
        logging.error("❌ Errore salvataggio dati: %s", e)


def sync_to_cloud():
    """Sincronizzazione avanzata intelligente con il cloud."""
    try:
        if os.path.exists(STORAGE_PATH):
            os.replace(STORAGE_PATH, CLOUD_SYNC_PATH)
            logging.info("☁️ Sincronizzazione intelligente cloud completata.")
    except OSError as e:
        logging.error("❌ Errore sincronizzazione cloud: %s", e)


async def main():
    """Download e salvataggio dei dati di mercato con fallback automatico."""
    try:
        assets = load_preset_assets()
        if not assets:
            logging.error("❌ Nessun asset trovato. Operazione terminata.")
            return

        mapping = load_auto_symbol_mapping()
        symbols = [
            standardize_symbol(
                symbol, mapping, provider="yfinance"
            )
            for category, asset_list in assets.items() for symbol in asset_list
        ]

        data_all = []

        if ENABLE_YFINANCE:
            data_yf = []
            for i in range(0, len(symbols), 50):
                batch = symbols[i:i + 50]
                data_batch = retry_request(
                    download_data_with_yfinance, batch,
                    start_date=start_date
                )
                if data_batch:
                    data_yf.extend(data_batch)
                delay_request(10)
            data_all.extend(data_yf)

        if ENABLE_MT5:
            # Usa i simboli originali da preset_asset per MT5
            mt5_symbols = [
                symbol for category,
                asset_list in assets.items() for symbol in asset_list
            ]
            data_mt5 = download_data_with_mt5(
                mt5_symbols, days=DAYS_HISTORY,
                timeframes=["1d", "h4", "h1"]
            )
            data_all.extend(data_mt5)

        if len(data_all) > CHUNK_SIZE:
            for chunk in range(0, len(data_all), CHUNK_SIZE):
                partial_data = data_all[chunk:chunk + CHUNK_SIZE]
                save_and_sync(partial_data)
        else:
            save_and_sync(data_all)
        logging.info(
            "🎉 Processo completato con successo! File generato: %s",
            STORAGE_PATH
        )
    except Exception as e:
        logging.error("❌ Errore durante l'esecuzione: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
