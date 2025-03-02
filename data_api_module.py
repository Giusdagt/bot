"""
Modulo per la gestione del caricamento dei dati di mercato.
"""

import asyncio
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import requests
import pandas as pd
from data_loader import load_market_data_apis

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DAYS_HISTORY = 60
services = load_market_data_apis()
STORAGE_PATH = "market_data.parquet"
CLOUD_SYNC_PATH = "/mnt/google_drive/trading_sync/market_data.parquet"

def ensure_all_columns(df):
    """
    Assicura che il DataFrame contenga tutte le colonne richieste.
    Se una colonna manca, viene aggiunta con valore pd.NA.
    """
    required_columns = [
        'coin_id', 'symbol', 'name', 'image', 'last_updated',
        'historical_prices', 'timestamp', "close", "open", "high",
        "low", "volume"
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def download_no_api_data(symbols=None, interval="1d"):
    """
    Scarica dati senza l'uso di API.
    """
    if symbols is None:
        symbols = get_top_usdt_pairs()
    sources = services["data_sources"]["no_api"]
    data = {}

    def fetch_data(source_name, url, symbol):
        response = requests.get(url)
        if response.status_code == 200:
            if symbol not in data:
                data[symbol] = {}
            data[symbol][source_name] = url
            logging.info("✅ Dati %s scaricati per %s", source_name, symbol)

    with ThreadPoolExecutor(max_workers=5) as executor:
        for symbol in symbols:
            executor.submit(
                fetch_data, "binance_data",
                f"{sources['binance_data']}/{symbol}/{interval}/"
                f"{symbol}-{interval}.zip", symbol
            )
            executor.submit(
                fetch_data, "cryptodatadownload",
                f"{sources['cryptodatadownload']}/Binance_{symbol}_d.csv",
                symbol
            )
    return data


def save_and_sync(data, filename="market_data.parquet"):
    """
    Salva i dati in formato Parquet e li sincronizza con Google Drive.
    """
    try:
        if not data:
            logging.warning("⚠️ Tentativo di salvataggio di dati vuoti. Operazione annullata.")
            return
        df = pd.DataFrame(data)
        df = ensure_all_columns(df)
        df.to_parquet(filename, index=False)
        logging.info("✅ Dati aggiornati in %s con tutte le colonne richieste.", filename)
        sync_to_cloud()
    except (ValueError, KeyError, OSError, IOError) as e:
        logging.error("❌ Errore durante il salvataggio dei dati di mercato: %s", e)


def sync_to_cloud():
    """
    Sincronizza i dati locali con Google Drive.
    """
    if os.path.exists(STORAGE_PATH):
        try:
            os.makedirs(os.path.dirname(CLOUD_SYNC_PATH), exist_ok=True)
            shutil.copy(STORAGE_PATH, CLOUD_SYNC_PATH)
            logging.info("☁️ Dati sincronizzati su Google Drive.")
        except OSError as sync_error:
            logging.error("❌ Errore nella sincronizzazione con Google Drive: %s", sync_error)


def main():
    """
    Funzione principale per l'aggiornamento dei dati.
    """
    logging.info("🔄 Avvio aggiornamento dati esclusivamente senza API...")
    top_usdt_pairs = get_top_usdt_pairs()
    data_no_api = download_no_api_data(symbols=top_usdt_pairs, interval="1d")
    if not data_no_api:
        logging.warning("⚠️ Nessun dato trovato senza API. Passaggio alle API solo se necessario...")
        asyncio.run(fetch_data_from_exchanges(aiohttp.ClientSession()))
    save_and_sync(data_no_api, STORAGE_PATH)
    logging.info("✅ Processo completato utilizzando principalmente dati senza API.")


if __name__ == "__main__":
    main()
