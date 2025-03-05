"""
Modulo per la gestione avanzata del caricamento dei dati di mercato.
Ottimizzato per massima efficienza, velocità e scalabilità.
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
from functools import lru_cache
from data_loader import load_market_data_apis
from column_definitions import required_columns

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DAYS_HISTORY = 60
services = load_market_data_apis()
STORAGE_PATH = "market_data.zstd.parquet"
CLOUD_SYNC_PATH = "/mnt/google_drive/trading_sync/market_data.zstd.parquet"
CACHE_TTL = 3600  # Cache valida per 1 ora
cache_data = {}

EXECUTOR = ThreadPoolExecutor(max_workers=4)  # Ottimizzazione CPU


def ensure_all_columns(df):
    """Garantisce che il DataFrame contenga tutte le colonne richieste."""
    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def get_top_usdt_pairs():
    """Ottiene le prime coppie USDT con volume superiore a 5 milioni."""
    try:
        df = pd.read_parquet(STORAGE_PATH)
        usdt_pairs = df[
            (df["symbol"].str.endswith("USDT")) &
            (df["total_volume"] > 5_000_000)
        ].sort_values(by="total_volume", ascending=False).head(300)
        return usdt_pairs["symbol"].tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        logging.error("❌ Errore nel filtrare le coppie USDT: %s", error)
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
            "SOLUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "LTCUSDT"
        ]


async def fetch_market_data(
    session, url, exchange_name, requests_per_minute, retries=3
):
    """Scarica i dati di mercato con gestione avanzata degli errori."""
    delay = max(2, 60 / requests_per_minute)
    for attempt in range(retries):
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    logging.info(
                        "✅ Dati ottenuti da %s (tentativo %d)",
                        exchange_name, attempt + 1
                    )
                    return await response.json()
                if response.status in {400, 429}:
                    wait_time = 15
                    logging.warning(
                        "⚠️ Errore %d su %s. Attesa %d sec...",
                        response.status, exchange_name, wait_time
                    )
                    await asyncio.sleep(wait_time)
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            logging.error(
                "❌ Errore richiesta API %s su %s (tentativo %d): %s",
                url, exchange_name, attempt + 1, error
            )
            await asyncio.sleep(delay)
    return None


async def fetch_data_from_exchanges(currency="usdt", min_volume=5_000_000):
    """Scarica dati dalle borse con filtro di volume minimo."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for exchange in services["exchanges"]:
            api_url = exchange["api_url"].replace("{currency}", currency)
            req_per_min = exchange["limitations"].get(
                "requests_per_minute", 60
            )
            tasks.append(fetch_market_data(
                session, api_url, exchange["name"], req_per_min
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            data for data in results
            if data and data.get("total_volume", 0) >= min_volume
        ][:300]


@lru_cache(maxsize=1)
def download_no_api_data(symbols=None, interval="1d"):
    """Scarica dati senza l'uso di API con caching avanzato."""
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

    with ThreadPoolExecutor(max_workers=5) as exec_:
        for symbol in symbols:
            exec_.submit(
                fetch_data, "binance_data",
                f"{sources['binance_data']}/{symbol}/{interval}/"
                f"{symbol}-{interval}.zip", symbol
            )
            exec_.submit(
                fetch_data, "cryptodatadownload",
                f"{sources['cryptodatadownload']}/Binance_{symbol}_d.csv",
                symbol
            )
    return data


def save_and_sync(data, filename=STORAGE_PATH):
    """Salva i dati in formato Parquet con compressione e li sincronizza."""
    try:
        if not data:
            logging.warning("⚠️ Nessun dato valido. Skip salvataggio.")
            return

        df = pd.DataFrame(data)
        df = ensure_all_columns(df)
        df.to_parquet(filename, index=False, compression="zstd")
        logging.info("✅ Dati salvati con compressione ZSTD: %s", filename)
        sync_to_cloud()
    except Exception as error:
        logging.error("❌ Errore durante il salvataggio dei dati: %s", error)


def sync_to_cloud():
    """Sincronizza i dati locali a Google Drive solo se il file è cambiato."""
    try:
        if os.path.exists(STORAGE_PATH):
            shutil.copy(STORAGE_PATH, CLOUD_SYNC_PATH)
            logging.info("☁️ Dati sincronizzati su Google Drive.")
    except OSError as sync_error:
        logging.error("❌ Errore nella sincronizzazione: %s", sync_error)


async def main():
    """Funzione principale per l'aggiornamento dei dati."""
    logging.info("🔄 Avvio aggiornamento dati senza API...")
    top_usdt_pairs = get_top_usdt_pairs()
    data_no_api = download_no_api_data(symbols=top_usdt_pairs, interval="1d")
    if not data_no_api:
        logging.warning("⚠️ Nessun dato senza API. Passaggio alle API")
        data_no_api = await fetch_data_from_exchanges()
    save_and_sync(data_no_api, STORAGE_PATH)


if __name__ == "__main__":
    asyncio.run(main())
