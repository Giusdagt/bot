"""
data_api_module.py
Modulo ultra-avanzato, dinamico e intelligente per download dati di mercato.
Performance massimizzata, completamente automatico e intelligente.
"""

import asyncio
import logging
import os
import sys
import aiohttp
import requests
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from data_loader import (
    load_market_data_apis,
    load_auto_symbol_mapping,
    standardize_symbol,
    USE_PRESET_ASSETS,
    load_preset_assets,
)
from column_definitions import required_columns

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

STORAGE_PATH = "market_data.zstd.parquet"
CLOUD_SYNC_PATH = "/mnt/google_drive/trading_sync/market_data.zstd.parquet"
DAYS_HISTORY = 60

executor = ThreadPoolExecutor(max_workers=8)


def ensure_all_columns(df):
    """Garantisce presenza di tutte le colonne necessarie."""
    existing_columns = df.columns
    missing_columns = [col for col in required_columns if col not in existing_columns]
    for col in missing_columns:
        df = df.with_columns(pl.lit(None).alias(col))
    return df


async def fetch_market_data(session, url, exchange_name, rpm, retries=3):
    """Scarica dati API con gestione avanzata."""
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=15) as response:
                if response.status == 200:
                    logging.info(
                        "✅ Dati ottenuti da %s (tentativo %d)",
                        exchange_name, attempt + 1
                    )
                    return await response.json()
                if response.status in {400, 429}:
                    await asyncio.sleep(15)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.error(
                "❌ Errore API %s (tentativo %d): %s",
                exchange_name, attempt + 1, e
            )
            await asyncio.sleep(max(1, 60 / rpm))
    return None


async def fetch_from_all_exchanges(symbols, days_history):
    """Scarica dati API con storico dinamico ultra-avanzato."""
    market_data_apis = load_market_data_apis()
    tasks = []
    async with aiohttp.ClientSession() as session:
        for exchange in market_data_apis["exchanges"]:
            for symbol in symbols:
                api_url = exchange["api_url"]
                api_url = api_url.replace("{symbol}", symbol)
                api_url = api_url.replace("{days}", str(days_history))
                rpm = exchange["limitations"].get("requests_per_minute", 60)
                tasks.append(
                    fetch_market_data(session, api_url, exchange["name"], rpm)
                )
        results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_data = []
    for data in results:
        if isinstance(data, (dict, list)):
            valid_data.extend(data if isinstance(data, list) else [data])

    return valid_data


@lru_cache(maxsize=128)
def download_no_api_data(symbols, interval="1d"):
    """Download prioritario e intelligente senza API."""
    market_data_apis = load_market_data_apis()
    sources = market_data_apis["data_sources"]["no_api"]
    data = []

    def fetch(symbol, source_name, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data.append({"symbol": symbol, "source": source_name})
            logging.info("✅ %s scaricato da %s", symbol, source_name)
        except requests.RequestException as e:
            logging.warning("⚠️ Errore fonte no-api '%s': %s", source_name, e)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                fetch,
                symbol,
                source_name,
                f"{sources[source_name]}/{symbol}/{interval}/{symbol}-{interval}.zip"
            )
            for source_name in sources
            for symbol in symbols
        ]
        for future in futures:
            future.result()

    return data


def save_and_sync(data):
    """Salvataggio intelligente e sincronizzazione selettiva."""
    if not data:
        logging.warning("⚠️ Nessun dato valido da salvare.")
        return

    df_new = pl.DataFrame(data)
    df_new = ensure_all_columns(df_new)

    if os.path.exists(STORAGE_PATH):
        existing_df = pl.read_parquet(STORAGE_PATH)
        df_final = pl.concat([existing_df, df_new]).unique()
    else:
        df_final = df_new

    try:
        df_final.write_parquet(STORAGE_PATH, compression="zstd")
        logging.info("✅ Dati salvati ultra-veloce: %s", STORAGE_PATH)
        sync_to_cloud()
    except Exception as e:
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
    symbols = (
        sum(load_preset_assets().values(), [])
        if USE_PRESET_ASSETS else
        list(load_auto_symbol_mapping().values())
    )

    symbols = [standardize_symbol(s, load_auto_symbol_mapping()) for s in symbols]

    data_no_api = download_no_api_data(tuple(symbols))

    if not data_no_api:
        logging.info("⚠️ Nessun dato senza API, passo alle API.")
        data_api = await fetch_from_all_exchanges(symbols, DAYS_HISTORY)
        save_and_sync(data_api)
    else:
        save_and_sync(data_no_api)


if __name__ == "__main__":
    asyncio.run(main())
