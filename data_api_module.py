"""
data_api_module.py
Modulo ultra-avanzato, dinamico e intelligente per download dati di mercato.
Performance massimizzata, completamente automatico e intelligente.
"""

import asyncio
import logging
import os
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import aiohttp
import requests
import polars as pl
from data_loader import (
    load_market_data_apis,
    load_auto_symbol_mapping,
    standardize_symbol,
    USE_PRESET_ASSETS,
    load_preset_assets,
)
from column_definitions import required_columns

print("data_api_module.py caricato ✅")

ENABLE_CLOUD_SYNC = False  # Imposta su True per attivare la sincronizzazione

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Determina la directory dello script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STORAGE_PATH = os.path.join(SCRIPT_DIR, "market_data.zstd.parquet")
CLOUD_SYNC_PATH = "/mnt/google_drive/trading_sync/market_data.zstd.parquet"
DAYS_HISTORY = 60

executor = ThreadPoolExecutor(max_workers=8)

def ensure_permissions(file_path):
    """Garantisce che il file abbia i permessi di lettura e scrittura."""
    try:
        # Controlla se il file esiste
        if not os.path.exists(file_path):
            # Crea un file vuoto se non esiste
            with open(file_path, 'w'):
                pass
        else:
            # Controlla se il file è vuoto
            if os.path.getsize(file_path) == 0:
                logging.warning("⚠️ File vuoto rilevato, ricreazione in corso: %s", file_path)
                with open(file_path, 'w'):
                    pass

        # Imposta i permessi di lettura e scrittura
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        logging.info("✅ Permessi garantiti per il file: %s", file_path)
    except Exception as e:
        logging.error("❌ Errore nell'impostare i permessi per %s: %s", file_path, e)

def ensure_all_columns(df):
    """Garantisce presenza di tutte le colonne necessarie."""
    existing_columns = df.columns
    missing_columns = [
        col for col in required_columns if col not in existing_columns]
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
    logging.info(
        "📡 Inizio download dati per i simboli: %s", symbols
    )
    market_data_apis = load_market_data_apis()
    tasks = []
    async with aiohttp.ClientSession() as session:
        for exchange in market_data_apis["exchanges"]:
            for symbol in symbols:
                api_url = ( exchange["api_url"].replace(
                    "{symbol}", symbol).replace("{days}", str(days_history))
                )
                rpm = exchange["limitations"].get("requests_per_minute", 60)
                tasks.append(
                    fetch_market_data(session, api_url, exchange["name"], rpm)
                )
        results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_data = []
    for data in results:
        if isinstance(data, (dict, list)):
            valid_data.extend(data if isinstance(data, list) else [data])

    logging.info("✅ Dati scaricati con successo: %s", valid_data)
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

    with ThreadPoolExecutor(max_workers=8) as local_executor:
        futures = [
            local_executor.submit(
                fetch,
                symbol,
                source_name,
                (
                    f"{sources[source_name]}/{symbol}/"
                    f"{interval}/{symbol}-{interval}.zip"
                )
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
    else:
        logging.info("✅ Dati da salvare: %s", data)

    # Garantisce i permessi per il file di salvataggio
    ensure_permissions(STORAGE_PATH)

    df_new = pl.DataFrame(data)
    df_new = ensure_all_columns(df_new)

    if os.path.exists(STORAGE_PATH) and os.path.getsize(STORAGE_PATH) > 0:
        existing_df = pl.read_parquet(STORAGE_PATH)
        df_final = pl.concat([existing_df, df_new]).unique()
    else:
        df_final = df_new

    try:
        df_final.write_parquet(STORAGE_PATH, compression="zstd")
        logging.info("✅ Dati salvati ultra-veloce: %s", STORAGE_PATH)

        # Sincronizzazione con il cloud (solo se abilitata)
        if ENABLE_CLOUD_SYNC:
            sync_to_cloud()
    except (OSError, IOError) as e:
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
    """
    Funzione principale per il download e la sincronizzazione dei dati di mercato.
    """
    data_api = None  # Inizializza la variabile
    try:
        symbols = (
            sum(load_preset_assets().values(), [])
            if USE_PRESET_ASSETS else
            list(load_auto_symbol_mapping().values())
        )
        logging.info("✅ Simboli caricati: %s", symbols)

        if not symbols:
            logging.error("❌ Nessun simbolo caricato. Operazione terminata.")
            return

        symbols = [standardize_symbol(s, load_auto_symbol_mapping()) for s in symbols]

        # Prova a scaricare i dati senza API
        data_no_api = download_no_api_data(tuple(symbols))
        if not data_no_api:
            logging.warning("⚠️ Nessun dato senza API disponibile.")

        # Se i dati senza API non sono disponibili, usa le API
        if not data_no_api:
            logging.info("⚠️ Nessun dato senza API, passo alle API.")
            data_api = await fetch_from_all_exchanges(symbols, DAYS_HISTORY)

        # Controlla se ci sono dati da salvare
        if not data_no_api and not data_api:
            logging.error("❌ Nessun dato disponibile da fonti no-API o API.")
            return

        # Salva e sincronizza i dati
        if data_no_api:
            save_and_sync(data_no_api)
        elif data_api:
            save_and_sync(data_api)
        else:
            logging.warning("⚠️ Nessun dato valido da salvare.")

        # Log finale per confermare i dati scaricati
        logging.info("✅ Dati API scaricati: %s", data_api)

    except Exception as e:
        logging.error("❌ Errore durante l'esecuzione: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
