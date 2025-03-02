import asyncio
import json
import logging
import os
import random
import shutil
import sys
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor
from data_loader import load_market_data_apis

# Impostazione del loop per Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configurazione del logging avanzato
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Numero di giorni di dati storici da scaricare
DAYS_HISTORY = 60

# Caricare le API disponibili
services = load_market_data_apis()

# 📌 Percorsi per la sincronizzazione dei dati
STORAGE_PATH = "market_data.json"
CLOUD_SYNC_PATH = "/mnt/google_drive/trading_sync/market_data.json"


# 📌 Scarica dati senza API in parallelo, senza modificare la logica originale
def download_no_api_data(symbols=["BTCUSDT"], interval="1d"):
    sources = services["data_sources"]["no_api"]
    data = {}

    def fetch_data(source_name, url, symbol):
        response = requests.get(url)
        if response.status == 200:
            if symbol not in data:
                data[symbol] = {}
            data[symbol][source_name] = url
            logging.info("✅ Dati %s scaricati per %s", source_name, symbol)

    with ThreadPoolExecutor(max_workers=5) as executor:
        for symbol in symbols:
            executor.submit(
                fetch_data,
                "binance_data",
                f"{sources['binance_data']}/{symbol}/{interval}/"
                f"{symbol}-{interval}.zip",
                symbol
            )
            executor.submit(
                fetch_data,
                "cryptodatadownload",
                f"{sources['cryptodatadownload']}/Binance_{symbol}_d.csv",
                symbol
            )

    return data


async def fetch_data_from_exchanges(
    session, currency="usdt", min_volume=5000000
):
    """Scarica solo coppie USDT con volume alto per ridurre i dati."""
    tasks = []
    exchange_limits = {}
    for exchange in services["exchanges"]:
        api_url = exchange["api_url"].replace("{currency}", currency)
        req_per_min = exchange["limitations"].get("requests_per_minute", 60)
        exchange_limits[exchange["name"]] = req_per_min
        tasks.append(
            fetch_market_data(session, api_url, exchange["name"], req_per_min)
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    filtered_results = [
        data for data in results
        if data is not None and data.get("total_volume", 0) >= min_volume
    ]

    return filtered_results[:300]


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
                    logging.info("✅ Dati ottenuti da %s al tentativo %d", exchange_name, attempt + 1)
                    return await response.json()
                if response.status in {400, 429}:  # Troppe richieste
                    wait_time = random.randint(10, 30)
                    logging.warning(
                        "⚠️ Errore %d su %s. Attesa %d sec per riprovare...",
                        response.status, exchange_name, wait_time
                    )
                    await asyncio.sleep(wait_time)
        except (aiohttp.ClientError, asyncio.TimeoutError) as client_error:
            logging.error(
                "❌ Errore richiesta API %s su %s al tentativo %d: %s",
                url, exchange_name, attempt + 1, client_error
            )
            await asyncio.sleep(delay)
    return None


def save_and_sync(data, filename):
    """Salva i dati senza modificare la logica originale."""
    with open(filename, "w", encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    logging.info("✅ Dati aggiornati in %s.", filename)
    sync_to_cloud()


def sync_to_cloud():
    """Sincronizza i dati con Google Drive solo se necessario."""
    if os.path.exists(STORAGE_PATH):
        try:
            os.makedirs(os.path.dirname(CLOUD_SYNC_PATH), exist_ok=True)
            shutil.copy(STORAGE_PATH, CLOUD_SYNC_PATH)
            logging.info("☁️ Dati sincronizzati su Google Drive.")
        except OSError as sync_error:
            logging.error(
                "❌ Errore nella sincronizzazione con Google Drive: %s",
                sync_error
            )


def main():
    logging.info("🔄 Avvio aggiornamento dati...")
    data_no_api = download_no_api_data()

    if not data_no_api:
        logging.warning(
            "⚠️ Nessun dato trovato senza API. Passaggio ai dati via API..."
        )
        asyncio.run(fetch_data_from_exchanges(aiohttp.ClientSession()))

    save_and_sync(data_no_api, STORAGE_PATH)
    logging.info("✅ Processo completato.")


if __name__ == "__main__":
    main()
