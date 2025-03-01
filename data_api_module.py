# data_api_module.py

import json
import os
import logging
import sys
import aiohttp
import asyncio
import random
import shutil
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
STORAGE_PATH = (
    "/mnt/usb_trading_data/market_data.json"
    if os.path.exists("/mnt/usb_trading_data") else "market_data.json"
)
CLOUD_SYNC_PATH = "/mnt/google_drive/trading_sync/market_data.json"

# ===========================
# 🔹 GESTIONE API MULTI-EXCHANGE
# ===========================


async def fetch_data_from_exchanges(session, currency):
    """Scarica dati dai vari exchange con gestione dinamica dei limiti API."""
    tasks = []
    exchange_limits = {}

    for exchange in services["exchanges"]:
        api_url = exchange["api_url"].replace("{currency}", currency)
        req_per_min = exchange["limitations"].get("requests_per_minute", 60)
        exchange_limits[exchange["name"]] = req_per_min
        tasks.append(
            fetch_market_data(
                session, api_url, exchange["name"], req_per_min
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [data for data in results if data is not None]


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
                    logging.info("✅ Dati ottenuti da %s", exchange_name)
                    return await response.json()
                if response.status in {400, 429}:  # Bad Request / Troppe richieste
                    wait_time = random.randint(10, 30)
                    logging.warning(
                        "⚠️ Errore %d su %s. Attesa %d sec prima di riprovare...",
                        response.status, exchange_name, wait_time
                    )
                    await asyncio.sleep(wait_time)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.error(
                "❌ Errore richiesta API %s su %s: %s",
                url, exchange_name, e
            )
            await asyncio.sleep(delay)
    return None


async def fetch_historical_data(
    session, coin_id, currency, days=DAYS_HISTORY, retries=3
):
    """Scarica i dati storici con gestione avanzata degli errori."""
    for exchange in services["exchanges"]:
        historical_url = (
            exchange["api_url"]
            .replace("{currency}", currency)
            .replace("{symbol}", coin_id)
        )
        for attempt in range(retries):
            try:
                async with session.get(
                    historical_url, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
            except Exception as e:
                logging.error(
                    "❌ Errore nel recupero dati storici %s da %s: %s",
                    coin_id, exchange['name'], e
                )
                await asyncio.sleep(2 ** attempt)

    return None


async def main_fetch_all_data(currency):
    """Scarica i dati di mercato con rispetto automatico dei limiti API."""
    async with aiohttp.ClientSession() as session:
        market_data = await fetch_data_from_exchanges(session, currency)

        if not market_data:
            logging.error("❌ Nessun dato di mercato disponibile.")
            return None

        tasks = [
            fetch_historical_data(session, crypto.get("id"), currency)
            for crypto in market_data[:300] if crypto.get("id")
        ]
        historical_data_list = await asyncio.gather(*tasks)

        final_data = []
        for crypto, historical_data in zip(
            market_data[:300], historical_data_list
        ):
            crypto["historical_prices"] = historical_data
            final_data.append(crypto)

        save_and_sync(final_data, STORAGE_PATH)
        return final_data


# ===========================
# 🔹 GESTIONE SINCRONIZZAZIONE
# ===========================


def save_and_sync(data, filename):
    """Salva e sincronizza i dati solo se necessario."""
    with open(filename, "w", encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    logging.info("✅ Dati aggiornati in %s.", filename)
    sync_to_cloud()


def sync_to_cloud():
    """Sincronizza i dati con Google Drive solo se il file è cambiato."""
    if os.path.exists(STORAGE_PATH):
        try:
            os.makedirs(os.path.dirname(CLOUD_SYNC_PATH), exist_ok=True)
            shutil.copy(STORAGE_PATH, CLOUD_SYNC_PATH)
            logging.info("☁️ Dati sincronizzati su Google Drive.")
        except Exception as e:
            logging.error("❌ Errore nella sincronizzazione con Google Drive: %s", e)


if __name__ == "__main__":
    asyncio.run(main_fetch_all_data("eur"))
