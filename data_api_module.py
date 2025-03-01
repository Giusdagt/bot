# data_api_module.py
import json
import os
import logging
import sys
import aiohttp
import asyncio
import random
import shutil
from datetime import datetime
from data_loader import load_market_data_apis

# Impostazione del loop per Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configurazione del logging avanzato
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Numero di giorni di dati storici da scaricare
DAYS_HISTORY = 60  # Default: 60 giorni

# Caricare le API disponibili
services = load_market_data_apis()

# 📌 Backup dei dati in locale, USB o Cloud
STORAGE_PATH = "/mnt/usb_trading_data/market_data.json" if os.path.exists("/mnt/usb_trading_data") else "market_data.json"
CLOUD_BACKUP = "/mnt/google_drive/trading_backup/market_data.json"

# ===========================
# 🔹 GESTIONE API MULTI-EXCHANGE
# ===========================


async def fetch_data_from_exchanges(session, currency):
    """Scarica dati dai vari exchange e passa al successivo se l'API raggiunge il limite."""
    for exchange in services["exchanges"]:
        api_url = exchange["api_url"].replace("{currency}", currency)
        requests_per_minute = exchange["limitations"]["requests_per_minute"]

        logging.info("🔄 Tentando di recuperare dati da %s (%d req/min)...", exchange['name'], requests_per_minute)

        data = await fetch_market_data(session, api_url, requests_per_minute)
        if data:
            logging.info("✅ Dati ottenuti con successo da %s!", exchange['name'])
            return data

        logging.warning("⚠️ Limite raggiunto su %s. Passo al prossimo exchange...", exchange['name'])

    logging.error("❌ Nessun exchange disponibile ha fornito dati validi.")
    return None


async def fetch_market_data(session, url, requests_per_minute, retries=3):
    """Scarica i dati di mercato attuali con gestione avanzata degli errori."""
    delay = max(2, 60 / requests_per_minute)

    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status in {400, 429}:
                    wait_time = random.randint(20, 40)
                    logging.warning("⚠️ Errore %d. Attesa %d secondi prima di riprovare...", response.status, wait_time)
                    await asyncio.sleep(wait_time)
        except Exception as e:
            logging.error("❌ Errore nella richiesta API %s: %s", url, e)
            await asyncio.sleep(delay)

    return None


async def fetch_historical_data(session, coin_id, currency, days=DAYS_HISTORY, retries=3):
    """Scarica i dati storici con gestione avanzata degli errori."""
    for exchange in services["exchanges"]:
        historical_url = exchange["api_url"].replace("{currency}", currency).replace("{symbol}", coin_id)

        for attempt in range(retries):
            try:
                async with session.get(historical_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        return await response.json()
            except Exception as e:
                logging.error("❌ Errore nel recupero dati storici %s da %s: %s", coin_id, exchange['name'], e)
                await asyncio.sleep(2 ** attempt)

    return None


async def main_fetch_all_data(currency):
    """Scarica sia i dati di mercato attuali che quelli storici, con failover su più exchange."""
    async with aiohttp.ClientSession() as session:
        market_data = await fetch_data_from_exchanges(session, currency)

        if not market_data:
            logging.error("❌ Errore: dati di mercato non disponibili, uso backup.")
            market_data = load_backup("market_data_backup.json")

        final_data = []
        for crypto in market_data[:300]:
            coin_id = crypto.get("id")
            if not coin_id:
                continue

            historical_data = await fetch_historical_data(session, coin_id, currency)
            crypto["historical_prices"] = historical_data
            final_data.append(crypto)

        save_backup(final_data, STORAGE_PATH)
        sync_to_cloud()
        return final_data

# ===========================
# 🔹 GESTIONE BACKUP
# ===========================


def save_backup(data, filename):
    """Salva un backup locale, su USB e Cloud dei dati API."""
    with open(filename, "w", encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    logging.info("✅ Backup dati salvato in %s.", filename)


def load_backup(filename):
    """Carica i dati salvati in precedenza in caso di errore API."""
    if os.path.exists(filename):
        with open(filename, "r", encoding='utf-8') as file:
            return json.load(file)
    logging.warning("⚠️ Backup %s non trovato, impossibile recuperare dati.", filename)
    return []


def sync_to_cloud():
    """Sincronizza i dati di mercato con il Cloud se la USB non è disponibile."""
    if not os.path.exists(STORAGE_PATH):
        try:
            os.makedirs(os.path.dirname(CLOUD_BACKUP), exist_ok=True)
            shutil.copy(STORAGE_PATH, CLOUD_BACKUP)
            logging.info("☁️ Dati di mercato sincronizzati su Google Drive.")
        except Exception as e:
            logging.error("❌ Errore nel backup su Google Drive: %s", e)


if __name__ == "__main__":
    asyncio.run(main_fetch_all_data("eur"))
