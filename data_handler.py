"""
data_handler.py normalizzazione dei dati
Gestione avanzata dei dati di mercato,
per IA, Deep Reinforcement Learning (DRL)
e WebSocket, con massima efficienza su CPU, RAM e Disco.
"""

import os
import logging
import asyncio
import gc
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import websockets
import polars as pl
from sklearn.preprocessing import MinMaxScaler
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import data_api_module
from indicators import (
    calculate_scalping_indicators,
    calculate_historical_indicators
)
from column_definitions import required_columns

# 📌 Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 Percorsi per sincronizzazione e dati compressi
SAVE_DIRECTORY = "/mnt/usb_trading_data/processed_data" if os.path.exists(
    "/mnt/usb_trading_data"
) else "D:/trading_data"
HISTORICAL_DATA_FILE = os.path.join(
    SAVE_DIRECTORY, "historical_data.zstd.parquet"
)
SCALPING_DATA_FILE = os.path.join(
    SAVE_DIRECTORY, "scalping_data.zstd.parquet"
)
RAW_DATA_FILE = "market_data.parquet"
CLOUD_SYNC = "/mnt/google_drive/trading_sync"

# 📌 WebSocket per dati in tempo reale (Scalping)
TOP_PAIRS = data_api_module.get_top_usdt_pairs()
WEBSOCKET_URLS = [
    f"wss://stream.binance.com:9443/ws/{pair.lower()}@trade"
    for pair in TOP_PAIRS
]

# 📌 Autenticazione Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
scaler = MinMaxScaler()

# 📌 Multi-threading per parallelizzazione
executor = ThreadPoolExecutor(max_workers=4)

# 📌 Cache & Buffer per massima velocità
cache_data = {}
buffer = []
BUFFER_SIZE = 100  # Ottimizzazione salvataggio per batch

# 📌 Definizione costante RETRY_DELAY
RETRY_DELAY = 1


def upload_to_drive(filepath):
    """Sincronizza un file su Google Drive solo se necessario."""
    try:
        if os.path.exists(filepath):
            file_drive = drive.CreateFile(
                {'title': os.path.basename(filepath)}
            )
            file_drive.SetContentFile(filepath)
            file_drive.Upload()
            logging.info("✅ File sincronizzato su Google Drive: %s", filepath)
    except IOError as e:
        logging.error("❌ Errore sincronizzazione Google Drive: %s", e)


async def process_websocket_message(message, pair):
    """Elabora e normalizza il messaggio ricevuto dal WebSocket."""
    try:
        df = pl.DataFrame([
            {
                "timestamp": datetime.utcnow(),
                "pair": pair,
                "price": float(message["p"]),
                "volume": float(message["q"])
            }
        ])
        df = calculate_scalping_indicators(df)  # Indicatori specifici per scalping
        buffer.append(df)

        if len(buffer) >= BUFFER_SIZE:
            df_batch = pl.concat(buffer)
            await asyncio.get_event_loop().run_in_executor(
                executor, save_processed_data, df_batch, SCALPING_DATA_FILE
            )
            buffer.clear()
            gc.collect()
            logging.info(
                "✅ Dati scalping aggiornati batch di %d messaggi", BUFFER_SIZE
            )
    except (ValueError, KeyError) as e:
        logging.error("❌ Errore elaborazione WebSocket: %s", e)


async def consume_websockets():
    """Consuma dati da più WebSocket con gestione CPU/RAM ottimizzata."""
    global RETRY_DELAY
    max_retry_delay = 30

    async def connect_to_websocket(url):
        global RETRY_DELAY
        while True:
            try:
                async with websockets.connect(url, timeout=10) as websocket:
                    logging.info("✅ Connessione WebSocket stabilita: %s", url)
                    RETRY_DELAY = 1
                    pair = url.split("/")[-1].split("@")[0].upper()
                    async for message in websocket:
                        await process_websocket_message(message, pair)
                        await asyncio.sleep(0.05)  # Riduce consumo CPU
            except websockets.ConnectionClosed:
                logging.warning(
                    "⚠️ WebSocket %s disconnesso. Riconnessione in %d sec...",
                    url, RETRY_DELAY
                )
                await asyncio.sleep(RETRY_DELAY)
                RETRY_DELAY = min(RETRY_DELAY * 2, max_retry_delay)
            except (websockets.WebSocketException, OSError) as e:
                logging.error(
                    "❌ Errore WebSocket %s: %s. Riprovo in %d sec...",
                    url, e, RETRY_DELAY
                )
                await asyncio.sleep(RETRY_DELAY)
                RETRY_DELAY = min(RETRY_DELAY * 2, max_retry_delay)

    await asyncio.gather(
        *[connect_to_websocket(url) for url in WEBSOCKET_URLS]
    )


if __name__ == "__main__":
    logging.info("🔄 Avvio sincronizzazione dati...")
    asyncio.run(consume_websockets())
