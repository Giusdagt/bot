"""
data_handler.py - Normalizzazione e gestione avanzata dei dati
per IA, Deep Reinforcement Learning (DRL) e WebSocket, 
con massima efficienza su CPU, RAM e Disco.
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

# 📌 Buffer per efficienza
buffer = []
BUFFER_SIZE = 100

# 📌 Costante per retry delay WebSocket
MAX_RETRY_DELAY = 30


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
        df = calculate_scalping_indicators(df)
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
    async def connect_to_websocket(url):
        retry_delay = 1
        while True:
            try:
                async with websockets.connect(url, timeout=10) as websocket:
                    logging.info("✅ Connessione WebSocket stabilita: %s", url)
                    retry_delay = 1
                    pair = url.split("/")[-1].split("@")[0].upper()
                    async for message in websocket:
                        await process_websocket_message(message, pair)
                        await asyncio.sleep(0.05)
            except (
                websockets.ConnectionClosed,
                websockets.WebSocketException,
                OSError
            ) as e:
                logging.warning(
                    "⚠️ WebSocket %s disconnesso. Riconnessione in %d sec...",
                    url, retry_delay
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)

    await asyncio.gather(
        *[connect_to_websocket(url) for url in WEBSOCKET_URLS]
    )


def process_historical_data():
    """Processa e normalizza i dati storici."""
    try:
        df = pl.read_parquet(RAW_DATA_FILE)
        df = calculate_historical_indicators(df)

        # 📌 Verifica che tutte le colonne necessarie siano presenti
        for col in required_columns:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))

        save_processed_data(df, HISTORICAL_DATA_FILE)
        logging.info("✅ Dati storici aggiornati.")
    except Exception as e:
        logging.error("❌ Errore elaborazione dati storici: %s", e)


def save_processed_data(df, filename):
    """Salva in formato Parquet con compressione ZSTD solo se necessario."""
    try:
        if os.path.exists(filename):
            df_old = pl.read_parquet(filename)
            if df.equals(df_old):
                logging.info("🔄 Nessuna modifica, skip del salvataggio.")
                return

        df.write_parquet(filename, compression="zstd")
        logging.info("✅ Dati salvati con compressione ZSTD: %s", filename)
        sync_to_cloud()
    except IOError as e:
        logging.error("❌ Errore nel salvataggio dati: %s", e)


if __name__ == "__main__":
    logging.info("🔄 Avvio sincronizzazione dati...")

    # 📌 Esegui il processo storico in parallelo con WebSocket
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, process_historical_data)
    loop.run_until_complete(consume_websockets())
