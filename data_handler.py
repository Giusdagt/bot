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
import websockets
import polars as pl
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import MinMaxScaler
from indicators import calculate_indicators
from column_definitions import required_columns
import data_api_module
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# 📌 Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 Percorsi per sincronizzazione e dati compressi
SAVE_DIRECTORY = "/mnt/usb_trading_data/processed_data" if os.path.exists(
    "/mnt/usb_trading_data") else "D:/trading_data"
HISTORICAL_DATA_FILE = os.path.join(
    SAVE_DIRECTORY, "historical_data.zstd.parquet")
SCALPING_DATA_FILE = os.path.join(
    SAVE_DIRECTORY, "scalping_data.zstd.parquet")
RAW_DATA_FILE = "market_data.parquet"
CLOUD_SYNC = "/mnt/google_drive/trading_sync"

# 📌 WebSocket per dati in tempo reale (Scalping)
TOP_PAIRS = data_api_module.get_top_usdt_pairs()  # Ottiene coppie dinamiche
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
cache_data = {}  # Evita rielaborazioni
buffer = []
BUFFER_SIZE = 100  # Ottimizzazione salvataggio per batch


def upload_to_drive(filepath):
    """Sincronizza un file su Google Drive solo se necessario."""
    try:
        if os.path.exists(filepath):
            file_drive = drive.CreateFile({'title': os.path.basename(filepath)})
            file_drive.SetContentFile(filepath)
            file_drive.Upload()
            logging.info("✅ File sincronizzato su Google Drive: %s", filepath)
    except IOError as e:
        logging.error("❌ Errore sincronizzazione Google Drive: %s", e)


async def process_websocket_message(message, pair):
    """Elabora e normalizza il messaggio ricevuto dal WebSocket."""
    try:
        df = pl.DataFrame([{
            "timestamp": datetime.utcnow(),
            "pair": pair,
            "price": float(message["p"]),
            "volume": float(message["q"])
        }])
        df = calculate_indicators(df)  # Calcola indicatori real-time
        buffer.append(df)

        if len(buffer) >= BUFFER_SIZE:
            df_batch = pl.concat(buffer)
            await asyncio.get_event_loop().run_in_executor(
                executor, save_processed_data, df_batch, SCALPING_DATA_FILE)
            buffer.clear()
            gc.collect()
            logging.info(
                "✅ Dati scalping aggiornati con batch di %d messaggi", BUFFER_SIZE)
    except (ValueError, KeyError) as e:
        logging.error("❌ Errore elaborazione WebSocket: %s", e)


async def consume_websockets():
    """Consuma dati da più WebSocket con gestione CPU/RAM ottimizzata."""
    global retry_delay
    retry_delay = 1
    max_retry_delay = 30

    async def connect_to_websocket(url):
        global retry_delay
        while True:
            try:
                async with websockets.connect(url, timeout=10) as websocket:
                    logging.info("✅ Connessione WebSocket stabilita: %s", url)
                    retry_delay = 1
                    pair = url.split("/")[-1].split("@")[0].upper()
                    async for message in websocket:
                        await process_websocket_message(message, pair)
                        await asyncio.sleep(0.05)  # Riduce consumo CPU
            except websockets.ConnectionClosed:
                logging.warning(
                    "⚠️ WebSocket %s disconnesso. Riconnessione in %d sec...", url, retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
            except Exception as e:
                logging.error(
                    "❌ Errore WebSocket %s: %s. Riprovo in %d sec...", url, e, retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

    await asyncio.gather(*[connect_to_websocket(url) for url in WEBSOCKET_URLS])


def normalize_data(df):
    """Normalizza i dati di mercato e garantisce tutte le colonne."""
    try:
        for col in required_columns:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        df = calculate_indicators(df)
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64]]
        df = df.with_columns([df[col].cast(pl.Float64) for col in numeric_cols])
        gc.collect()
        return df
    except (ValueError, KeyError) as e:
        logging.error("❌ Errore normalizzazione dati: %s", e)
        return df


def save_processed_data(df, filename):
    """Salva dati in formato Parquet con compressione ZSTD solo se necessario."""
    try:
        if os.path.exists(filename):
            df_old = pl.read_parquet(filename)
            if df.equals(df_old):
                logging.info("🔄 Nessuna modifica, skip del salvataggio.")
                return

        df.write_parquet(filename, compression="zstd")
        logging.info("✅ Dati salvati con compressione ZSTD: %s", filename)
        sync_to_cloud()
    except Exception as e:
        logging.error("❌ Errore nel salvataggio dati: %s", e)


def sync_to_cloud():
    """Sincronizza i dati con Google Drive solo se il file è cambiato."""
    if os.path.exists(HISTORICAL_DATA_FILE):
        try:
            cloud_file = CLOUD_SYNC + "/" + os.path.basename(HISTORICAL_DATA_FILE)
            if os.path.exists(cloud_file):
                local_size = os.path.getsize(HISTORICAL_DATA_FILE)
                cloud_size = os.path.getsize(cloud_file)
                if abs(local_size - cloud_size) < 1024 * 50:
                    logging.info("🔄 Nessuna modifica, salto sincronizzazione.")
                    return
            shutil.copy(HISTORICAL_DATA_FILE, CLOUD_SYNC)
            logging.info("☁️ Dati sincronizzati su Google Drive.")
        except OSError as sync_error:
            logging.error("❌ Errore nella sincro con Google Drive: %s", sync_error)


if __name__ == "__main__":
    logging.info("🔄 Avvio sincronizzazione dati...")
    asyncio.run(consume_websockets())
