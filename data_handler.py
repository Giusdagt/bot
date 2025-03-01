"""
data_handler.py
Modulo per la gestione dei dati di mercato, inclusa la normalizzazione,
il backup su cloud, l'elaborazione dei dati grezzi e il supporto al trading.
"""

import os
import shutil
import json
import logging
import asyncio
import websockets
import pandas as pd
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import data_api_module
from indicators import TradingIndicators
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# 📌 Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 Percorsi per backup e dati
SAVE_DIRECTORY = (
    "/mnt/usb_trading_data/processed_data"
    if os.path.exists("/mnt/usb_trading_data") else "D:/trading_data"
)
HISTORICAL_DATA_FILE = os.path.join(SAVE_DIRECTORY, "historical_data.parquet")
SCALPING_DATA_FILE = os.path.join(SAVE_DIRECTORY, "scalping_data.parquet")
RAW_DATA_FILE = "market_data.json"
MAX_AGE = 30 * 24 * 60 * 60  # 30 giorni in secondi
CLOUD_BACKUP = "/mnt/google_drive/trading_backup"

# 📌 WebSocket per dati in tempo reale (Scalping)
WEBSOCKET_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# 📌 Autenticazione Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
scaler = MinMaxScaler()


def upload_to_drive(filepath):
    """Carica un file su Google Drive."""
    try:
        file_drive = drive.CreateFile({'title': os.path.basename(filepath)})
        file_drive.SetContentFile(filepath)
        file_drive.Upload()
        logging.info("✅ File caricato su Google Drive: %s", filepath)
    except Exception as e:
        logging.error("❌ Errore caricamento Google Drive: %s", e)


def download_from_drive(filename, save_path):
    """Scarica un file da Google Drive."""
    try:
        file_list = drive.ListFile({'q': f"title = '{filename}'"}).GetList()
        if file_list:
            file_drive = file_list[0]
            file_drive.GetContentFile(save_path)
            logging.info("✅ File scaricato da Google Drive: %s", save_path)
    except Exception as e:
        logging.error("❌ Errore download Google Drive: %s", e)


def normalize_data(df):
    """Normalizza i dati di mercato."""
    try:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df
    except Exception as e:
        logging.error("❌ Errore normalizzazione dati: %s", e)
        return df


def load_processed_data(filename=HISTORICAL_DATA_FILE):
    """Carica i dati elaborati da un file parquet."""
    try:
        if os.path.exists(filename):
            return pd.read_parquet(filename)
        logging.warning("⚠️ Nessun file trovato: %s", filename)
        return pd.DataFrame()
    except Exception as e:
        logging.error("❌ Errore caricamento dati: %s", e)
        return pd.DataFrame()


def should_update_data(filename=HISTORICAL_DATA_FILE):
    """Verifica se i dati devono essere aggiornati."""
    if not os.path.exists(filename):
        return True
    file_age = time.time() - os.path.getmtime(filename)
    return file_age > MAX_AGE


def ensure_directory_exists(directory):
    """Crea la directory se non esiste."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def fetch_and_prepare_data():
    """Scarica, elabora e salva i dati di mercato."""
    try:
        if not should_update_data():
            logging.info("✅ Dati aggiornati. Carico i dati esistenti.")
            return load_processed_data()

        logging.info("📥 Avvio del processo di scaricamento ed elaborazione...")
        ensure_directory_exists(SAVE_DIRECTORY)

        if not os.path.exists(RAW_DATA_FILE):
            logging.warning(
                "⚠️ File dati di mercato non trovato. Scaricamento in corso..."
            )
            asyncio.run(data_api_module.main_fetch_all_data("eur"))

        return process_raw_data()
    except Exception as e:
        logging.error("❌ Errore durante il processo di dati: %s", e)
        return pd.DataFrame()


def process_raw_data():
    """Elabora i dati dal file JSON grezzo e li salva come parquet."""
    try:
        with open(RAW_DATA_FILE, "r") as json_file:
            raw_data = json.load(json_file)

        df_historical = pd.DataFrame(
            [{"timestamp": datetime.utcfromtimestamp(entry["timestamp"] / 1000),
              "coin_id": crypto.get("id", "unknown"),
              "close": entry["close"]}
             for crypto in raw_data for entry in crypto.get("historical_prices", [])]
        )
        df_historical.set_index("timestamp", inplace=True)
        df_historical.sort_index(inplace=True)
        df_historical = normalize_data(df_historical)

        save_processed_data(df_historical)
        return df_historical
    except Exception as e:
        logging.error("❌ Errore elaborazione dati grezzi: %s", e)
        return pd.DataFrame()


def save_processed_data(df, filename=HISTORICAL_DATA_FILE):
    """Salva i dati elaborati in formato parquet."""
    try:
        ensure_directory_exists(SAVE_DIRECTORY)
        df.to_parquet(filename, index=True)
        logging.info("✅ Dati salvati in: %s", filename)
    except Exception as e:
        logging.error("❌ Errore durante il salvataggio dei dati: %s", e)


async def process_websocket_message(message):
    """Elabora il messaggio ricevuto dal WebSocket per dati real-time."""
    try:
        data = json.loads(message)
        price = float(data["p"])
        timestamp = datetime.fromtimestamp(data["T"] / 1000.0)

        df = pd.DataFrame([[timestamp, price]],
                          columns=["timestamp", "price"])
        df.set_index("timestamp", inplace=True)

        # 📊 Calcolo indicatori tecnici
        df["rsi"] = TradingIndicators.relative_strength_index(df)
        df["macd"], df["macd_signal"] = (
            TradingIndicators.moving_average_convergence_divergence(df)
        )
        df["ema"] = TradingIndicators.exponential_moving_average(df)
        df["bollinger_upper"], df["bollinger_lower"] = (
            TradingIndicators.bollinger_bands(df)
        )

        # 📌 Normalizzazione e salvataggio dati
        df = normalize_data(df)
        save_processed_data(df, SCALPING_DATA_FILE)
        logging.info("✅ Dati scalping aggiornati: %s", df.tail(1))

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logging.error("❌ Errore elaborazione WebSocket: %s", e)


async def consume_websocket():
    """Consuma dati dal WebSocket per operazioni di scalping."""
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        logging.info("✅ Connessione WebSocket stabilita.")
        try:
            async for message in websocket:
                await process_websocket_message(message)
        except websockets.ConnectionClosed:
            logging.warning("⚠️ Connessione WebSocket chiusa. Riconnessione...")
            await asyncio.sleep(5)
            await consume_websocket()
        except Exception as e:
            logging.error("❌ Errore WebSocket: %s", e)
            await asyncio.sleep(5)
            await consume_websocket()


def backup_file(file_path):
    """Effettua il backup del file su Google Drive e USB."""
    try:
        # Copia il file su Google Drive
        upload_to_drive(file_path)
        # Copia il file su USB
        usb_path = os.path.join(SAVE_DIRECTORY, os.path.basename(file_path))
        shutil.copy(file_path, usb_path)
        logging.info("✅ Backup completato per %s", file_path)
    except Exception as e:
        logging.error("❌ Errore backup: %s", e)


def calculate_time_difference(timestamp):
    """Calcola la differenza di tempo dal timestamp fornito a ora."""
    try:
        now = datetime.utcnow()
        past = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        difference = now - past
        return timedelta(seconds=difference.total_seconds())
    except Exception as e:
        logging.error("❌ Errore calcolo differenza tempo: %s", e)
        return None


if __name__ == "__main__":
    logging.info("🔄 Avvio della sincronizzazione dei dati...")
    fetch_and_prepare_data()
    asyncio.run(consume_websocket())
