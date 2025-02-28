# data_handler.py
import os
import pandas as pd
import asyncio
import json
import logging
import websockets
import time
import shutil  # Aggiunto per la gestione dei backup
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import data_api_module
from indicators import TradingIndicators

# 📌 Configurazioni di salvataggio e backup
SAVE_DIRECTORY = (
    "/mnt/usb_trading_data/processed_data" if os.path.exists("/mnt/usb_trading_data")
    else "D:/trading_data/processed_data"
)
BACKUP_DIRECTORY = (
    "/mnt/usb_trading_data/backup_data" if os.path.exists("/mnt/usb_trading_data")
    else "D:/trading_data/backup_data"
)

HISTORICAL_DATA_FILE = os.path.join(SAVE_DIRECTORY, "historical_data.parquet")
SCALPING_DATA_FILE = os.path.join(SAVE_DIRECTORY, "scalping_data.parquet")
RAW_DATA_FILE = "market_data.json"
MAX_AGE = 30 * 24 * 60 * 60  # 30 giorni in secondi

# 📌 WebSocket URL per dati in tempo reale per scalping
WEBSOCKET_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# 📌 Configurazione logging avanzato
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 Creazione dello scaler per la normalizzazione dei dati
scaler = MinMaxScaler()


def load_data():
    """Carica i dati storici normalizzati dal file parquet."""
    if os.path.exists(HISTORICAL_DATA_FILE):
        return pd.read_parquet(HISTORICAL_DATA_FILE)
    return pd.DataFrame()


async def process_websocket_message(message):
    """Elabora il messaggio ricevuto dal WebSocket per dati real-time per scalping."""
    try:
        data = json.loads(message)
        price = float(data["p"])  # Prezzo dell'ultima operazione
        timestamp = datetime.fromtimestamp(data["T"] / 1000.0)

        df = pd.DataFrame([[timestamp, price]], columns=["timestamp", "price"])
        df.set_index("timestamp", inplace=True)

        # 📌 Calcolo indicatori tecnici in tempo reale per scalping
        df["rsi"] = TradingIndicators.relative_strength_index(df)
        df["macd"], df["macd_signal"] = (
            TradingIndicators.moving_average_convergence_divergence(df)
        )
        df["ema"] = TradingIndicators.exponential_moving_average(df)
        df["bollinger_upper"], df["bollinger_lower"] = (
            TradingIndicators.bollinger_bands(df)
        )

        # 📌 Normalizzazione dei dati per scalping
        df = normalize_data(df)

        # 📌 Salvataggio dati per scalping
        save_processed_data(df, SCALPING_DATA_FILE)
        logging.info(f"✅ Dati scalping aggiornati e salvati: {df.tail(1)}")

    except Exception as e:
        logging.error(f"❌ Errore nell'elaborazione del messaggio WebSocket: {e}")


async def consume_websocket():
    """Consuma dati dal WebSocket per operazioni di scalping."""
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        logging.info("✅ Connessione WebSocket stabilita per dati real-time.")
        try:
            async for message in websocket:
                await process_websocket_message(message)
        except websockets.ConnectionClosed:
            logging.warning(
                "⚠️ Connessione WebSocket chiusa. Riconnessione in corso..."
            )
            await asyncio.sleep(5)
            await consume_websocket()
        except Exception as e:
            logging.error(f"❌ Errore durante la ricezione dei dati WebSocket: {e}")
            await asyncio.sleep(5)
            await consume_websocket()


def normalize_data(df):
    """Normalizza i dati per il trading AI."""
    try:
        cols_to_normalize = [
            "close", "open", "high", "low", "volume", "rsi", "macd",
            "macd_signal", "ema", "bollinger_upper", "bollinger_lower"
        ]
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        return df
    except Exception as e:
        logging.error(f"❌ Errore durante la normalizzazione dei dati: {e}")
        return df


def save_processed_data(df, filename):
    """Salva i dati elaborati e gestisce il backup delle versioni precedenti."""
    try:
        if os.path.exists(filename):
            backup_file = os.path.join(
                BACKUP_DIRECTORY, f"backup_{int(time.time())}.parquet"
            )
            shutil.move(filename, backup_file)
            logging.info(f"🛠 Backup creato: {backup_file}")

        df.to_parquet(filename)
        logging.info(f"📂 Dati salvati in {filename}")

        # 📌 Pulizia backup vecchi
        clean_old_backups(BACKUP_DIRECTORY)

    except Exception as e:
        logging.error(f"❌ Errore nel salvataggio dei dati: {e}")


def clean_old_backups(directory, max_files=5):
    """Elimina le versioni più vecchie dei backup mantenendo solo le ultime."""
    try:
        backups = sorted(
            [os.path.join(directory, f) for f in os.listdir(directory)],
            key=os.path.getmtime,
        )
        while len(backups) > max_files:
            os.remove(backups.pop(0))
            logging.info(f"🗑 Backup vecchio eliminato.")

    except Exception as e:
        logging.error(f"❌ Errore nella gestione dei backup: {e}")
