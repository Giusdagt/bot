"""
Modulo demo_module.py
Questo modulo fornisce funzionalità per simulare operazioni di trading 
senza inviarle realmente. È utile per test, training automatico 
o quando le credenziali di trading non sono disponibili.
Funzionalità principali:
- Simulazione di trade con profitti casuali.
- Salvataggio dei dati delle operazioni simulate in un file Parquet.
"""
import logging
import random
from pathlib import Path
import polars as pl

MODEL_DIR = (
    Path("/mnt/usb_trading_data/models")
    if Path("/mnt/usb_trading_data").exists()
    else Path("D:/trading_data/models")
)
TRADE_FILE = MODEL_DIR / "demo_trades.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def demo_trade(symbol, market_data):
    """
    Simula un'operazione senza inviarla realmente.
    Ideale per training automatico o mancanza di credenziali.
    """
    if market_data is None or market_data.height == 0:
        logging.warning("⚠️ Nessun dato per il demo trade su %s.", symbol)
        return

    fake_profit = round(random.uniform(-5, 10), 2)
    result = {
        "symbol": symbol,
        "profit": fake_profit,
        "timestamp": pl.Series([pl.datetime_now()])
    }
    logging.info(
        "🧪 Trade simulato per %s | Profitto: %.2f $", symbol, fake_profit
    )

    # Salvataggio su disco
    df = (
        pl.read_parquet(TRADE_FILE)
        if TRADE_FILE.exists()
        else pl.DataFrame({"symbol": [], "profit": [], "timestamp": []})
    )
    new_row = pl.DataFrame(result)
    df = pl.concat([df, new_row])
    df.write_parquet(TRADE_FILE, compression="zstd", mode="overwrite")
