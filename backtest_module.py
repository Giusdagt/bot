# demo_module.py
import logging
import random
import polars as pl
from pathlib import Path

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
        logging.warning(f"⚠️ Nessun dato per il demo trade su {symbol}.")
        return

    fake_profit = round(random.uniform(-5, 10), 2)
    result = {
        "symbol": symbol,
        "profit": fake_profit,
        "timestamp": pl.Series([pl.datetime_now()])
    }
    logging.info(f"🧪 Trade simulato per {symbol} | Profitto: {fake_profit} $")

    # Salvataggio su disco
    df = (
        pl.read_parquet(TRADE_FILE)
        if TRADE_FILE.exists()
        else pl.DataFrame({"symbol": [], "profit": [], "timestamp": []})
    )
    new_row = pl.DataFrame(result)
    df = pl.concat([df, new_row])
    df.write_parquet(TRADE_FILE, compression="zstd", mode="overwrite")


# backtest_module.py
import numpy as np
import random
import logging


def run_backtest(symbol, historical_data):
    """
    Esegue un backtest simulato sui dati storici del simbolo indicato.
    """
    if historical_data is None or historical_data.height < 50:
        logging.warning(f"⚠️ Dati insufficienti per il backtest di {symbol}.")
        return {
            "symbol": symbol,
            "win_rate": 0.0,
            "avg_profit": 0.0
        }

    # Simula risultati
    simulated_trades = 50
    profits = [random.uniform(-3, 6) for _ in range(simulated_trades)]
    win_rate = sum(1 for p in profits if p > 0) / simulated_trades
    avg_profit = np.mean(profits)

    logging.info(
        (
            f"📊 Backtest completato su {symbol} | "
            f"Win Rate: {win_rate:.2%} | Avg Profit: {avg_profit:.2f} $"
        )
    )
    return {
        "symbol": symbol,
        "win_rate": win_rate,
        "avg_profit": avg_profit
    }
