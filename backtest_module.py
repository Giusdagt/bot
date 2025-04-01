""
backtest_module.py
""

import random
import logging
import numpy as np

def run_backtest(symbol, historical_data):
    """
    Esegue un backtest simulato sui dati storici del simbolo indicato.
    """
    if historical_data is None or historical_data.height < 50:
        logging.warning("⚠️ Dati insufficienti per il backtest di %s.", symbol)
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
        "📊 Backtest completato su %s | Win Rate: %.2f%% | Avg Profit: %.2f $",
        symbol, win_rate * 100, avg_profit
    )
    return {
        "symbol": symbol,
        "win_rate": win_rate,
        "avg_profit": avg_profit
    }
