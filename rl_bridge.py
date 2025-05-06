"""
Modulo ponte per evitare import ciclici tra ai_model,
drl_agent, drl_super_integration e position_manager
"""
import asyncio
import logging
import sys
from stable_baselines3.common.vec_env import DummyVecEnv
from drl_agent import GymTradingEnv, DRLAgent, DRLSuperAgent
from data_handler import (
    get_normalized_market_data, get_available_assets,
    process_historical_data
)
from ai_model import AIModel, fetch_account_balances

logging.basicConfig(level=logging.INFO)

async def load_data():
    await process_historical_data()
    return True  # placeholder per future espansioni

if __name__ == "__main__":
    try:
        # Carica i dati elaborati e i bilanci
        asyncio.run(load_data())
        balances = fetch_account_balances()

        all_assets = get_available_assets()
        market_data = {
            symbol: get_normalized_market_data(symbol)
            for symbol in all_assets
        }

        ai_model = AIModel(market_data, balances)

        for symbol in ai_model.active_assets:
            try:
                env_raw = GymTradingEnv(
                    data=market_data[symbol],
                    symbol=symbol
                )
                env = DummyVecEnv([lambda: env_raw])

                agent_discrete = DRLSuperAgent(
                    state_size=512, env=env
                )
                agent_discrete.train(steps=200_000)

                agent_continuous = DRLSuperAgent(
                    state_size=512, env=env
                )
                agent_continuous.train(steps=200_000)

            except Exception as e:
                logging.error("⚠️ Errore su %s: %s", symbol, e)

        print("✅ Agenti DRL addestrati e salvati")

    except Exception as e:
        logging.error("Errore nel main: %s", e)
        sys.exit(1)

