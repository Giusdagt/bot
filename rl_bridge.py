"""
Modulo ponte per evitare import ciclici tra ai_model,
drl_agent, drl_super_integration e position_manager.
"""

from drl_super_integration import DRLSuperManager
import asyncio
import logging
from ai_model import AIModel, fetch_account_balances
from data_handler import get_available_assets, get_normalized_market_data
from drl_agent import DRLSuperAgent, GymTradingEnv, DRLAgent

if __name__ == "__main__":
    try:
        data = asyncio.run(load_data())
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
                from stable_baselines3.common.vec_env import DummyVecEnv
                env = DummyVecEnv([lambda: env_raw])

                agent_discrete = DRLSuperAgent(
                    state_size=512, action_space_type="discrete", env=env
                )
                agent_discrete.train(steps=200_000)

                agent_continuous = DRLSuperAgent(
                    state_size=512, action_space_type="continuous", env=env
                )
                agent_continuous.train(steps=200_000)

            except Exception as e:
                logging.error(f"⚠️ Errore su {symbol}: {e}")
        print("✅ Agenti DRL addestrati e salvati")
    except Exception as e:
        logging.error(f"Errore nel main: {e}")
