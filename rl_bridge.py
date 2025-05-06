"""
Modulo ponte per evitare import ciclici tra ai_model,
drl_agent, drl_super_integration e position_manager
"""
from data_handler import (
    process_historical_data, get_normalized_market_data,
    get_available_assets
)
from ai_model import AIModel, fetch_account_balances

if __name__ == "__main__":
    try:
        # Carica i dati elaborati e i bilanci
        data = asyncio.run(load_data())
        balances = fetch_account_balances()

        # Recupera tutti gli asset e i relativi dati di mercato
        all_assets = get_available_assets()
        market_data = {
            symbol: get_normalized_market_data(symbol)
            for symbol in all_assets
        }

        # Inizializza il modello AI per ottenere gli asset attivi
        ai_model = AIModel(market_data, balances)

    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Errore durante il caricamento dei dati: {e}")
        exit(1)

    # Addestramento DRL solo sugli asset attivi
    for symbol in ai_model.active_assets:
        try:
            env_raw = GymTradingEnv(
                data=market_data[symbol],
                symbol=symbol
            )
            env = DummyVecEnv([lambda: env_raw])

            # Addestra un agente con spazio discreto
            agent_discrete = DRLSuperAgent(
                state_size=512, action_space_type="discrete", env=env
            )
            agent_discrete.train(steps=200_000)

            # Addestra un agente con spazio continuo
            agent_continuous = DRLSuperAgent(
                state_size=512, action_space_type="continuous", env=env
            )
            agent_continuous.train(steps=200_000)

        except Exception as e:
            logging.error(f"⚠️ Errore su {symbol}: {e}")

    print("✅ Agenti DRL addestrati e salvati")
