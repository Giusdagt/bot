"""
drl_super_integration.py
Questo modulo fornisce l'integrazione per DRLSuperAgent,
un sistema avanzato di reinforcement learning
per la gestione autonoma di strategie di trading.
Include la classe DRLSuperManager, che gestisce
molteplici agenti RL (PPO, DQN, A2C, SAC),
addestrandoli e selezionando le migliori azioni
basate sugli stati del mercato.
Funzionalità principali:
- Addestramento e aggiornamento degli agenti RL in background.
- Selezione delle migliori azioni con confidenza associata.
- Ottimizzazione per ambienti di trading algoritmico.
- Caricamento e salvataggio automatico dei modelli.
"""

import logging
from pathlib import Path
import numpy as np
import threading
import time
import joblib
from drl_agent import DRLSuperAgent

MODEL_PATH = (
    Path("/mnt/usb_trading_data/models") if
    Path("/mnt/usb_trading_data").exists() else
    Path("D:/trading_data/models")
)
MODEL_PATH.mkdir(parents=True, exist_ok=True)


class DRLSuperManager:
    def __init__(self, state_size=512):
        self.super_agents = {
            "PPO": DRLSuperAgent(algo="PPO", state_size=state_size),
            "DQN": DRLSuperAgent(algo="DQN", state_size=state_size),
            "A2C": DRLSuperAgent(algo="A2C", state_size=state_size),
            "SAC": DRLSuperAgent(algo="SAC", state_size=state_size)
        }
        self.last_states = []
        self.state_size = state_size
        self.save_interval = 3600  # ogni ora
        logging.info(
            "🧠 DRLSuperManager inizializzato con 4 agenti RL"
        )

    def save_all(self):
        for name, agent in self.super_agents.items():
            path = MODEL_PATH / f"agent_{name}.joblib"
            joblib.dump(agent.drl_agent, path)
            logging.info(f"💾 Agente {name} salvato su disco.")

    def load_all(self):
        for name in self.super_agents:
            path = MODEL_PATH / f"agent_{name}.joblib"
            if path.exists():
                try:
                    self.super_agents[name].drl_agent = joblib.load(path)
                    logging.info(f"📂 Agente {name} caricato da disco.")
                except Exception as e:
                    logging.warning(
                        f"⚠️ Errore caricamento agente {name}: {e}"
                    )
            else:
                logging.info(
                    f"📁 Nessun modello per {name}, inizializzo da zero."
                )

    def update_all(self, full_state: np.ndarray, outcome: float):
        for name, agent in self.super_agents.items():
            logging.info(
                "Aggiornamento agente: %s con risultato: %f", name, outcome
            )
            agent.drl_agent.update(full_state, outcome)

    def get_best_action_and_confidence(self, full_state: np.ndarray):
        best = None
        best_confidence = -1
        best_algo = None

        for name, agent in self.super_agents.items():
            action, confidence = agent.predict(full_state.reshape(1, -1))
            if confidence > best_confidence:
                best_confidence = confidence
                best = action
                best_algo = name

        return best, best_confidence, best_algo

    def train_background(self, steps=5000):
        for name, agent in self.super_agents.items():
            agent.train(steps=steps)
            logging.info("🎯 Addestramento completato: %s", name)

    def reinforce_best_agent(self, full_state: np.ndarray, outcome: float):
        action, confidence, best_algo = (
            self.get_best_action_and_confidence(full_state)
        )
        if outcome > 0.5:
            logging.info(
                f"🎯 Rinforzo positivo su {best_algo} | Outcome: {outcome}"
            )
            self.super_agents[best_algo].drl_agent.update(full_state, outcome)
            self.super_agents[best_algo].train(steps=1000)
        else:
            logging.info(
                f"⚠️ Nessun rinforzo su {best_algo} (outcome: {outcome})"
            )

    def start_auto_training(self, interval_hours=6):
        def loop():
            while True:
                self.train_background(steps=5000)
                self.save_all()
                time.sleep(interval_hours * 3600)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
