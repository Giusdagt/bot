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
"""
import logging
from pathlib import Path
import numpy as np
import threading
import time
from drl_agent import DRLSuperAgent

MODEL_PATH = (
    Path("/mnt/usb_trading_data/models") if
    Path("/mnt/usb_trading_data").exists() else
    Path("D:/trading_data/models")
)
MODEL_PATH.mkdir(parents=True, exist_ok=True)


class DRLSuperManager:
    """
    DRLSuperManager: Wrapper per integrare
    DRLSuperAgent nel sistema AI principale.
    Addestra e aggiorna autonomamente PPO/DQN/A2C/SAC
    su array compressi senza occupare risorse.
    """
    def __init__(self, state_size=512):
        self.super_agents = {
            "PPO": DRLSuperAgent(algo="PPO", state_size=state_size),
            "DQN": DRLSuperAgent(algo="DQN", state_size=state_size),
            "A2C": DRLSuperAgent(algo="A2C", state_size=state_size),
            "SAC": DRLSuperAgent(algo="SAC", state_size=state_size)
        }
        self.last_states = []
        self.state_size = state_size
        logging.info("🧠 DRLSuperManager inizializzato con 4 agenti RL")

    def update_all(self, full_state: np.ndarray, outcome: float):
        """Aggiorna tutti gli agenti con lo stato attuale e il risultato."""
        for name, agent in self.super_agents.items():
            logging.info(
                "Aggiornamento agente: %s con risultato: %f", name, outcome
            )
            agent.drl_agent.update(full_state, outcome)

    def get_best_action_and_confidence(self, full_state: np.ndarray):
        """
        Seleziona l'azione migliore tra tutti gli agenti.
        Restituisce: (azione, confidenza, nome_modello)
        """
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
        """Addestramento leggero in background (chiamato in loop)."""
        for name, agent in self.super_agents.items():
            agent.train(steps=steps)
            logging.info("🎯 Addestramento completato: %s", name)

    def reinforce_best_agent(self, full_state: np.ndarray, outcome: float):
        """
        Addestra solo l'agente che ha preso l'azione migliore.
        Ultra leggero e preciso.
        """
        action, confidence, best_algo = self.get_best_action_and_confidence(full_state)
        logging.info(f"🎯 Allenamento mirato su {best_algo}")
        self.super_agents[best_algo].train(steps=1000)

    def start_auto_training(self, interval_hours=6):
        """
        Avvia il training continuo in background ogni X ore.
        """
        def loop():
            while True:
                self.train_background(steps=5000)
                time.sleep(interval_hours * 3600)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
