# DRLSuperAgent Integration for AIModel (Autonomous, Compressed Learning)
# File: drl_super_integration.py

import logging
from pathlib import Path
import numpy as np
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
