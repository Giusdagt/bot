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
import threading
import time
import numpy as np
import joblib
from drl_agent import DRLSuperAgent

print("drl_super_integration.py caricato ✅")

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
            "PPO": DRLSuperAgent(state_size=state_size),
            "DQN": DRLSuperAgent(state_size=state_size),
            "A2C": DRLSuperAgent(state_size=state_size),
            "SAC": DRLSuperAgent(state_size=state_size)
        }
        self.last_states = []
        self.state_size = state_size
        self.save_interval = 3600  # ogni ora
        logging.info(
            "🧠 DRLSuperManager inizializzato con 4 agenti RL"
        )

    def save_all(self):
        """
        Salva tutti gli agenti RL su disco.
        Ogni agente viene salvato come file .joblib
        nella directory specificata.
        """
        for name, agent in self.super_agents.items():
            path = MODEL_PATH / f"agent_{name}.joblib"
            joblib.dump(agent.drl_agent, path)
            logging.info("💾 Agente %s salvato su disco.", name)

    def load_all(self):
        """
        Carica gli agenti RL salvati su disco.
        Se un file modello esiste, l'agente viene caricato.
        In caso contrario, viene inizializzato da zero.
        """
        for name, agent in self.super_agents.items():
            path = MODEL_PATH / f"agent_{name}.joblib"
            if path.exists():
                try:
                    agent.drl_agent = joblib.load(path)
                    logging.info("📂 Agente %s caricato da disco.", name)
                except FileNotFoundError as e:
                    logging.warning(
                        "⚠️ Errore caricamento agente %s: %s", name, e
                    )
            else:
                logging.info(
                    "📁 Nessun modello per %s, inizializzo da zero.",
                    name
                )

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
        """Addestramento in background."""
        for name, agent in self.super_agents.items():
            agent.train(steps=steps)
            logging.info("🎯 Addestramento completato: %s", name)

    def reinforce_best_agent(self, full_state: np.ndarray, outcome: float):
        """
        Rinforza l'agente che ha preso l'azione migliore
        in base allo stato attuale.
        Usa anche il livello di confidenza
        per decidere se rinforzare o no.
        """
        action, confidence, best_algo = (
            self.get_best_action_and_confidence(full_state)
        )
        if action in [1, 2] and confidence > 0.7 and outcome > 0.5:
            logging.info(
                "🎯 Rinforzo positivo su %s | Outcome: %.2f | "
                "Confidenza: %.2f | Azione: %s",
                best_algo,
                outcome,
                confidence,
                action
            )
            self.super_agents[best_algo].drl_agent.update(full_state, outcome)
            self.super_agents[best_algo].train(steps=1000)
        else:
            logging.info(
                "⚠️ Rinforzo ignorato su %s | Outcome: %.2f | "
                "Confidenza: %.2f | Azione: %s",
                best_algo,
                outcome,
                confidence,
                action
            )

    def start_auto_training(self, interval_hours=6):
        """
        Avvia un processo in background
        per addestrare gli agenti RL
        automaticamente a intervalli regolari.
        Argomenti:
        interval_hours (int): Intervallo di tempo in ore
        tra un ciclo di addestramento e l'altro.
        """
        def loop():
            while True:
                self.train_background(steps=5000)
                self.save_all()
                time.sleep(interval_hours * 3600)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
