# drl_agent.py - Agente DRL Compresso e Autonomo
import numpy as np
import logging


class DRLAgent:
    def __init__(self):
        self.memory = []  # memoria locale compressa
        self.weights = None  # vettore compresso di apprendimento
        self.max_memory = 500  # massimo numero di stati
        self.trained = False
        logging.info(
          "🧠 DRLAgent inizializzato (memoria compressa e autonoma)"
        )

    def update(self, state: np.ndarray, reward: float):
        """Aggiorna la memoria e i pesi su base compressa."""
        self.memory.append((state, reward))
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]

        X = np.array([s for s, _ in self.memory])
        y = np.array([r for _, r in self.memory])

        if len(X) >= X.shape[1]:
            self.weights = np.linalg.pinv(X) @ y
            self.trained = True

    def predict(self, symbol: str, state: np.ndarray) -> float:
        """Predice la probabilità di successo da 0 a 1."""
        if self.weights is None:
            return 0.5  # fallback se non addestrato
        value = np.dot(state, self.weights)
        return float(np.clip(1 / (1 + np.exp(-value)), 0, 1))

    def get_confidence(self, symbol: str, state: np.ndarray) -> float:
        """Restituisce un punteggio di confidenza tra 0.1 e 1.0"""
        if self.weights is None:
            return 0.1
        variance = np.var(
          [np.dot(s, self.weights) for s, _ in self.memory]
        ) + 1e-6
        distance = np.linalg.norm(
          state - self.memory[-1][0]
        ) if self.memory else 1.0
        confidence = 1 / (1 + variance * distance)
        return float(np.clip(confidence, 0.1, 1.0))


if __name__ == "__main__":
    agent = DRLAgent()
    dummy_state = np.random.rand(300)
    for i in range(50):
        reward = np.random.uniform(-1, 1)
        agent.update(dummy_state, reward)

    print("✅ Probabilità:", agent.predict("BTCUSDT", dummy_state))
    print("✅ Confidenza:", agent.get_confidence("BTCUSDT", dummy_state))
