"""
volatility_tools.py
"""
import numpy as np


class VolatilityPredictor:
    """
    Predizione della volatilità intelligente.
    Usa fallback con std, ma può anche apprendere nel tempo.
    """
    def __init__(self):
        self.memory = []  # solo array, senza file
        self.max_size = 100
        self.weights = np.random.rand(10)  # pseudo-rete base con 10 input
        self.trained = False

    def update(self, features: np.ndarray, target_volatility: float):
        """
        Aggiorna i pesi in modo semplice, senza crescere.
        """
        if len(self.memory) >= self.max_size:
            self.memory.pop(0)
        self.memory.append((features, target_volatility))

        if len(self.memory) >= 10:
            feature_matrix = np.array([f for f, _ in self.memory])
            y = np.array([v for _, v in self.memory])
            # pseudo addestramento: regressione lineare normalizzata
            self.weights = np.linalg.pinv(feature_matrix) @ y
            self.trained = True

    def predict_volatility(self, features: np.ndarray):
        """
        Predice la volatilità. Se non è addestrato, fallback std.
        """
        if not self.trained:
            return np.std(features, axis=1)
        return features @ self.weights  # prodotto scalare
