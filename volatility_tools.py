"""
volatility_tools.py
"""
import numpy as np


class VolatilityPredictor:
    def __init__(self):
        # inizializza il modello se ne hai uno
        pass

    def predict_volatility(self, features: np.ndarray):
        # Placeholder — usa il tuo vero modello se necessario
        return np.std(features, axis=1)
