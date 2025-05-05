"""
DRL avanzato + PPO/DQN/A2C/SAC con ReplayBuffer,
DummyVecEnv, Environment autonomo
DRLSuperAgent - Agente di Decisione Reinforcement Learning
Auto-Migliorante
"""

from pathlib import Path
import logging
import numpy as np
import polars as pl
import asyncio
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from data_handler import (
    process_historical_data, get_normalized_market_data,
    get_available_assets
)

print("drl_agent.py caricato ✅")

MODEL_PATH = Path(
    ("/mnt/usb_trading_data/models") if
    Path("/mnt/usb_trading_data").exists() else
    Path("D:/trading_data/models")
)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)

SEQUENCE_LENGTH = 10


async def load_data():
    """
    Carica i dati elaborati dal modulo data_handler.
    """
    await process_historical_data()
    if not Path("D:/trading_data/processed_data.zstd.parquet").exists():
        raise FileNotFoundError("Il file dei dati elaborati non esiste.")
    data = pl.read_parquet("D:/trading_data/processed_data.zstd.parquet")
    if data.is_empty():
        raise ValueError("Il file dei dati elaborati è vuoto.")
    return data


class GymTradingEnv(gym.Env):
    """
    Ambiente compatibile con Gym per simulazioni di trading.
    """
    def __init__(self, data, symbol, state_size=512, initial_balance=100):
        super().__init__()
        self.data = data.select(pl.col(pl.NUMERIC_DTYPES)).to_numpy()
        self.total_steps = len(self.data)
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.state_size = state_size
        self.current_step = SEQUENCE_LENGTH
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(SEQUENCE_LENGTH * self.data.shape[1],),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell

    def reset(self, seed=None, options=None):
        """
        Reimposta l'ambiente di simulazione di trading.
        Returns:
        np.ndarray: Un nuovo stato iniziale generato casualmente.
        """
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = SEQUENCE_LENGTH
        obs = self.data[
            self.current_step - SEQUENCE_LENGTH:self.current_step
        ].astype(np.float32).flatten()
        return obs, {}

    def step(self, action):
        """
        Esegue un passo nell'ambiente di simulazione di trading.
        Args:
        action (int): L'azione scelta
        (0 = hold, 1 = buy, 2 = sell).
        Returns:
        tuple: Una tupla contenente:
        - obs (np.ndarray): Il nuovo stato osservato.
        - reward (float): Il reward ottenuto dall'azione.
        - done (bool): Indica se l'episodio è terminato.
        - info (dict): Informazioni aggiuntive (vuoto in questo caso).
        """
        if self.current_step >= self.total_steps - 1:
            obs = self.data[
                self.current_step - SEQUENCE_LENGTH:self.current_step
            ].astype(np.float32).flatten()
            return obs, 0.0, True, False, {}
 
        price_now = self.data[self.current_step][0]
        price_next = self.data[self.current_step + 1][0]

        # Aggiorna la posizione in base all'azione
        if action == 1:  # Buy
            reward = price_next - price_now
            self.position = 1
        elif action == 2:  # Sell
            reward = price_now - price_next
            self.position = -1
        else:  # Hold
            reward = 0
            self.position = 0

        # Calcola il reward basato sulla posizione aggiornata
        reward = (price_now - price_next) * self.position
        self.balance += reward
        self.current_step += 1

        done = self.current_step >= self.total_steps - 1 or self.balance <= 0
        truncated = False  # oppure metti logica se serve

        obs = self.data[
            self.current_step - SEQUENCE_LENGTH:self.current_step
        ].astype(np.float32).flatten()

        return obs, reward, done, truncated, {}

    def render(self, mode='human'):
        """
        Mostra lo stato corrente dell'ambiente.
        """
        if mode != 'human':
            raise NotImplementedError(
                f"La modalità '{mode}' non è supportata."
            )
        print(
            f"""Step: {self.current_step},
            Balance: {self.balance},
            Position: {self.position}"""
            )


class DRLAgent:
    """
    DRLAgent - Agente di Decisione Reinforcement Learning Compresso
    """
    def __init__(self, state_size=512, max_memory=5000):
        self.state_size = state_size
        self.memory = []
        self.weights = np.random.normal(0, 0.1, state_size).astype(np.float32)
        self.max_memory = max_memory
        self.learning_rate = 0.01
        logging.info(
            "🧠 DRLAgent attivo | stato: %d, memoria max: %d",
            state_size,
            max_memory
        )

    def predict(self, state: np.ndarray) -> float:
        """
        Calcola una previsione basata sullo stato fornito.
        Args:
        state (np.ndarray): Lo stato attuale dell'ambiente.
        Returns:
        float: Valore previsto normalizzato tra 0 e 1.
        """
        value = np.dot(state, self.weights)
        return float(np.clip(1 / (1 + np.exp(-value)), 0, 1))

    def get_confidence(self, state: np.ndarray) -> float:
        """
        Restituisce la confidenza sulla predizione attuale.
        Basata su similarità tra stato attuale e memoria recente.
        Non cresce nel tempo.
        """
        if len(self.memory) < 10:
            return 0.1  # minima confidenza iniziale

        # Seleziona ultimi 10 stati
        recent_states = np.array([s for s, _ in self.memory[-10:]])
        dot_products = np.dot(recent_states, state)
        similarity = np.mean(dot_products)

        # Calcola varianza dei valori predetti
        predictions = [np.dot(s, self.weights) for s in recent_states]
        variance = np.var(predictions) + 1e-6

        # Formula di confidenza
        confidence = 1 / (1 + variance * (1 - similarity))
        return float(np.clip(confidence, 0.1, 1.0))

    def update(self, state: np.ndarray, outcome: float):
        """
        Aggiorna i pesi dell'agente DRL in base allo
        stato attuale e al risultato.
        Args:
        state (np.ndarray): Lo stato attuale dell'ambiente.
        outcome (float): Il risultato osservato
        (ad esempio, il reward).
        """
        self.memory.append((state, outcome))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
        gradient = (outcome - np.dot(state, self.weights)) * state
        self.weights += self.learning_rate * gradient
        self.weights = np.clip(self.weights, -2, 2).astype(np.float32)

    def compress_memory(self):
        """
        Riduce la memoria dell'agente mantenendo solo gli ultimi stati
        fino al limite massimo definito.
        """
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]

    def save(self):
        """
        Salva i pesi dell'agente DRL in un file compresso.
        """
        np.savez_compressed(
            str(MODEL_PATH / "agent_weights.npz"), weights=self.weights
        )

    def load(self):
        """
        Carica i pesi dell'agente DRL da un file salvato.
        """
        data = np.load(str(MODEL_PATH / "agent_weights.npz"))
        self.weights = data["weights"]


class DRLSuperAgent:
    """
    Agente DRL avanzato che supporta PPO, DQN, A2C, SAC con ambiente autonomo.
    """
    def __init__(self, state_size=512, action_space_type="discrete", env=None):
        self.state_size = state_size
        
        self.env = env or DummyVecEnv([lambda: GymTradingEnv(
            data=pl.DataFrame(np.zeros((SEQUENCE_LENGTH, state_size))),  # Dummy data
            symbol="DUMMY",  # simbolo fittizio
            state_size=state_size,
            initial_balance=100
        )])
        self.drl_agent = DRLAgent(state_size=state_size)

        # Seleziona automaticamente l'algoritmo
        self.algo = self._select_algorithm()
        self.model = self._init_model(self.algo)
        logging.info(f"Algoritmo selezionato: {self.algo}")

    def _select_algorithm(self):
        """
        Seleziona automaticamente l'algoritmo più adatto
        in base al tipo di spazio di azione.
        Returns:
        str: Nome dell'algoritmo selezionato.
        """
        space = self.env.envs[0].action_space
        logging.info(f"Tipo di spazio di azione: {type(space)}")

        if isinstance(space, spaces.Box):
            if self.state_size > 256:
                return "SAC"
            else:
                return "A2C"
        elif isinstance(space, spaces.Discrete):
            if self.state_size < 256:
                return "DQN"
            elif self.state_size <= 512:
                return "A2C"
            else:
                return "PPO"
        else:
            raise ValueError(
                f" spazio azione non supportato: {type(self.env.action_space)}"
            )

    def _init_model(self, algo):
        if algo == "PPO":
            return PPO(
                "MlpPolicy", self.env, verbose=0,
                policy_kwargs={"net_arch": [dict(pi=[128, 64], vf=[128, 64])]}
            )
        if algo == "DQN":
            return DQN(
                "MlpPolicy", self.env, verbose=0,
                policy_kwargs={"net_arch": [128, 64]}
            )
        if algo == "A2C":
            return A2C(
                "MlpPolicy", self.env, verbose=0,
                policy_kwargs={"net_arch": [dict(pi=[128, 64], vf=[128, 64])]}
            )
        if algo == "SAC":
            return SAC(
                "MlpPolicy", self.env, verbose=0,
                policy_kwargs={"net_arch": [128, 64]}
            )
        raise ValueError("Algoritmo non supportato")

    def train(self, steps=5000):
        """
        Addestra il modello DRL per un numero specificato di passi.
        Args:
        steps (int): Numero di passi di addestramento. Default è 5000.
        """
        self.model.learn(total_timesteps=steps, reset_num_timesteps=False)
        self.model.save(str(MODEL_PATH / f"{self.algo}_model"))
        self.drl_agent.compress_memory()
        self.drl_agent.save()
        logging.info("💪 %s aggiornato.", type(self.model).__name__)

    def predict(self, state):
        """
        Effettua una previsione basata sullo stato fornito.
        Args:
        state (np.ndarray): Lo stato attuale dell'ambiente.
        Returns:
        tuple: L'azione prevista (int)
        e la confidenza associata (float).
        """
        action, _ = self.model.predict(state, deterministic=True)
        confidence = self.drl_agent.predict(state)
        return int(action), confidence

    def save(self):
        """
        Salva il modello DRL e i pesi dell'agente DRL su file.
        """
        self.model.save(str(MODEL_PATH / f"{self.algo}_model"))
        self.drl_agent.save()

    def load(self):
        """
        Carica il modello DRL e i pesi dell'agente DRL
        da file salvati.
        """
        self.model.load(str(MODEL_PATH / f"{self.algo}_model"))
        self.drl_agent.load()


if __name__ == "__main__":
    from ai_model import AIModel, fetch_account_balances
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