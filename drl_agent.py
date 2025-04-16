""" 
DRL avanzato + PPO/DQN/A2C/SAC con ReplayBuffer,
DummyVecEnv, Environment autonomo
DRLSuperAgent - Agente di Decisione Reinforcement Learning
Auto-Migliorante 
"""

import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
import logging

MODEL_PATH = Path(
    ("/mnt/usb_trading_data/models") if
    Path("/mnt/usb_trading_data").exists() else
    Path("D:/trading_data/models")
)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)


class GymTradingEnv(gym.Env):
    """
    Ambiente compatibile con Gym per simulazioni di trading.
    """
    def __init__(self, state_size=512):
        super(GymTradingEnv, self).__init__()
        self.state_size = state_size
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(state_size,), dtype=np.float32
        )
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return np.random.uniform(
            -1, 1, size=self.state_size
        ).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        reward = np.random.uniform(-1, 1) * (action - 1)  # semplicesimulazione
        done = self.current_step > 100
        obs = np.random.uniform(-1, 1, size=self.state_size).astype(np.float32)
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass


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
            f"🧠 DRLAgent attivo | stato: {state_size}, memoria max: {max_memory}"
        )

    def predict(self, state: np.ndarray) -> float:
        value = np.dot(state, self.weights)
        return float(np.clip(1 / (1 + np.exp(-value)), 0, 1))

    def update(self, state: np.ndarray, outcome: float):
        self.memory.append((state, outcome))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
        gradient = (outcome - np.dot(state, self.weights)) * state
        self.weights += self.learning_rate * gradient
        self.weights = np.clip(self.weights, -2, 2).astype(np.float32)

    def compress_memory(self):
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]

    def save(self):
        np.savez_compressed(
            str(MODEL_PATH / "agent_weights.npz"), weights=self.weights
        )

    def load(self):
        data = np.load(str(MODEL_PATH / "agent_weights.npz"))
        self.weights = data["weights"]


class DRLSuperAgent:
    """
    Agente DRL avanzato che supporta PPO, DQN, A2C, SAC con ambiente autonomo.
    """
    def __init__(self, algo="PPO", state_size=512):
        self.state_size = state_size
        self.env = DummyVecEnv([lambda: GymTradingEnv(state_size=state_size)])
        self.model = self._init_model(algo)
        self.drl_agent = DRLAgent(state_size=state_size)

    def _init_model(self, algo):
        if algo == "PPO":
            return PPO("MlpPolicy", self.env, verbose=0)
        elif algo == "DQN":
            return DQN("MlpPolicy", self.env, verbose=0)
        elif algo == "A2C":
            return A2C("MlpPolicy", self.env, verbose=0)
        elif algo == "SAC":
            return SAC("MlpPolicy", self.env, verbose=0)
        else:
            raise ValueError("Algoritmo non supportato")

    def train(self, steps=100_000):
        self.model.learn(total_timesteps=steps)
        self.save()

    def predict(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        confidence = self.drl_agent.predict(state)
        return int(action), confidence

    def save(self):
        self.model.save(str(MODEL_PATH / "super_agent_model"))
        self.drl_agent.save()

    def load(self):
        self.model.load(str(MODEL_PATH / "super_agent_model"))
        self.drl_agent.load()


if __name__ == "__main__":
    agent = DRLSuperAgent(algo="PPO")
    agent.train(steps=200_000)
    print("✅ DRL super agent addestrato e salvato.")
