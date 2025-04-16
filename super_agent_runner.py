import threading
import random
import time
from drl_agent import DRLSuperAgent


def auto_train_super_agent():
    algos = ["PPO", "DQN", "A2C", "SAC"]
    while True:
        algo = random.choice(algos)
        print(
          f"🧠 Avvio addestramento DRLSuperAgent con {algo}..."
        )
        agent = DRLSuperAgent(algo=algo)
        agent.train(steps=50_000)  # Addestramento leggero ma continuo
        print(f"✅ {algo} addestrato e salvato.")
        time.sleep(6 * 3600)  # Ripeti ogni 6 ore


if __name__ == "__main__":
    thread = threading.Thread(
      target=auto_train_super_agent, daemon=True
    )
    thread.start()

    while True:
        time.sleep(3600)
