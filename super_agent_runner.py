"""
Questo modulo esegue l'addestramento automatico
del DRLSuperAgent utilizzando diversi algoritmi
di reinforcement learning. Il processo viene eseguito
in un thread separato e ripetuto ogni 6 ore.
"""
import threading
import random
import time
from drl_agent import DRLSuperAgent

print("super_agent_runner.py caricato✅")


def auto_train_super_agent():
    """
    Esegue l'addestramento continuo del DRLSuperAgent
    utilizzando algoritmi di reinforcement learning.
    Gli algoritmi vengono scelti casualmente tra PPO, DQN, A2C e SAC.
    L'addestramento viene ripetuto ogni 6 ore.
    """
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
