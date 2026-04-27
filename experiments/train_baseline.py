import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from models.baseline_agent import SingleAgentWrapper

TIMESTEPS = 200_000
SAVE_PATH = "results/baseline_ppo"

env = SingleAgentWrapper()
check_env(env, warn=True)

print("Training baseline PPO agent...")
model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=64, n_epochs=5)
model.learn(total_timesteps=TIMESTEPS)
model.save(SAVE_PATH)

print(f"\nModel saved to {SAVE_PATH}.zip")
