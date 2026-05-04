import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env.ride_share_env import RideShareEnv, NUM_AGENTS, GRID_SIZE
from explanations.explain import generate_step_explanations
from explanations.consistency_check import ConsistencyChecker

TIMESTEPS = 200_000
SAVE_PATH = "results/explainable_ppo"
CONSISTENCY_REWARD_WEIGHT = 0.5


class ExplainableAgentWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = RideShareEnv()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(202,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.checker = ConsistencyChecker()
        self._step_count = 0

    def reset(self, seed=None, options=None):
        self.obs, _ = self.env.reset(seed=seed)
        self.agent_list = list(self.obs.keys())
        self.agent_idx = 0
        self.checker = ConsistencyChecker()
        self._step_count = 0
        return self.obs[self.agent_list[self.agent_idx]], {}

    def step(self, action):
        agent = self.agent_list[self.agent_idx]

        actions = {a: self._greedy_action(a) for a in self.env.agents}
        actions[agent] = int(action)

        explanations = generate_step_explanations(self.env, actions)
        self.checker.log_step(explanations)

        self.obs, rewards, terminations, truncations, infos = self.env.step(actions)

        base_reward = rewards.get(agent, 0.0)

        consistency = self.checker.consistency_score()
        consistency_bonus = CONSISTENCY_REWARD_WEIGHT * consistency

        reward = base_reward + consistency_bonus

        terminated = terminations.get(agent, False)
        truncated = truncations.get(agent, False)
        done = terminated or truncated

        self._step_count += 1

        if not done and self.env.agents:
            self.agent_list = list(self.env.agents)
            self.agent_idx = (self.agent_idx + 1) % len(self.agent_list)
            next_agent = self.agent_list[self.agent_idx]
            obs = self.obs[next_agent]
        else:
            obs = np.zeros(202, dtype=np.float32)

        return obs, reward, terminated, truncated, infos

    def _greedy_action(self, agent):
        if not self.env.passengers:
            return 0
        row, col = self.env.positions[agent]
        nearest = min(self.env.passengers, key=lambda p: abs(p[0] - row) + abs(p[1] - col))
        dr = nearest[0] - row
        dc = nearest[1] - col
        if abs(dr) >= abs(dc):
            return 1 if dr < 0 else 2
        else:
            return 3 if dc < 0 else 4


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    env = ExplainableAgentWrapper()
    check_env(env, warn=True)

    print("Training explainable PPO agent with consistency reward...")
    print(f"Consistency reward weight: {CONSISTENCY_REWARD_WEIGHT}")
    print(f"Timesteps: {TIMESTEPS}\n")

    model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=64, n_epochs=5)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(SAVE_PATH)

    print(f"\nExplainable model saved to {SAVE_PATH}.zip")