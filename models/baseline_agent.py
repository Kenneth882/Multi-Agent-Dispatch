import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.ride_share_env import RideShareEnv, NUM_AGENTS, GRID_SIZE


class SingleAgentWrapper(gym.Env):
    """
    Wraps RideShareEnv so stable-baselines3 PPO can train on it.
    PPO controls all agents by treating each step as N sequential decisions.
    """

    def __init__(self):
        super().__init__()
        self.env = RideShareEnv()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(202,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        self.obs, _ = self.env.reset(seed=seed)
        self.agent_list = list(self.obs.keys())
        self.agent_idx = 0
        return self.obs[self.agent_list[self.agent_idx]], {}

    def step(self, action):
        agent = self.agent_list[self.agent_idx]

        # Build actions: PPO controls current agent, others move greedily
        actions = {a: self._greedy_action(a) for a in self.env.agents}
        actions[agent] = int(action)

        self.obs, rewards, terminations, truncations, infos = self.env.step(actions)

        reward = rewards.get(agent, 0.0)
        terminated = terminations.get(agent, False)
        truncated = truncations.get(agent, False)
        done = terminated or truncated

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
