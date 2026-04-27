import numpy as np
import sys
import os
from gymnasium import spaces
from pettingzoo import ParallelEnv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ride-share-marl", "pipeline"))
from demand_generator import get_spawn_weights

GRID_SIZE = 10
NUM_AGENTS = 20
MAX_STEPS = 200
PASSENGERS_PER_STEP = 3


class RideShareEnv(ParallelEnv):
    metadata = {"name": "ride_share_v0"}

    def __init__(self):
        self.possible_agents = [f"driver_{i}" for i in range(NUM_AGENTS)]
        self.spawn_weights = get_spawn_weights().flatten()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0

        # Agent positions: random grid cells
        self.positions = {
            a: [np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)]
            for a in self.agents
        }

        # Passengers: set of (row, col) tuples
        self.passengers = set()
        self._spawn_passengers()

        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        return {a: self._observe(a) for a in self.agents}, self.infos

    def step(self, actions):
        self.rewards = {a: -0.01 for a in self.agents}  # idle penalty

        for agent, action in actions.items():
            row, col = self.positions[agent]

            if action == 1:   row = max(0, row - 1)       # up
            elif action == 2: row = min(GRID_SIZE-1, row + 1)  # down
            elif action == 3: col = max(0, col - 1)       # left
            elif action == 4: col = min(GRID_SIZE-1, col + 1)  # right

            self.positions[agent] = [row, col]

            if (row, col) in self.passengers:
                self.passengers.remove((row, col))
                self.rewards[agent] += 1.0

        self._spawn_passengers()
        self.step_count += 1

        done = self.step_count >= MAX_STEPS
        self.truncations = {a: done for a in self.agents}

        if done:
            self.agents = []

        obs = {a: self._observe(a) for a in self.agents}
        return obs, self.rewards, self.terminations, self.truncations, self.infos

    def _spawn_passengers(self):
        cells = np.random.choice(GRID_SIZE * GRID_SIZE, size=PASSENGERS_PER_STEP, p=self.spawn_weights)
        for cell in cells:
            self.passengers.add((cell // GRID_SIZE, cell % GRID_SIZE))

    def _observe(self, agent):
        row, col = self.positions[agent]

        passenger_grid = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
        for (r, c) in self.passengers:
            passenger_grid[r * GRID_SIZE + c] = 1.0

        agent_grid = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
        for a, (r, c) in self.positions.items():
            if a != agent:
                agent_grid[r * GRID_SIZE + c] = 1.0

        pos = np.array([row / (GRID_SIZE - 1), col / (GRID_SIZE - 1)], dtype=np.float32)
        return np.concatenate([pos, passenger_grid, agent_grid])

    def observation_space(self, agent):
        return spaces.Box(low=0.0, high=1.0, shape=(202,), dtype=np.float32)

    def action_space(self, agent):
        return spaces.Discrete(5)
