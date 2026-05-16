import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import defaultdict
from stable_baselines3 import PPO
from env.ride_share_env import RideShareEnv, MAX_STEPS, PASSENGERS_PER_STEP
from lime import lime_tabular

EPISODES = 3
MODEL_PATH = "results/baseline_ppo"

feature_names = (
    ["own_row", "own_col"] +
    [f"passenger_cell_{i}" for i in range(100)] +
    [f"other_driver_cell_{i}" for i in range(100)]
)

action_labels = ["stay", "up", "down", "left", "right"]


def make_predict_fn(model):
    def predict_fn(observations):
        probs = []
        for obs in observations:
            action, _ = model.predict(obs.astype(np.float32), deterministic=True)
            dist = np.zeros(5)
            dist[int(action)] = 1.0
            probs.append(dist)
        return np.array(probs)
    return predict_fn


class LimeConsistencyChecker:
    def __init__(self):
        self.state_to_explanations = defaultdict(list)
        self.total_logged = 0

    def state_signature(self, driver_pos, passenger_positions):
        sorted_passengers = tuple(sorted([tuple(p) for p in passenger_positions]))
        return (tuple(driver_pos), sorted_passengers)

    def log(self, sig, top_feature):
        self.state_to_explanations[sig].append(top_feature)
        self.total_logged += 1

    def consistency_score(self):
        if not self.state_to_explanations:
            return 0.0
        consistent = 0
        total = 0
        for sig, features in self.state_to_explanations.items():
            if len(features) < 2:
                continue
            total += len(features)
            most_common = max(set(features), key=features.count)
            consistent += features.count(most_common)
        if total == 0:
            return 1.0
        return consistent / total

    def summary(self):
        return {
            "lime_consistency_score": round(self.consistency_score(), 4),
            "total_explanations_logged": self.total_logged,
            "unique_states_seen": len(self.state_to_explanations),
        }

    def print_summary(self):
        s = self.summary()
        print("\n--- LIME Consistency Report ---")
        print(f"LIME Consistency Score:     {s['lime_consistency_score']:.2%}")
        print(f"Total Explanations Logged:  {s['total_explanations_logged']}")
        print(f"Unique States Seen:         {s['unique_states_seen']}")


def run_lime_eval():
    model = PPO.load(MODEL_PATH)
    predict_fn = make_predict_fn(model)

    background = np.random.uniform(0, 1, size=(200, 202)).astype(np.float32)
    explainer = lime_tabular.LimeTabularExplainer(
        background,
        feature_names=feature_names,
        class_names=action_labels,
        mode="classification"
    )

    results = []

    for ep in range(EPISODES):
        env = RideShareEnv()
        obs, _ = env.reset(seed=ep)
        checker = LimeConsistencyChecker()

        total_pickups = 0
        total_wait = 0
        total_steps_with_passenger = {a: 0 for a in env.agents}
        passenger_spawn_step = {}
        step = 0

        for p in env.passengers:
            passenger_spawn_step[p] = 0

        while env.agents:
            actions = {}
            for agent in env.agents:
                action, _ = model.predict(obs[agent], deterministic=True)
                actions[agent] = int(action)

            for agent in env.agents:
                observation = obs[agent]
                action_taken = actions[agent]

                if agent == "driver_0":
                    try:
                        explanation = explainer.explain_instance(
                            observation,
                            predict_fn,
                            num_features=1,
                            labels=[action_taken]
                        )
                        exp_list = explanation.as_list(label=action_taken)
                        top_feature = exp_list[0][0] if exp_list else "unknown"
                    except Exception:
                        top_feature = "unknown"

                    driver_pos = list(env.positions[agent])
                    passenger_positions = list(env.passengers)
                    sig = checker.state_signature(driver_pos, passenger_positions)
                    checker.log(sig, top_feature)

            prev_passengers = set(env.passengers)
            obs, rewards, terminations, truncations, _ = env.step(actions)
            step += 1

            for p in env.passengers:
                if p not in passenger_spawn_step:
                    passenger_spawn_step[p] = step

            picked_up = prev_passengers - env.passengers
            for p in picked_up:
                if p in passenger_spawn_step:
                    total_wait += step - passenger_spawn_step[p]
                    total_pickups += 1

            for agent, r in rewards.items():
                if r > 0:
                    total_steps_with_passenger[agent] += 1

        total_spawned = MAX_STEPS * PASSENGERS_PER_STEP
        avg_wait = total_wait / total_pickups if total_pickups > 0 else float("inf")
        response_rate = total_pickups / total_spawned
        utilization = sum(total_steps_with_passenger.values()) / (len(total_steps_with_passenger) * MAX_STEPS)
        lime_consistency = checker.consistency_score()

        results.append({
            "pickups": total_pickups,
            "avg_wait": avg_wait,
            "response_rate": response_rate,
            "utilization": utilization,
            "lime_consistency": lime_consistency,
        })

        print(f"Episode {ep+1:>2} | Pickups: {total_pickups:>4} | Avg Wait: {avg_wait:.1f} | Response Rate: {response_rate:.2%} | LIME Consistency: {lime_consistency:.2%}")

    print("\n--- LIME Eval Results (avg over 3 episodes) ---")
    print(f"Average Pickups:        {np.mean([r['pickups'] for r in results]):.1f}")
    print(f"Avg Wait Time:          {np.mean([r['avg_wait'] for r in results]):.2f} steps")
    print(f"Order Response Rate:    {np.mean([r['response_rate'] for r in results]):.2%}")
    print(f"Vehicle Utilization:    {np.mean([r['utilization'] for r in results]):.2%}")
    print(f"LIME Consistency Score: {np.mean([r['lime_consistency'] for r in results]):.2%}")

    print("\n--- Comparison ---")
    print(f"Rule-Based Consistency: 97.06%")
    print(f"LIME Consistency:       {np.mean([r['lime_consistency'] for r in results]):.2%}")


if __name__ == "__main__":
    run_lime_eval()