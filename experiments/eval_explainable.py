import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from env.ride_share_env import RideShareEnv, MAX_STEPS, PASSENGERS_PER_STEP
from explanations.explain import generate_step_explanations
from explanations.consistency_check import ConsistencyChecker

EPISODES = 10

parser = argparse.ArgumentParser()
parser.add_argument("--weight", type=float, default=0.1, help="Consistency reward weight used during training")
args = parser.parse_args()

MODEL_PATH = f"results/explainable_ppo_w{args.weight}"

model = PPO.load(MODEL_PATH)
results = []

for ep in range(EPISODES):
    env = RideShareEnv()
    obs, _ = env.reset(seed=ep)
    checker = ConsistencyChecker()

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

        explanations = generate_step_explanations(env, actions)
        checker.log_step(explanations)

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
    consistency = checker.consistency_score()

    results.append({
        "pickups": total_pickups,
        "avg_wait": avg_wait,
        "response_rate": response_rate,
        "utilization": utilization,
        "consistency": consistency,
    })

    print(f"Episode {ep+1:>2} | Pickups: {total_pickups:>4} | Avg Wait: {avg_wait:.1f} steps | Response Rate: {response_rate:.2%} | Utilization: {utilization:.2%} | Consistency: {consistency:.2%}")

print(f"\n--- Explainable Model Results (weight={args.weight}, avg over 10 episodes) ---")
print(f"Average Pickups:         {np.mean([r['pickups'] for r in results]):.1f}")
print(f"Avg Wait Time:           {np.mean([r['avg_wait'] for r in results]):.2f} steps")
print(f"Order Response Rate:     {np.mean([r['response_rate'] for r in results]):.2%}")
print(f"Vehicle Utilization:     {np.mean([r['utilization'] for r in results]):.2%}")
print(f"Explanation Consistency: {np.mean([r['consistency'] for r in results]):.2%}")

checker.save(f"results/explainable_consistency_report_w{args.weight}.json")