import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from env.ride_share_env import RideShareEnv, MAX_STEPS, PASSENGERS_PER_STEP
from explanations.explain import generate_step_explanations
from explanations.consistency_check import ConsistencyChecker

EPISODES = 10
MODEL_PATH = "results/baseline_ppo"

model = PPO.load(MODEL_PATH)
results = []

for ep in range(EPISODES):
    env = RideShareEnv()
    obs, _ = env.reset(seed=ep)
    checker = ConsistencyChecker()  # fresh checker per episode

    total_pickups = 0
    total_wait = 0
    total_steps_with_passenger = {a: 0 for a in env.agents}
    passenger_spawn_step = {}
    passenger_spawn_pos = {}
    prev_positions = {}
    total_actual_dist = 0
    total_optimal_dist = 0
    step = 0

    for p in env.passengers:
        passenger_spawn_step[p] = 0
        passenger_spawn_pos[p] = p

    while env.agents:
        actions = {}
        for agent in env.agents:
            action, _ = model.predict(obs[agent], deterministic=True)
            actions[agent] = int(action)

        prev_positions = {a: list(env.positions[a]) for a in env.agents}
        explanations = generate_step_explanations(env, actions)
        checker.log_step(explanations)

        prev_passengers = set(env.passengers)
        obs, rewards, terminations, truncations, _ = env.step(actions)
        step += 1

        for p in env.passengers:
            if p not in passenger_spawn_step:
                passenger_spawn_step[p] = step
                passenger_spawn_pos[p] = p

        picked_up = prev_passengers - env.passengers
        for p in picked_up:
            if p in passenger_spawn_step:
                total_wait += step - passenger_spawn_step[p]
                total_pickups += 1
            for agent in env.possible_agents:
                if tuple(env.positions.get(agent, p)) == p or tuple(prev_positions.get(agent, ())) == tuple(p):
                    actual = abs(prev_positions.get(agent, [p[0], p[1]])[0] - p[0]) + \
                             abs(prev_positions.get(agent, [p[0], p[1]])[1] - p[1]) + 1
                    optimal = 1
                    total_actual_dist += actual
                    total_optimal_dist += optimal
                    break

        for agent, r in rewards.items():
            if r > 0:
                total_steps_with_passenger[agent] += 1

    total_spawned = MAX_STEPS * PASSENGERS_PER_STEP
    avg_wait = total_wait / total_pickups if total_pickups > 0 else float("inf")
    response_rate = total_pickups / total_spawned
    utilization = sum(total_steps_with_passenger.values()) / (len(total_steps_with_passenger) * MAX_STEPS)
    detour_ratio = total_actual_dist / total_optimal_dist if total_optimal_dist > 0 else float("inf")
    consistency = checker.consistency_score()

    results.append({
        "pickups": total_pickups,
        "avg_wait": avg_wait,
        "response_rate": response_rate,
        "utilization": utilization,
        "detour_ratio": detour_ratio,
        "consistency": consistency,
    })

    print(f"Episode {ep+1:>2} | Pickups: {total_pickups:>4} | Avg Wait: {avg_wait:.1f} | Response Rate: {response_rate:.2%} | Utilization: {utilization:.2%} | Detour: {detour_ratio:.2f} | Consistency: {consistency:.2%}")

print("\n--- Baseline Results (avg over 10 episodes) ---")
print(f"Average Pickups:         {np.mean([r['pickups'] for r in results]):.1f}")
print(f"Avg Wait Time:           {np.mean([r['avg_wait'] for r in results]):.2f} steps")
print(f"Order Response Rate:     {np.mean([r['response_rate'] for r in results]):.2%}")
print(f"Vehicle Utilization:     {np.mean([r['utilization'] for r in results]):.2%}")
print(f"Avg Detour Ratio:        {np.mean([r['detour_ratio'] for r in results]):.4f}")
print(f"Explanation Consistency: {np.mean([r['consistency'] for r in results]):.2%}")

import json
os.makedirs("results", exist_ok=True)
with open("results/baseline_results.json", "w") as f:
    json.dump({
        "model": "baseline",
        "episodes": results,
        "averages": {
            "avg_pickups": round(np.mean([r['pickups'] for r in results]), 2),
            "avg_wait": round(np.mean([r['avg_wait'] for r in results]), 4),
            "response_rate": round(np.mean([r['response_rate'] for r in results]), 4),
            "utilization": round(np.mean([r['utilization'] for r in results]), 4),
            "detour_ratio": round(np.mean([r['detour_ratio'] for r in results]), 4),
            "consistency": round(np.mean([r['consistency'] for r in results]), 4),
        }
    }, f, indent=2)
print("Results saved to results/baseline_results.json")
checker.save()