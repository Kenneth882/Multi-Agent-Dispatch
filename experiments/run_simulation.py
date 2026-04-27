import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.ride_share_env import RideShareEnv

env = RideShareEnv()
obs, _ = env.reset(seed=42)

total_pickups = 0
total_steps = 200

print(f"Starting simulation with {len(env.agents)} drivers on a 10x10 grid\n")

for step in range(total_steps):
    # Random actions for now (no learning yet)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    obs, rewards, terminations, truncations, _ = env.step(actions)

    step_pickups = sum(1 for r in rewards.values() if r > 0)
    total_pickups += step_pickups

    if step % 20 == 0:
        print(f"Step {step:>3} | Passengers waiting: {len(env.passengers):>2} | Pickups this step: {step_pickups} | Total pickups so far: {total_pickups}")

    if not env.agents:
        break

print(f"\n--- Episode Done ---")
print(f"Total pickups:     {total_pickups}")
print(f"Missed passengers: {len(env.passengers)}")
print(f"Pickup rate:       {total_pickups / (total_pickups + len(env.passengers)) * 100:.1f}%")
