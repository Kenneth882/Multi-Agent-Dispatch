import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
import json
import numpy as np


def state_signature(driver_pos, passenger_pos, other_driver_positions):
    sorted_others = tuple(sorted([tuple(p) for p in other_driver_positions]))
    return (tuple(driver_pos), tuple(passenger_pos) if passenger_pos else None, sorted_others)


class ConsistencyChecker:
    def __init__(self):
        self.state_to_explanations = defaultdict(list)
        self.state_to_reasons = defaultdict(list)
        self.total_logged = 0

    def log(self, explanation_record):
        driver_pos = explanation_record["driver_pos"]
        passenger_pos = explanation_record["passenger_pos"]
        reason = explanation_record["reason"]
        other_positions = []

        sig = state_signature(driver_pos, passenger_pos, other_positions)
        self.state_to_explanations[sig].append(explanation_record["explanation"])
        self.state_to_reasons[sig].append(reason)
        self.total_logged += 1

    def log_step(self, explanations_dict):
        for agent, record in explanations_dict.items():
            self.log(record)

    def consistency_score(self):
        if not self.state_to_reasons:
            return 0.0

        consistent_count = 0
        total_comparable = 0

        for sig, reasons in self.state_to_reasons.items():
            if len(reasons) < 2:
                continue
            total_comparable += len(reasons)
            most_common = max(set(reasons), key=reasons.count)
            consistent_count += reasons.count(most_common)

        if total_comparable == 0:
            return 1.0

        return consistent_count / total_comparable

    def reason_distribution(self):
        distribution = defaultdict(int)
        for reasons in self.state_to_reasons.values():
            for r in reasons:
                distribution[r] += 1
        return dict(distribution)

    def get_inconsistent_states(self):
        inconsistent = []
        for sig, reasons in self.state_to_reasons.items():
            if len(set(reasons)) > 1:
                inconsistent.append({
                    "state": sig,
                    "reasons_seen": list(set(reasons)),
                    "occurrences": len(reasons),
                })
        return inconsistent

    def summary(self):
        score = self.consistency_score()
        distribution = self.reason_distribution()
        inconsistent = self.get_inconsistent_states()

        return {
            "consistency_score": round(score, 4),
            "total_explanations_logged": self.total_logged,
            "total_unique_states": len(self.state_to_reasons),
            "reason_distribution": distribution,
            "inconsistent_state_count": len(inconsistent),
        }

    def print_summary(self):
        s = self.summary()
        print("\n--- Explanation Consistency Report ---")
        print(f"Consistency Score:        {s['consistency_score']:.2%}")
        print(f"Total Explanations:       {s['total_explanations_logged']}")
        print(f"Unique States Seen:       {s['total_unique_states']}")
        print(f"Inconsistent States:      {s['inconsistent_state_count']}")
        print(f"Reason Distribution:      {s['reason_distribution']}")

    def save(self, path="results/consistency_report.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        print(f"Consistency report saved to {path}")


if __name__ == "__main__":
    from env.ride_share_env import RideShareEnv
    from explanations.explain import generate_step_explanations

    checker = ConsistencyChecker()
    env = RideShareEnv()
    obs, _ = env.reset(seed=42)

    for step in range(200):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        explanations = generate_step_explanations(env, actions)
        checker.log_step(explanations)
        obs, rewards, terminations, truncations, _ = env.step(actions)
        if not env.agents:
            break

    checker.print_summary()
    checker.save()