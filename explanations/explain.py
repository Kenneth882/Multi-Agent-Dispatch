import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REASON_CLOSEST = "closest"
REASON_AVAILABLE = "available"
REASON_IDLE = "idle_reduction"


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def explain_dispatch(driver_id, driver_pos, passenger_pos, other_driver_positions):
    distance = manhattan_distance(driver_pos, passenger_pos)

    other_distances = [manhattan_distance(d, passenger_pos) for d in other_driver_positions]

    if not other_distances or distance <= min(other_distances):
        reason = REASON_CLOSEST
        explanation = (
            f"Driver {driver_id} was assigned because they were the closest "
            f"driver to the passenger at a distance of {distance} cells."
        )
    else:
        reason = REASON_AVAILABLE
        explanation = (
            f"Driver {driver_id} was assigned due to availability "
            f"at a distance of {distance} cells from the passenger."
        )

    return {
        "driver_id": driver_id,
        "distance": distance,
        "reason": reason,
        "explanation": explanation,
        "driver_pos": tuple(driver_pos),
        "passenger_pos": tuple(passenger_pos),
    }


def explain_idle(driver_id, driver_pos, nearest_passenger_pos):
    if nearest_passenger_pos is None:
        return {
            "driver_id": driver_id,
            "reason": REASON_IDLE,
            "explanation": f"Driver {driver_id} remained idle because no passengers were available.",
            "driver_pos": tuple(driver_pos),
            "passenger_pos": None,
        }

    distance = manhattan_distance(driver_pos, nearest_passenger_pos)
    return {
        "driver_id": driver_id,
        "reason": REASON_IDLE,
        "explanation": (
            f"Driver {driver_id} remained idle with the nearest passenger "
            f"at distance {distance}."
        ),
        "driver_pos": tuple(driver_pos),
        "passenger_pos": tuple(nearest_passenger_pos),
    }


def generate_step_explanations(env, actions):
    explanations = {}
    passenger_list = list(env.passengers)

    for agent in env.agents:
        action = actions.get(agent, 0)
        driver_pos = list(env.positions[agent])
        other_positions = [
            list(env.positions[a]) for a in env.agents if a != agent
        ]

        if action == 0:
            nearest = (
                min(passenger_list, key=lambda p: manhattan_distance(driver_pos, list(p)))
                if passenger_list else None
            )
            nearest_pos = list(nearest) if nearest else None
            explanations[agent] = explain_idle(agent, driver_pos, nearest_pos)
        else:
            if passenger_list:
                nearest = min(
                    passenger_list,
                    key=lambda p: manhattan_distance(driver_pos, list(p))
                )
                explanations[agent] = explain_dispatch(
                    agent, driver_pos, list(nearest), other_positions
                )
            else:
                explanations[agent] = explain_idle(agent, driver_pos, None)

    return explanations