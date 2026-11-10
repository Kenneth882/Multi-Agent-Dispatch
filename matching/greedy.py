"""
matching/greedy.py
-------------------
Greedy matching algorithm for SP2.

Each idle driver is independently assigned to the nearest unassigned waiting
request (minimum Manhattan distance). Drivers are processed in ID order for
determinism. Ties in distance are broken by request ID (lexicographic).

Time complexity: O(D × R) per step where D = idle drivers, R = waiting requests.
"""
from matching.base import BaseMatcher


class GreedyMatcher(BaseMatcher):
    """
    Each driver picks the nearest unassigned waiting request.

    Processing order: drivers sorted by ID (deterministic).
    Tie-break: smallest request ID wins.
    """

    def match(self, eligible_drivers, waiting_requests, context):
        if not eligible_drivers or not waiting_requests:
            return []

        assignments = []
        available = list(waiting_requests)  # shallow copy — we remove as we assign

        for driver in sorted(eligible_drivers, key=lambda d: d.id):
            if not available:
                break
            best = min(
                available,
                key=lambda r: (
                    context.manhattan_distance(driver.position, r.pickup),
                    r.id,
                ),
            )
            assignments.append((driver.id, best.id))
            available.remove(best)

        return assignments
