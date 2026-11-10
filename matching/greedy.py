"""
matching/greedy.py
-------------------
Passenger-first greedy matching algorithm for SP2.

Processing order
----------------
Requests are processed in arrival order (request_time ascending), with
request ID as a deterministic tie-break for requests that arrived at the
same simulation timestep. For each request, the nearest idle driver within
pickup_radius cells is assigned. If no driver is within radius, the request
is skipped this step (it remains WAITING and will be reconsidered next step).

Why passenger-first?
--------------------
Driver-first greedy (each driver picks its nearest request) can starve older
requests: a newly arrived request near many drivers may be picked up
immediately while an older request far from all drivers waits indefinitely.
Passenger-first ensures older requests get priority for available drivers.

Tie-breaking
------------
When multiple drivers are equidistant from a request, the driver with the
lexicographically smallest ID wins. This makes the algorithm fully
deterministic given the same inputs.

Configurable pickup radius
--------------------------
pickup_radius (default None = unlimited): maximum Manhattan distance a driver
may travel to pick up a request. Pairs beyond this radius are skipped.
Set to a positive integer to model realistic dispatch constraints.

Time complexity: O(R × D) per step where R = waiting requests, D = idle drivers.
"""
from matching.base import BaseMatcher


class GreedyMatcher(BaseMatcher):
    """
    Passenger-first greedy matcher.

    Args:
        pickup_radius: Maximum Manhattan distance for a driver-request pair.
                       None (default) means unlimited.
    """

    def __init__(self, pickup_radius=None):
        if pickup_radius is not None and pickup_radius < 0:
            raise ValueError(f"pickup_radius must be >= 0 or None, got {pickup_radius}")
        self.pickup_radius = pickup_radius

    def match(self, eligible_drivers, waiting_requests, context):
        if not eligible_drivers or not waiting_requests:
            return []

        # Sort requests: oldest first, then by ID for same-timestep determinism
        ordered_requests = sorted(
            waiting_requests,
            key=lambda r: (r.request_time, r.id),
        )

        available_drivers = list(eligible_drivers)
        assignments = []

        for req in ordered_requests:
            if not available_drivers:
                break

            # Find nearest eligible driver within pickup_radius
            best_driver = None
            best_dist = float("inf")
            for driver in available_drivers:
                d = context.manhattan_distance(driver.position, req.pickup)
                if self.pickup_radius is not None and d > self.pickup_radius:
                    continue
                if d < best_dist or (d == best_dist and driver.id < best_driver.id):
                    best_dist = d
                    best_driver = driver

            if best_driver is not None:
                assignments.append((best_driver.id, req.id))
                available_drivers.remove(best_driver)

        return assignments
