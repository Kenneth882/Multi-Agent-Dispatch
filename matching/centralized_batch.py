"""
matching/centralized_batch.py
------------------------------
Centralized batch matching via the Hungarian algorithm (SP2).

Objective
---------
1. Maximize the number of feasible matches (driver-request pairs within
   max_pickup_distance, if set).
2. Among all maximum-cardinality matchings, minimize total Manhattan pickup
   distance.

This is implemented by assigning a large sentinel cost (_SENTINEL) to
infeasible pairs. The Hungarian algorithm then naturally avoids them unless
no feasible alternative exists, which cannot happen because infeasible pairs
are only selected when the number of drivers exceeds the number of feasible
requests (the extra drivers are left unmatched via the sentinel filter).

max_pickup_distance
-------------------
Optional maximum Manhattan distance for a driver-request pair to be
considered feasible. Default None = unlimited (all pairs are feasible).
Pairs beyond this distance receive sentinel cost and are never selected.
If no feasible pair exists for a driver, that driver is left unmatched.

Unmatched requests
------------------
When there are more requests than drivers, some requests will not be matched
this step. They remain WAITING and are reconsidered next step. The solver
always assigns at most min(n_drivers, n_requests) pairs.

Time complexity: O(min(D, R)³) per step using scipy's Jonker-Volgenant solver.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from matching.base import BaseMatcher

_SENTINEL = 10 ** 6  # cost assigned to infeasible pairs


class CentralizedBatchMatcher(BaseMatcher):
    """
    Global optimal assignment minimising total Manhattan pickup distance,
    subject to an optional max_pickup_distance feasibility constraint.

    Objective: maximize feasible matches, then minimize total pickup distance.

    Args:
        max_pickup_distance: Maximum Manhattan distance for a feasible pair.
                             None (default) = unlimited.
    """

    def __init__(self, max_pickup_distance=None):
        if max_pickup_distance is not None and max_pickup_distance < 0:
            raise ValueError(
                f"max_pickup_distance must be >= 0 or None, got {max_pickup_distance}"
            )
        self.max_pickup_distance = max_pickup_distance

    def match(self, eligible_drivers, waiting_requests, context):
        if not eligible_drivers or not waiting_requests:
            return []

        drivers = sorted(eligible_drivers, key=lambda d: d.id)
        requests = sorted(waiting_requests, key=lambda r: r.id)

        n_d = len(drivers)
        n_r = len(requests)

        # Build cost matrix (n_d × n_r)
        cost = np.full((n_d, n_r), _SENTINEL, dtype=np.int64)
        for i, driver in enumerate(drivers):
            for j, req in enumerate(requests):
                d = context.manhattan_distance(driver.position, req.pickup)
                if self.max_pickup_distance is None or d <= self.max_pickup_distance:
                    cost[i, j] = d

        row_ind, col_ind = linear_sum_assignment(cost)

        assignments = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < _SENTINEL:
                assignments.append((drivers[i].id, requests[j].id))

        return assignments
