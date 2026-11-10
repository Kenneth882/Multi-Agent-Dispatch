"""
matching/centralized_batch.py
------------------------------
Centralized batch matching via the Hungarian algorithm (SP2).

Builds a cost matrix of Manhattan distances between every idle driver and
every waiting request, then solves the global optimal assignment using
scipy.optimize.linear_sum_assignment.

When the number of drivers and requests differ, the cost matrix is padded
with a large sentinel value so the solver always receives a square-ish
matrix. Padded assignments (sentinel rows/cols) are discarded.

Time complexity: O(min(D, R)³) per step.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from matching.base import BaseMatcher

_SENTINEL = 10 ** 6  # cost for dummy padding cells


class CentralizedBatchMatcher(BaseMatcher):
    """
    Global optimal assignment minimising total Manhattan pickup distance.

    Uses scipy.optimize.linear_sum_assignment (Jonker-Volgenant algorithm).
    """

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
                cost[i, j] = context.manhattan_distance(driver.position, req.pickup)

        row_ind, col_ind = linear_sum_assignment(cost)

        assignments = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < _SENTINEL:
                assignments.append((drivers[i].id, requests[j].id))

        return assignments
