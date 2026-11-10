"""
matching/sparse.py
-------------------
Sparse matching with a distance cutoff (SP2).

Only driver-request pairs within `max_distance` Manhattan cells are
considered eligible. The eligible subproblem is solved optimally with the
Hungarian algorithm, so within-range assignments are globally optimal.

Pairs beyond the cutoff are assigned cost _SENTINEL and are never selected.
If no driver-request pair is within range, an empty list is returned.

This reduces computation in large, sparse scenarios where most drivers are
far from most requests, and models realistic dispatch constraints (e.g. a
driver in Inwood should not be dispatched to Battery Park).

Known limitations
-----------------
- The cost matrix is still built for all D×R pairs before filtering. For a
  10×10 grid with ≤20 drivers this is negligible, but a true sparse
  implementation would only build the submatrix of eligible pairs.
- The cutoff is a hard Euclidean (Manhattan) threshold, not a travel-time
  threshold. In a real network, travel time depends on road topology.

Time complexity: O(D×R) to build cost matrix + O(min(D,R)³) for Hungarian.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from matching.base import BaseMatcher

_SENTINEL = 10 ** 6


class SparseMatcher(BaseMatcher):
    """
    Batch matching restricted to driver-request pairs within max_distance cells.

    Args:
        max_distance: Maximum Manhattan distance for a driver-request pair to
                      be eligible. Default 3 cells (≈ 30% of the 10-cell grid).
    """

    def __init__(self, max_distance: int = 3):
        if max_distance < 0:
            raise ValueError(f"max_distance must be >= 0, got {max_distance}")
        self.max_distance = max_distance

    def match(self, eligible_drivers, waiting_requests, context):
        if not eligible_drivers or not waiting_requests:
            return []

        drivers = sorted(eligible_drivers, key=lambda d: d.id)
        requests = sorted(waiting_requests, key=lambda r: r.id)

        n_d = len(drivers)
        n_r = len(requests)

        cost = np.full((n_d, n_r), _SENTINEL, dtype=np.int64)
        any_eligible = False
        for i, driver in enumerate(drivers):
            for j, req in enumerate(requests):
                d = context.manhattan_distance(driver.position, req.pickup)
                if d <= self.max_distance:
                    cost[i, j] = d
                    any_eligible = True

        if not any_eligible:
            return []

        row_ind, col_ind = linear_sum_assignment(cost)

        assignments = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < _SENTINEL:
                assignments.append((drivers[i].id, requests[j].id))

        return assignments
