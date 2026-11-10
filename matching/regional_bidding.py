"""
matching/regional_bidding.py
-----------------------------
Regional Agent Bidding matching algorithm (SP2).

The grid is divided into an n_regions × n_regions partition. Each driver
bids only on requests within its own region. Within each region, the
nearest-driver-to-request assignment is used (greedy within region).

Cross-region fallback (cross_region=True, default):
  If a driver's home region has no waiting requests, it is allowed to bid
  on requests in any region. This prevents idle drivers from sitting out
  when nearby regions have unmet demand.

When cross_region=False, drivers strictly bid only within their home region.
This models a fully decentralized dispatch with hard regional boundaries.

Region assignment uses the driver's current position and the request's
pickup cell. Both are mapped to a region by:
    region_row = position_row * n_regions // grid_size
    region_col = position_col * n_regions // grid_size

Known limitations vs. true auction-based bidding
-------------------------------------------------
- This is greedy-within-region, not iterative auction bidding. A true
  auction would have drivers submit scalar bids (e.g. inverse distance),
  a central auctioneer resolve conflicts across regions in multiple rounds,
  and drivers update bids based on competition. That is not implemented here.
- The cross_region=True fallback makes the algorithm behave identically to
  passenger-first greedy when all regions are empty except one. The
  decentralized property only holds when cross_region=False.
- Region boundaries are static (fixed at construction). Dynamic region
  resizing based on demand density is not supported.

Time complexity: O(D × R) per step (same as greedy, but filtered by region).
"""
from collections import defaultdict

from matching.base import BaseMatcher


class RegionalBiddingMatcher(BaseMatcher):
    """
    Decentralized matching: drivers bid on requests within their region.

    Args:
        n_regions:    Number of region divisions per axis. Default 2 gives a
                      2×2 = 4-region partition of the 10×10 grid.
        cross_region: If True (default), drivers with no in-region requests
                      fall back to bidding globally. If False, they sit idle.
    """

    def __init__(self, n_regions: int = 2, cross_region: bool = True):
        if n_regions < 1:
            raise ValueError(f"n_regions must be >= 1, got {n_regions}")
        self.n_regions = n_regions
        self.cross_region = cross_region

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def match(self, eligible_drivers, waiting_requests, context):
        if not eligible_drivers or not waiting_requests:
            return []

        grid_size = context.config.grid_size

        # Group requests by region
        requests_by_region: dict[tuple, list] = defaultdict(list)
        for req in waiting_requests:
            requests_by_region[self._region(req.pickup, grid_size)].append(req)

        assignments = []
        assigned_requests: set[str] = set()

        # Sort drivers for determinism
        for driver in sorted(eligible_drivers, key=lambda d: d.id):
            home = self._region(driver.position, grid_size)
            candidates = [
                r for r in requests_by_region.get(home, [])
                if r.id not in assigned_requests
            ]

            if not candidates and self.cross_region:
                # Fall back: bid on all unassigned requests globally
                candidates = [r for r in waiting_requests if r.id not in assigned_requests]

            if not candidates:
                continue

            best = min(
                candidates,
                key=lambda r: (
                    context.manhattan_distance(driver.position, r.pickup),
                    r.id,
                ),
            )
            assignments.append((driver.id, best.id))
            assigned_requests.add(best.id)

        return assignments

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _region(self, position: tuple[int, int], grid_size: int) -> tuple[int, int]:
        """Map a grid cell to its (region_row, region_col)."""
        r, c = position
        return (
            r * self.n_regions // grid_size,
            c * self.n_regions // grid_size,
        )
