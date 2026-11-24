"""Sequential price auctions within regions, with an optional cross-region broker."""

from collections import Counter, defaultdict, deque
from math import isfinite

from matching.base import BaseMatcher


class RegionalBiddingMatcher(BaseMatcher):
    """Dispatch with local auctions or the original driver-first greedy prototype.

    Drivers bid on pickups in their own region. If cross_region is enabled,
    a broker auctions leftover cross-region pairs after local awards are fixed.
    The default partition has two divisions per axis. Prices are step-local.
    max_iterations limits bidder turns in each regional or broker auction.
    """

    def __init__(
        self, n_regions: int = 2, cross_region: bool = True, max_pickup_distance=None,
        *, mode: str = "auction", epsilon: float = 0.5, max_iterations: int = 10_000,
    ):
        if n_regions < 1:
            raise ValueError(f"n_regions must be >= 1, got {n_regions}")
        if max_pickup_distance is not None and max_pickup_distance < 0:
            raise ValueError("max_pickup_distance must be nonnegative")
        if mode not in {"auction", "greedy"}:
            raise ValueError("mode must be 'auction' or 'greedy'")
        if not isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.n_regions = n_regions
        self.cross_region = cross_region
        self.max_pickup_distance = max_pickup_distance
        self.mode = mode
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def match(self, eligible_drivers, waiting_requests, context):
        stats = Counter({
            "regional_bids": 0,
            "regional_auction_rounds": 0,
            "regional_displacements": 0,
            "regional_iteration_limit_hits": 0,
            "regional_cross_assignments": 0,
            "regional_communication_events": 0,
        })
        if not eligible_drivers or not waiting_requests:
            context.matcher_metrics.update(stats)
            return []
        drivers = sorted(eligible_drivers, key=lambda driver: driver.id)
        requests = sorted(waiting_requests, key=lambda request: request.id)
        grid_size = context.config.grid_size
        driver_regions = {d.id: self._region(d.position, grid_size) for d in drivers}
        request_regions = {r.id: self._region(r.pickup, grid_size) for r in requests}
        requests_by_region = defaultdict(list)
        for request in requests:
            requests_by_region[request_regions[request.id]].append(request)

        if self.mode == "greedy":
            assignments = self._greedy(drivers, requests, driver_regions, request_regions, requests_by_region, context)
        else:
            drivers_by_region = defaultdict(list)
            for driver in drivers:
                drivers_by_region[driver_regions[driver.id]].append(driver)
            assignments = []
            for region, local_drivers in sorted(drivers_by_region.items()):
                assignments.extend(self._auction(
                    local_drivers, requests_by_region[region], context, stats,
                ))

            if self.cross_region:
                assigned_drivers = {d for d, _ in assignments}
                assigned_requests = {r for _, r in assignments}
                remaining_drivers = [d for d in drivers if d.id not in assigned_drivers]
                remaining_requests = [r for r in requests if r.id not in assigned_requests]
                regions = {driver_regions[d.id] for d in remaining_drivers}
                regions.update(request_regions[r.id] for r in remaining_requests)
                if remaining_drivers and remaining_requests and len(regions) > 1:
                    stats["regional_communication_events"] += len(regions)
                    awards = self._auction(
                        remaining_drivers, remaining_requests, context, stats,
                        broker_regions=(driver_regions, request_regions),
                    )
                    stats["regional_communication_events"] += 2 * len(awards)
                    assignments.extend(awards)

        stats["regional_cross_assignments"] = sum(
            driver_regions[d] != request_regions[r] for d, r in assignments
        )
        context.matcher_metrics.update(stats)
        return sorted(assignments)

    def _auction(self, drivers, requests, context, stats, broker_regions=None):
        if not drivers or not requests:
            return []

        options = {}
        for driver in drivers:
            choices = []
            for request in requests:
                if broker_regions is not None:
                    driver_regions, request_regions = broker_regions
                    if driver_regions[driver.id] == request_regions[request.id]:
                        continue
                distance = context.pickup_distance(driver, request)
                if self.max_pickup_distance is None or distance <= self.max_pickup_distance:
                    choices.append((request.id, distance))
            if choices:
                options[driver.id] = choices

        if not options:
            return []
        k = min(len(options), len(requests))
        max_distance = max(distance for choices in options.values() for _, distance in choices)
        reward = 1 + k * (max_distance + self.epsilon)
        prices = defaultdict(float)
        owners = {}
        queue = deque(options)
        for _ in range(self.max_iterations):
            if not queue:
                break
            driver = queue.popleft()
            stats["regional_auction_rounds"] += 1
            best_request, best, second = None, 0.0, 0.0  # idle is a free outside option
            for request, distance in options[driver]:
                surplus = reward - distance - prices[request]
                if surplus > best:
                    best_request, best, second = request, surplus, best
                elif surplus > second:
                    second = surplus
            if best_request is None:
                continue
            prices[best_request] += best - second + self.epsilon
            if best_request in owners:
                queue.append(owners[best_request])
                stats["regional_displacements"] += 1
            owners[best_request] = driver
            stats["regional_bids"] += 1

        if queue:
            stats["regional_iteration_limit_hits"] += 1
        return [(driver, request) for request, driver in owners.items()]

    def _greedy(self, drivers, requests, driver_regions, request_regions, requests_by_region, context):
        assignments = []
        used = set()
        for driver in drivers:
            home = driver_regions[driver.id]
            local = requests_by_region[home]
            best = self._best_candidate(driver, (r for r in local if r.id not in used), context)
            if best is None and self.cross_region:
                candidates = (r for r in requests if r.id not in used and request_regions[r.id] != home)
                best = self._best_candidate(driver, candidates, context)
            if best is not None:
                assignments.append((driver.id, best.id))
                used.add(best.id)
        return assignments

    def _best_candidate(self, driver, candidates, context):
        scored = ((context.pickup_distance(driver, r), r) for r in candidates)
        distance, best = min(scored, key=lambda pair: (pair[0], pair[1].id), default=(0, None))
        if self.max_pickup_distance is not None and distance > self.max_pickup_distance:
            return None
        return best

    def _region(self, position, grid_size):
        row, col = position
        return row * self.n_regions // grid_size, col * self.n_regions // grid_size
