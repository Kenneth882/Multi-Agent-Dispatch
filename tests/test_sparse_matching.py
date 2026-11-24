import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import isspmatrix_csr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.matching_env import Driver, MatchingContext, PassengerRequest, SimConfig
from matching import CentralizedBatchMatcher, SparseMatcher
from matching import sparse


def inputs(driver_cells, request_cells):
    drivers = {f"d{i}": Driver(f"d{i}", cell) for i, cell in enumerate(driver_cells)}
    requests = {
        f"r{i}": PassengerRequest(f"r{i}", "test", cell, cell, 0)
        for i, cell in enumerate(request_cells)
    }
    return list(drivers.values()), list(requests.values()), MatchingContext(0, drivers, requests, SimConfig())


@pytest.mark.parametrize(("driver_cells", "request_cells"), [
    ([], []), ([(0, 0)], []), ([], [(0, 0)]), ([(0, 0)], [(9, 9)]),
])
def test_no_candidates(driver_cells, request_cells):
    drivers, requests, context = inputs(driver_cells, request_cells)
    assert SparseMatcher(1).match(drivers, requests, context) == []
    assert context.candidate_pairs_considered == 0


def test_csr_contains_only_feasible_edges_and_private_dummies(monkeypatch):
    drivers, requests, context = inputs([(0, 0), (0, 1), (9, 9), (5, 5)], [(0, 0), (9, 8), (8, 8)])
    solver = sparse.min_weight_full_bipartite_matching

    def checked_solver(graph):
        assert isspmatrix_csr(graph)
        assert graph.shape == (3, 7)
        assert graph.nnz == 3 + 3  # three real edges and one dummy per request
        assert graph[0, 0] == 1  # shifted zero-distance edge
        return solver(graph)

    monkeypatch.setattr(sparse, "min_weight_full_bipartite_matching", checked_solver)
    result = SparseMatcher(1).match(drivers, requests, context)
    assert set(result) == {("d0", "r0"), ("d2", "r1")}
    assert context.candidate_pairs_considered == 3


@pytest.mark.parametrize("cutoff", [0, 1, 3, 18])
@pytest.mark.parametrize(("n_drivers", "n_requests"), [(2, 5), (5, 2), (10, 10)])
def test_sparse_matches_dense_objective_with_competition(cutoff, n_drivers, n_requests):
    rng = np.random.default_rng(42)
    for _ in range(10):
        # Concentrate some vertices in the same cells to create competition and ties.
        driver_cells = list(map(tuple, rng.integers(0, 5, size=(n_drivers, 2))))
        request_cells = list(map(tuple, rng.integers(0, 5, size=(n_requests, 2))))
        drivers, requests, context = inputs(driver_cells, request_cells)
        sparse_result = SparseMatcher(cutoff).match(drivers, requests, context)
        dense_result = CentralizedBatchMatcher(cutoff).match(drivers, requests, context)

        def objective(pairs):
            distances = [
                context.manhattan_distance(driver_cells[int(d[1:])], request_cells[int(r[1:])])
                for d, r in pairs
            ]
            assert all(distance <= cutoff for distance in distances)
            assert len(pairs) == len({d for d, _ in pairs}) == len({r for _, r in pairs})
            return len(pairs), sum(distances)

        assert objective(sparse_result) == objective(dense_result)
