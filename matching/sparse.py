"""Exact distance-limited matching over a CSR bipartite graph."""

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from matching.base import BaseMatcher


class SparseMatcher(BaseMatcher):
    """Maximize feasible matches, then minimize total Manhattan pickup distance."""

    def __init__(self, max_distance: int = 3):
        if max_distance < 0:
            raise ValueError(f"max_distance must be >= 0, got {max_distance}")
        self.max_distance = max_distance

    def match(self, eligible_drivers, waiting_requests, context):
        if not eligible_drivers or not waiting_requests:
            return []

        drivers = sorted(eligible_drivers, key=lambda driver: driver.id)
        requests = sorted(waiting_requests, key=lambda request: request.id)
        driver_cells = [driver.position for driver in drivers]
        request_cells = [request.pickup for request in requests]

        # Put the smaller partition on the row side to limit augmentations
        # and the number of private unmatched columns.
        drivers_are_rows = len(drivers) <= len(requests)
        row_cells, col_cells = (
            (driver_cells, request_cells) if drivers_are_rows else (request_cells, driver_cells)
        )
        grid_size = context.config.grid_size
        radius = int(min(self.max_distance, 2 * (grid_size - 1)))
        graph, real_edge_count = _build_graph(row_cells, col_cells, grid_size, radius)
        context.candidate_pairs_considered += real_edge_count
        if graph is None:
            return []

        row_indices, col_indices = min_weight_full_bipartite_matching(graph)
        matches = [
            (row, col) if drivers_are_rows else (col, row)
            for row, col in zip(row_indices, col_indices)
            if col < len(col_cells)
        ]
        return [(drivers[d].id, requests[r].id) for d, r in sorted(matches)]


def _build_graph(row_cells, col_cells, grid_size, radius):
    """Build CSR directly from grid buckets and one private dummy per row."""
    columns_by_cell = defaultdict(list)
    for column, cell in enumerate(col_cells):
        columns_by_cell[cell].append(column)

    # Rows sharing a cell reuse the same eligible columns and shifted costs.
    neighbors = {}
    for row, col in dict.fromkeys(row_cells):
        edges = []
        for y in range(max(0, row - radius), min(grid_size, row + radius + 1)):
            remaining = radius - abs(row - y)
            for x in range(max(0, col - remaining), min(grid_size, col + remaining + 1)):
                weight = 1 + abs(row - y) + abs(col - x)
                edges.extend((index, weight) for index in columns_by_cell.get((y, x), ()))
        edges.sort()
        neighbors[row, col] = np.asarray(edges, dtype=np.intp).reshape(-1, 2)

    n_rows, n_cols = len(row_cells), len(col_cells)
    real_edge_count = sum(len(neighbors[cell]) for cell in row_cells)
    if not real_edge_count:
        return None, 0

    # An extra dummy costs more than the largest possible total real cost.
    # A shift of +1 keeps co-located pairs explicit and leaves equal-cardinality
    # distance comparisons unchanged.
    dummy_cost = n_rows * (radius + 1) + 1
    edge_count = real_edge_count + n_rows
    index_dtype = np.int32 if max(edge_count, n_cols + n_rows) <= np.iinfo(np.int32).max else np.int64
    data = np.empty(edge_count, dtype=np.float64)
    indices = np.empty(edge_count, dtype=index_dtype)
    indptr = np.empty(n_rows + 1, dtype=index_dtype)
    indptr[0] = 0
    for row, cell in enumerate(row_cells):
        edges = neighbors[cell]
        start = indptr[row]
        end = start + len(edges)
        indices[start:end] = edges[:, 0]
        data[start:end] = edges[:, 1]
        indices[end] = n_cols + row
        data[end] = dummy_cost
        indptr[row + 1] = end + 1
    graph = csr_matrix((data, indices, indptr), shape=(n_rows, n_cols + n_rows))
    return graph, real_edge_count
