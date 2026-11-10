"""
Base interface for Senior Project 2 matching algorithms.
All matchers receive a list of driver positions and passenger positions
and return a dict mapping driver_id -> passenger index (or None).
"""
from abc import ABC, abstractmethod


class BaseMatcher(ABC):
    """
    Abstract base for ride-sharing matching algorithms.

    Args:
        driver_positions: list of (row, col) tuples
        passenger_positions: list of (row, col) tuples

    Returns:
        dict mapping driver index -> passenger index, or None if unassigned
    """

    @abstractmethod
    def match(self, driver_positions: list, passenger_positions: list) -> dict:
        raise NotImplementedError
