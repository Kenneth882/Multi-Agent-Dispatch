"""
Base interface for Senior Project 2 matching algorithms.

SP2 interface: match(eligible_drivers, waiting_requests, context)
  eligible_drivers  -- list[Driver]  idle drivers available for assignment
  waiting_requests  -- list[PassengerRequest]  unassigned, non-expired requests
  context           -- MatchingContext  read-only view of sim state + utilities
  returns           -- list[tuple[str, str]]  [(driver_id, request_id), ...]

Only idle drivers and waiting requests are passed in; the matcher must not
assign the same driver or request more than once per call.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.matching_env import Driver, PassengerRequest, MatchingContext


class BaseMatcher(ABC):
    @abstractmethod
    def match(
        self,
        eligible_drivers: list["Driver"],
        waiting_requests: list["PassengerRequest"],
        context: "MatchingContext",
    ) -> list[tuple[str, str]]:
        """Return a list of (driver_id, request_id) pairs."""
        raise NotImplementedError
