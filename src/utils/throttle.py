"""Rolling token budget enforcement helpers."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Tuple


class TokenLimitExceeded(Exception):
    """Raised when a single request exceeds the configured token quota."""


class PerMinuteTokenThrottle:
    """
    Enforce a per-minute input token budget using a rolling window.

    Call ``wait_for_budget`` before issuing a request to block until sufficient
    budget remains, then call ``record`` with the actual input token count after
    the request completes.
    """

    def __init__(self, limit: int):
        if limit <= 0:
            raise ValueError("Token limit must be positive.")
        self.limit = limit
        self._window: Deque[Tuple[float, int]] = deque()
        self._current_tokens = 0

    def _prune(self, now: float | None = None) -> float:
        now = now if now is not None else time.monotonic()
        while self._window and now - self._window[0][0] >= 60:
            _, tokens = self._window.popleft()
            self._current_tokens -= tokens
        return now

    def wait_for_budget(self, tokens: int) -> float:
        """
        Block until the requested tokens fit under the rolling minute limit.

        Returns the total seconds spent waiting. Raises TokenLimitExceeded if the
        requested amount itself exceeds the configured per-minute allowance.
        """
        if tokens > self.limit:
            raise TokenLimitExceeded(
                f"Request requires {tokens:,} tokens but limit is {self.limit:,}."
            )

        waited = 0.0
        while True:
            now = self._prune()
            if self._current_tokens + tokens <= self.limit:
                return waited

            oldest_time, _ = self._window[0]
            sleep_seconds = max(0.0, 60 - (now - oldest_time))
            time.sleep(sleep_seconds)
            waited += sleep_seconds

    def record(self, tokens: int) -> None:
        """Record actual tokens consumed."""
        now = self._prune()
        self._window.append((now, tokens))
        self._current_tokens += tokens
