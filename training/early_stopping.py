"""Epoch-level stopping on consecutive deterioration, not lack of a new best."""

import math


class ConsecutiveDeclineStopper:
    """Stop after three successive worsening epoch-end metrics."""

    def __init__(self, max_declines=3, lower_is_better=False):
        self.max_declines = max_declines
        self.lower_is_better = lower_is_better
        self.previous = None
        self.declines = 0

    def update(self, value):
        """Count one completed epoch; equality or recovery breaks the streak."""
        if not math.isfinite(value):
            raise ValueError('Early-stop metric must be finite.')
        worse = self.previous is not None and (
            value > self.previous if self.lower_is_better else value < self.previous
        )
        self.declines = self.declines + 1 if worse else 0
        self.previous = value
        return self.declines >= self.max_declines
