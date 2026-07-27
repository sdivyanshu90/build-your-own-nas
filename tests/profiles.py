"""Scaling for property-based test budgets.

The default example counts are tuned so the property suite finishes in a few seconds on a
laptop. The nightly workflow wants a much deeper search of the same properties, and a
developer chasing a rare counterexample wants the same knob.

Explicit ``@settings(max_examples=...)`` decorators override a Hypothesis *profile*, so a
profile alone cannot scale these tests. Routing the counts through :func:`scaled` gives one
environment variable that works everywhere::

    HYPOTHESIS_SCALE=10 pytest tests/property
"""

from __future__ import annotations

import os

#: Environment variable multiplying every property test's example budget.
SCALE_VARIABLE = "HYPOTHESIS_SCALE"


def scale_factor() -> float:
    """Return the configured example-budget multiplier.

    Returns:
        The multiplier; ``1.0`` when unset or unparseable.
    """
    raw = os.environ.get(SCALE_VARIABLE, "")
    if not raw:
        return 1.0
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    return value if value > 0 else 1.0


def scaled(base: int) -> int:
    """Scale a base example count by the configured factor.

    Args:
        base: Default number of examples.

    Returns:
        The scaled count, never below one.
    """
    return max(1, int(base * scale_factor()))
