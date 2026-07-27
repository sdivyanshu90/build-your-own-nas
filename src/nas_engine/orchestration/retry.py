"""Retry policy and error classification.

Retrying is not free. Every retry costs a full evaluation budget, and a search with a
fixed evaluation cap spends that budget on a candidate it has already seen. So the policy
must be conservative and, above all, *correct about which failures are worth retrying*.

The decision has two inputs:

1. **The failure's own classification.** :func:`~nas_engine.evaluation.result.classify_failure`
   already decided whether the underlying exception is transient. An invalid architecture
   is never transient; an out-of-memory error usually is.
2. **The configured policy.** An operator may disable retries entirely (a strict
   reproducibility run), or disable them for a specific class (a machine where timeouts
   mean the configuration is wrong, not that the machine was busy).

Backoff
-------
Exponential backoff with a cap. The delay exists because the two genuinely retriable
failure classes — memory pressure and timeouts — are both caused by *contention*, and
retrying instantly reproduces the contention that caused the failure. Backoff is disabled
by default (``backoff_seconds = 0``) because in a single-machine sequential search the
contending work has already finished by the time the retry is scheduled.

There is no jitter. Jitter matters when many independent clients retry against a shared
service and would otherwise synchronise; here retries are serialised by one engine, so
jitter would add nondeterminism for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass

from nas_engine.evaluation.result import EvaluationFailure, FailureKind


@dataclass(frozen=True)
class RetryDecision:
    """The outcome of consulting the retry policy.

    Attributes:
        should_retry: Whether the candidate goes back to the queue.
        reason: Human-readable explanation, recorded on the candidate.
        delay_seconds: How long to wait before the retry.
        attempts_remaining: Retries left after this decision.
    """

    should_retry: bool
    reason: str
    delay_seconds: float = 0.0
    attempts_remaining: int = 0


@dataclass(frozen=True)
class RetryPolicy:
    """Decides whether a failed evaluation is retried.

    Attributes:
        max_retries: Retries allowed per candidate, beyond the first attempt.
        retry_on_timeout: Whether a wall-clock timeout is retriable.
        retry_on_resource_error: Whether a memory or resource failure is retriable.
        backoff_seconds: Delay before the first retry.
        backoff_multiplier: Multiplier applied per subsequent retry.
        max_backoff_seconds: Cap on the computed delay.

    Raises:
        ValueError: If any field is out of range.
    """

    max_retries: int = 1
    retry_on_timeout: bool = True
    retry_on_resource_error: bool = True
    backoff_seconds: float = 0.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 60.0

    def __post_init__(self) -> None:
        """Validate the policy.

        Raises:
            ValueError: If any field is out of range.
        """
        if self.max_retries < 0:
            msg = f"max_retries must be non-negative, received {self.max_retries}"
            raise ValueError(msg)
        if self.backoff_seconds < 0:
            msg = f"backoff_seconds must be non-negative, received {self.backoff_seconds}"
            raise ValueError(msg)
        if self.backoff_multiplier < 1.0:
            msg = (
                f"backoff_multiplier must be >= 1, received {self.backoff_multiplier}; a "
                "multiplier below 1 would shorten each successive delay"
            )
            raise ValueError(msg)
        if self.max_backoff_seconds < 0:
            msg = f"max_backoff_seconds must be non-negative, received {self.max_backoff_seconds}"
            raise ValueError(msg)

    def backoff_for(self, attempt: int) -> float:
        """Return the delay before the retry that follows ``attempt``.

        Args:
            attempt: Zero-based index of the attempt that just failed.

        Returns:
            The delay in seconds, capped at ``max_backoff_seconds``.
        """
        if self.backoff_seconds <= 0:
            return 0.0
        delay = self.backoff_seconds * (self.backoff_multiplier ** max(0, attempt))
        return min(delay, self.max_backoff_seconds)

    def decide(self, failure: EvaluationFailure, *, attempt: int) -> RetryDecision:
        """Decide whether a failed evaluation should be retried.

        Args:
            failure: The failure record.
            attempt: Zero-based index of the attempt that failed.

        Returns:
            A :class:`RetryDecision`.
        """
        remaining = self.max_retries - attempt

        if not failure.retriable:
            return RetryDecision(
                should_retry=False,
                reason=(
                    f"{failure.kind.value} failures are permanent: the same architecture "
                    "and seed would fail identically"
                ),
                attempts_remaining=max(0, remaining),
            )
        if failure.kind is FailureKind.TIMEOUT and not self.retry_on_timeout:
            return RetryDecision(
                should_retry=False,
                reason="retry.retry_on_timeout is disabled",
                attempts_remaining=max(0, remaining),
            )
        if failure.kind is FailureKind.RESOURCE and not self.retry_on_resource_error:
            return RetryDecision(
                should_retry=False,
                reason="retry.retry_on_resource_error is disabled",
                attempts_remaining=max(0, remaining),
            )
        if remaining <= 0:
            return RetryDecision(
                should_retry=False,
                reason=(
                    f"retry allowance of {self.max_retries} is exhausted after "
                    f"{attempt + 1} attempts"
                ),
                attempts_remaining=0,
            )

        return RetryDecision(
            should_retry=True,
            reason=(f"{failure.kind.value} failure is retriable; {remaining} attempt(s) remain"),
            delay_seconds=self.backoff_for(attempt),
            attempts_remaining=remaining,
        )


__all__ = ["RetryDecision", "RetryPolicy"]
