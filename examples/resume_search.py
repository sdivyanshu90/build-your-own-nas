"""Interrupt a search and resume it, showing that no work is lost or repeated.

A NAS run can take hours. Machines get rebooted, jobs get pre-empted, and someone always
presses Ctrl-C. This example makes the recovery path concrete:

1. Run a short search and stop it early.
2. Corrupt the state the way a crash would: leave a candidate stuck in ``RUNNING``.
3. Resume. The engine recovers the interrupted candidate, restores the strategy's exact
   generator position, and continues rather than replaying.

Run it with::

    python examples/resume_search.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nas_engine import SearchConfig, SearchEngine
from nas_engine.orchestration.lifecycle import CandidateState
from nas_engine.persistence.models import CandidateRecord


def build_config(output_dir: Path, *, max_evaluations: int) -> SearchConfig:
    """Build a configuration with a given evaluation budget.

    Both halves of the run must share an output directory: that is where the database and
    the checkpoints live.

    Args:
        output_dir: Shared output directory.
        max_evaluations: Evaluation budget for this segment.

    Returns:
        A validated configuration.
    """
    return SearchConfig.from_mapping(
        {
            "project": {"name": "resume-demo", "output_dir": str(output_dir)},
            "dataset": {
                "provider": "synthetic",
                "batch_size": 32,
                "options": {
                    "num_classes": 4,
                    "input_size": 16,
                    "train_samples": 192,
                    "validation_samples": 96,
                    "test_samples": 96,
                    "seed": 42,
                },
            },
            "search_space": {"preset": "tiny_cnn"},
            "algorithm": {
                "name": "regularized_evolution",
                "params": {"population_size": 4, "tournament_size": 2},
            },
            "budget": {"max_evaluations": max_evaluations, "epochs": 1},
            "evaluation": {"measure_latency": False},
            "hardware": {"device": "cpu"},
            "logging": {"level": "WARNING"},
            "retry": {"max_retries": 1},
        }
    )


def main() -> int:
    """Run, interrupt, and resume a search.

    Returns:
        A process exit code.
    """
    with tempfile.TemporaryDirectory(prefix="nas-resume-") as temporary:
        output_dir = Path(temporary)

        # --- Segment 1: a short run -------------------------------------------------
        first_engine = SearchEngine(build_config(output_dir, max_evaluations=4))
        try:
            first = first_engine.run()
            search_id = first.search_id
            print(f"Segment 1 finished: {first.engine_state.completed} evaluations")
            print(f"  search id: {search_id}")
            print(f"  best so far: {first.best.architecture_hash[:12] if first.best else 'none'}")
        finally:
            first_engine.close()

        # --- Simulate a crash --------------------------------------------------------
        # A process that dies mid-evaluation leaves its candidate in RUNNING with no
        # result. Nothing in the database says what happened; the state is the only clue.
        crash_engine = SearchEngine(build_config(output_dir, max_evaluations=4))
        try:
            victim = crash_engine.repository.list_candidates(search_id)[0]
            with crash_engine.repository.database.session() as session:
                record = session.get(CandidateRecord, victim.id)
                if record is not None:
                    record.status = CandidateState.RUNNING.value
            print(f"\nSimulated a crash: candidate {victim.architecture_hash[:12]} is RUNNING")
        finally:
            crash_engine.close()

        # --- Segment 2: resume with a larger budget ----------------------------------
        second_engine = SearchEngine(build_config(output_dir, max_evaluations=8))
        try:
            second = second_engine.resume(search_id)
            print(f"\nSegment 2 finished: {second.engine_state.completed} evaluations total")
            print(f"  resumed:    {second.resumed}")
            print(f"  duplicates: {second.engine_state.duplicates} (0 means nothing replayed)")
            print(f"  best:       {second.best.architecture_hash[:12] if second.best else 'none'}")
            for warning in second.warnings:
                print(f"  warning: {warning}")

            counts = second_engine.repository.count_candidates_by_status(search_id)
            print(f"\nFinal candidate states: {counts}")
            print(
                "\nThe recovered candidate was returned to the queue and re-evaluated; the "
                "strategy continued its random stream from where the checkpoint left it, so "
                "no architecture was proposed twice."
            )
        finally:
            second_engine.close()

        print("\nFrom the command line the same flow is:")
        print("  nas-engine search --config configs/evolution.yaml")
        print("  # ...interrupted...")
        print("  nas-engine resume --config configs/evolution.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
