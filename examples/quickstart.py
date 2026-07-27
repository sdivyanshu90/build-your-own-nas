"""Run a complete search in a few lines, then use the winning architecture.

This is the shortest path from "nothing" to "a trained model I can call". It uses the
synthetic dataset, so it needs no network access, no GPU, and no configuration file, and
finishes in a few seconds.

Run it with::

    python examples/quickstart.py

Then look at ``artifacts/quickstart/`` for the database, the saved weights, and the report.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from nas_engine import SearchConfig, SearchEngine
from nas_engine.reporting import ReportGenerator


def build_config(output_dir: Path) -> SearchConfig:
    """Build a small, fast, CPU-only search configuration.

    In a real project this would live in a YAML file and be loaded with
    ``SearchConfig.from_yaml("configs/random_search.yaml")``. Building it inline here keeps
    the example self-contained.

    Args:
        output_dir: Where the database, artifacts, and reports go.

    Returns:
        A validated configuration.
    """
    return SearchConfig.from_mapping(
        {
            "project": {
                "name": "quickstart",
                "description": "Smallest end-to-end example.",
                "output_dir": str(output_dir),
            },
            "dataset": {
                "provider": "synthetic",
                "batch_size": 32,
                "options": {
                    "num_classes": 4,
                    "input_channels": 3,
                    "input_size": 16,
                    "train_samples": 256,
                    "validation_samples": 128,
                    "test_samples": 128,
                    "noise_scale": 0.5,
                    "seed": 42,
                },
            },
            "search_space": {"preset": "tiny_cnn"},
            "algorithm": {"name": "random_search"},
            "budget": {"max_evaluations": 6, "epochs": 2},
            "training": {"optimizer": {"learning_rate": 0.005}, "topk": 2},
            "evaluation": {"measure_latency": True, "latency_repeats": 3},
            "hardware": {"device": "cpu"},
            "reproducibility": {"seed": 42},
        }
    )


def main() -> int:
    """Run the search, print the result, and use the winning model.

    Returns:
        A process exit code.
    """
    with tempfile.TemporaryDirectory(prefix="nas-quickstart-") as temporary:
        output_dir = Path(temporary)
        config = build_config(output_dir)

        # 1. Run the search. Everything is persisted as it goes, so an interruption here
        #    can be resumed with `engine.resume(...)`.
        engine = SearchEngine(config)
        try:
            result = engine.run()
            print(result.summary())

            # 2. Inspect the trade-offs the search actually found.
            print("\nPareto front:")
            for candidate in result.pareto_front:
                accuracy = candidate.metrics["validation_accuracy"]
                parameters = int(candidate.metrics["trainable_parameters"])
                print(
                    f"  {candidate.architecture_hash[:12]}  "
                    f"accuracy={accuracy:.4f}  parameters={parameters:,}"
                )

            # 3. Reload the winner and run inference with it.
            spec, model = engine.load_best_model(result.search_id)
            model.eval()
            with torch.no_grad():
                batch = torch.randn(4, spec.input_channels, spec.input_size, spec.input_size)
                logits = model(batch)
            print(f"\nInference on a batch of 4: logits shape {tuple(logits.shape)}")
            print(f"Predicted classes: {logits.argmax(dim=1).tolist()}")

            # 4. Score the winner once on the held-out test split. Doing this more than
            #    once per search reintroduces the selection bias it exists to avoid.
            weights = engine.repository.get_candidate(
                result.best.candidate_id if result.best else ""
            ).artifacts.get("weights")
            if weights:
                test_metrics = engine.evaluator.evaluate_on_test(
                    spec, weights_path=engine.artifact_root / weights
                )
                print(f"Held-out test accuracy: {test_metrics['test_accuracy']:.4f}")

            # 5. Write a report you can read later.
            generator = ReportGenerator(
                engine.repository,
                objectives=config.objectives.build_objectives(),
                constraints=config.objectives.build_constraints(),
                output_dir=config.report_dir,
                artifact_root=config.artifact_dir,
            )
            artifacts = generator.generate(result.search_id)
            print(f"\nReport written to {artifacts.markdown}")
            print(f"(inside a temporary directory that is about to be removed: {output_dir})")
        finally:
            engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
