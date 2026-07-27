"""Unit tests for datasets, loaders, and the training stack.

Covers: synthetic data determinism and learnability, split disjointness, fidelity views,
loader construction, metric aggregation, optimiser and scheduler construction, early
stopping, training checkpoints, and the training loop's guard rails.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from nas_engine.architectures.hashing import architecture_hash
from nas_engine.architectures.spec import ArchitectureSpec
from nas_engine.datasets.base import DatasetBundle
from nas_engine.datasets.loaders import (
    FidelityView,
    LoaderSettings,
    ResizedDataset,
    build_dataloaders,
    deterministic_subset,
)
from nas_engine.datasets.registry import (
    available_providers,
    build_dataset,
    get_provider,
    register_provider,
)
from nas_engine.datasets.synthetic import SyntheticDatasetProvider, SyntheticImageDataset
from nas_engine.exceptions import (
    CheckpointError,
    CheckpointVersionError,
    ConfigurationError,
    DatasetError,
    EvaluationTimeoutError,
    NonFiniteLossError,
)
from nas_engine.models.builder import build_model
from nas_engine.training.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    TrainingCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from nas_engine.training.early_stopping import EarlyStopping, MonitorMode
from nas_engine.training.metrics import (
    EpochMetrics,
    MetricAggregator,
    accuracy,
    topk_accuracy,
)
from nas_engine.training.optimizers import (
    OptimizerSettings,
    OptimizerType,
    build_optimizer,
    split_parameter_groups,
)
from nas_engine.training.schedulers import (
    SchedulerSettings,
    SchedulerType,
    build_scheduler,
)
from nas_engine.training.trainer import Trainer, TrainingSettings, evaluation_mode

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------- data --
class TestSyntheticDataset:
    def test_is_deterministic_for_a_given_seed(self) -> None:
        first = SyntheticImageDataset(num_samples=8, num_classes=3, channels=3, size=8, seed=1)
        second = SyntheticImageDataset(num_samples=8, num_classes=3, channels=3, size=8, seed=1)
        assert torch.equal(first[0][0], second[0][0])
        assert first[0][1] == second[0][1]

    def test_different_seeds_produce_different_data(self) -> None:
        first = SyntheticImageDataset(num_samples=8, num_classes=3, channels=3, size=8, seed=1)
        second = SyntheticImageDataset(num_samples=8, num_classes=3, channels=3, size=8, seed=2)
        assert not torch.equal(first[0][0], second[0][0])

    def test_labels_are_balanced(self) -> None:
        dataset = SyntheticImageDataset(num_samples=12, num_classes=4, channels=1, size=8, seed=3)
        counts = torch.bincount(dataset.targets, minlength=4)
        assert counts.tolist() == [3, 3, 3, 3]

    def test_shapes_match_the_request(self) -> None:
        dataset = SyntheticImageDataset(num_samples=4, num_classes=2, channels=3, size=12, seed=4)
        image, label = dataset[0]
        assert image.shape == (3, 12, 12)
        assert 0 <= label < 2
        assert len(dataset) == 4

    def test_rejects_degenerate_configurations(self) -> None:
        with pytest.raises(DatasetError, match="requires num_samples"):
            SyntheticImageDataset(num_samples=0, num_classes=2, channels=1, size=8, seed=1)
        with pytest.raises(DatasetError, match="requires num_samples"):
            SyntheticImageDataset(num_samples=4, num_classes=1, channels=1, size=8, seed=1)
        with pytest.raises(DatasetError, match="noise_scale"):
            SyntheticImageDataset(
                num_samples=4, num_classes=2, channels=1, size=8, seed=1, noise_scale=-1
            )

    def test_task_is_learnable_by_a_linear_probe(self) -> None:
        # If a linear model cannot beat chance on this data, the dataset is noise and every
        # downstream training test would be measuring nothing.
        dataset = SyntheticImageDataset(
            num_samples=256, num_classes=4, channels=3, size=8, seed=5, noise_scale=0.3
        )
        features = dataset._data.reshape(256, -1)
        labels = dataset.targets
        probe = nn.Linear(features.shape[1], 4)
        optimiser = torch.optim.Adam(probe.parameters(), lr=0.05)
        for _ in range(120):
            optimiser.zero_grad()
            loss = nn.functional.cross_entropy(probe(features), labels)
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
        predicted = probe(features).argmax(dim=1)
        assert float((predicted == labels).float().mean()) > 0.6

    def test_splits_share_templates_but_not_examples(self) -> None:
        bundle = SyntheticDatasetProvider(
            num_classes=3,
            input_size=8,
            train_samples=16,
            validation_samples=16,
            test_samples=16,
            seed=6,
        ).build()
        train_first = bundle.train[0][0]
        validation_first = bundle.validation[0][0]
        assert not torch.equal(train_first, validation_first)

    def test_bundle_reports_its_sizes(self, synthetic_bundle: DatasetBundle) -> None:
        sizes = synthetic_bundle.split_sizes()
        assert sizes == {"train": 96, "validation": 48, "test": 48}
        assert "synthetic" in synthetic_bundle.summary()

    def test_bundle_validates_its_metadata(self, synthetic_bundle: DatasetBundle) -> None:
        with pytest.raises(DatasetError, match="at least 2 classes"):
            DatasetBundle(
                name="bad",
                train=synthetic_bundle.train,
                validation=synthetic_bundle.validation,
                test=synthetic_bundle.test,
                num_classes=1,
                input_channels=3,
                input_size=8,
            )
        with pytest.raises(DatasetError, match="input_size"):
            DatasetBundle(
                name="bad",
                train=synthetic_bundle.train,
                validation=synthetic_bundle.validation,
                test=synthetic_bundle.test,
                num_classes=2,
                input_channels=3,
                input_size=2,
            )


class TestRegistry:
    def test_default_providers_are_registered(self) -> None:
        assert {"synthetic", "cifar10"} <= set(available_providers())

    def test_unknown_provider_is_reported(self) -> None:
        with pytest.raises(DatasetError, match="unknown dataset provider"):
            get_provider("nope")

    def test_bad_options_are_reported(self) -> None:
        with pytest.raises(DatasetError, match="rejected the supplied options"):
            get_provider("synthetic", not_a_field=1)

    def test_build_dataset_returns_a_bundle(self) -> None:
        bundle = build_dataset(
            "synthetic",
            num_classes=2,
            input_size=8,
            train_samples=8,
            validation_samples=8,
            test_samples=8,
        )
        assert bundle.num_classes == 2

    def test_custom_providers_can_be_registered(self) -> None:
        class Custom:
            name = "custom-test"

            def build(self) -> DatasetBundle:  # pragma: no cover - never called
                raise NotImplementedError

        register_provider("custom-test", Custom, overwrite=True)
        assert "custom-test" in available_providers()
        with pytest.raises(DatasetError, match="already registered"):
            register_provider("custom-test", Custom)


class TestLoaders:
    def test_builds_train_and_validation_loaders(self, synthetic_bundle: DatasetBundle) -> None:
        loaders = build_dataloaders(synthetic_bundle, LoaderSettings(batch_size=16), seed=1)
        assert loaders.test is None
        assert loaders.train_examples == 96
        assert loaders.input_size == 16

    def test_test_loader_is_opt_in(self, synthetic_bundle: DatasetBundle) -> None:
        loaders = build_dataloaders(
            synthetic_bundle, LoaderSettings(batch_size=16), seed=1, include_test=True
        )
        assert loaders.test is not None

    def test_shuffling_is_reproducible(self, synthetic_bundle: DatasetBundle) -> None:
        def first_batch_labels() -> list[int]:
            loaders = build_dataloaders(synthetic_bundle, LoaderSettings(batch_size=8), seed=42)
            labels: list[int] = next(iter(loaders.train))[1].tolist()
            return labels

        assert first_batch_labels() == first_batch_labels()

    def test_data_fraction_reduces_the_training_split(
        self, synthetic_bundle: DatasetBundle
    ) -> None:
        loaders = build_dataloaders(
            synthetic_bundle,
            LoaderSettings(batch_size=8),
            seed=1,
            fidelity=FidelityView(train_fraction=0.5),
        )
        assert loaders.train_examples == 48

    def test_resolution_fidelity_resizes_every_split(self, synthetic_bundle: DatasetBundle) -> None:
        loaders = build_dataloaders(
            synthetic_bundle,
            LoaderSettings(batch_size=8),
            seed=1,
            fidelity=FidelityView(resolution=8),
        )
        images, _ = next(iter(loaders.train))
        assert images.shape[-2:] == (8, 8)
        assert loaders.input_size == 8

    def test_subsets_are_deterministic(self, synthetic_bundle: DatasetBundle) -> None:
        first = deterministic_subset(synthetic_bundle.train, 0.25, seed=9)
        second = deterministic_subset(synthetic_bundle.train, 0.25, seed=9)
        assert torch.equal(first[0][0], second[0][0])

    def test_full_fraction_returns_the_original(self, synthetic_bundle: DatasetBundle) -> None:
        assert deterministic_subset(synthetic_bundle.train, 1.0, seed=1) is (synthetic_bundle.train)

    def test_invalid_fraction_is_rejected(self, synthetic_bundle: DatasetBundle) -> None:
        with pytest.raises(DatasetError, match="fraction must lie"):
            deterministic_subset(synthetic_bundle.train, 0.0, seed=1)

    def test_resize_target_is_validated(self, synthetic_bundle: DatasetBundle) -> None:
        with pytest.raises(DatasetError, match="resize target must be positive"):
            ResizedDataset(synthetic_bundle.train, 0)

    def test_loader_settings_are_validated(self) -> None:
        with pytest.raises(DatasetError, match="batch_size"):
            LoaderSettings(batch_size=0)
        with pytest.raises(DatasetError, match="num_workers"):
            LoaderSettings(num_workers=-1)

    def test_fidelity_view_is_validated(self) -> None:
        with pytest.raises(DatasetError, match="train_fraction"):
            FidelityView(train_fraction=0.0)
        with pytest.raises(DatasetError, match="resolution"):
            FidelityView(resolution=2)

    def test_fidelity_view_describes_itself(self) -> None:
        assert FidelityView().is_full
        assert FidelityView().describe() == "full fidelity"
        assert "50%" in FidelityView(train_fraction=0.5).describe()


# ------------------------------------------------------------------------ training --
class TestMetrics:
    def test_accuracy_counts_correct_predictions(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
        targets = torch.tensor([0, 1, 1])
        assert accuracy(logits, targets) == pytest.approx(2 / 3)

    def test_accuracy_validates_shapes(self) -> None:
        with pytest.raises(ValueError, match="logits must be 2-D"):
            accuracy(torch.zeros(3), torch.zeros(3, dtype=torch.long))
        with pytest.raises(ValueError, match="targets must be 1-D"):
            accuracy(torch.zeros(3, 2), torch.zeros(2, dtype=torch.long))

    def test_topk_is_clamped_to_the_class_count(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        targets = torch.tensor([0])
        assert topk_accuracy(logits, targets, 10) == 1.0

    def test_topk_is_at_least_top1(self) -> None:
        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))
        assert topk_accuracy(logits, targets, 3) >= accuracy(logits, targets)

    def test_topk_rejects_non_positive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            topk_accuracy(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), 0)

    def test_empty_batches_score_zero(self) -> None:
        assert accuracy(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long)) == 0.0
        assert topk_accuracy(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long), 2) == 0.0

    def test_aggregation_is_example_weighted(self) -> None:
        aggregator = MetricAggregator()
        aggregator.update({"loss": 1.0}, batch_size=9)
        aggregator.update({"loss": 0.0}, batch_size=1)
        assert aggregator.compute()["loss"] == pytest.approx(0.9)

    def test_aggregation_rejects_empty_batches(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            MetricAggregator().update({"loss": 1.0}, batch_size=0)

    def test_empty_aggregator_computes_nothing(self) -> None:
        assert MetricAggregator().compute() == {}

    def test_reset_clears_state(self) -> None:
        aggregator = MetricAggregator()
        aggregator.update({"loss": 1.0}, batch_size=4)
        aggregator.reset()
        assert aggregator.compute() == {}

    def test_epoch_metrics_serialise(self) -> None:
        payload = EpochMetrics(
            epoch=1,
            phase="train",
            loss=0.5,
            accuracy=0.8,
            topk_accuracy=0.9,
            examples=10,
            duration_seconds=1.0,
            learning_rate=0.01,
        ).to_dict()
        assert payload["phase"] == "train"
        assert payload["learning_rate"] == 0.01


class TestOptimizers:
    def test_builds_sgd(self) -> None:
        model = nn.Linear(4, 2)
        optimiser = build_optimizer(model, OptimizerSettings(name=OptimizerType.SGD))
        assert isinstance(optimiser, torch.optim.SGD)

    def test_builds_adamw(self) -> None:
        model = nn.Linear(4, 2)
        optimiser = build_optimizer(model, OptimizerSettings(name=OptimizerType.ADAMW))
        assert isinstance(optimiser, torch.optim.AdamW)

    def test_normalisation_and_bias_are_exempt_from_decay(self) -> None:
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
        groups = split_parameter_groups(model, 0.01, decay_normalization=False)
        assert len(groups) == 2
        assert groups[1]["weight_decay"] == 0.0
        # BatchNorm weight and bias, plus the convolution bias if present.
        assert len(groups[1]["params"]) >= 2  # type: ignore[arg-type]

    def test_decay_can_be_applied_uniformly(self) -> None:
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
        groups = split_parameter_groups(model, 0.01, decay_normalization=True)
        assert len(groups) == 1

    def test_zero_decay_uses_a_single_group(self) -> None:
        model = nn.Linear(4, 2)
        assert len(split_parameter_groups(model, 0.0, decay_normalization=False)) == 1

    def test_model_without_parameters_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="no trainable parameters"):
            build_optimizer(nn.Identity(), OptimizerSettings())

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"learning_rate": 0.0},
            {"weight_decay": -1.0},
            {"momentum": 1.0},
            {"beta1": 1.0},
            {"eps": 0.0},
        ],
    )
    def test_hyperparameters_are_validated(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ConfigurationError):
            OptimizerSettings(**kwargs)  # type: ignore[arg-type]


class TestSchedulers:
    def test_cosine_anneals_towards_the_floor(self) -> None:
        optimiser = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1.0)
        scheduler = build_scheduler(
            optimiser,
            SchedulerSettings(name=SchedulerType.COSINE, min_lr_factor=0.0),
            total_steps=10,
            steps_per_epoch=10,
        )
        rates = []
        for _ in range(10):
            rates.append(optimiser.param_groups[0]["lr"])
            optimiser.step()
            scheduler.step()
        assert rates[0] == pytest.approx(1.0)
        assert rates[-1] < 0.1
        assert rates == sorted(rates, reverse=True)

    def test_constant_schedule_holds_the_rate(self) -> None:
        optimiser = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.5)
        scheduler = build_scheduler(
            optimiser,
            SchedulerSettings(name=SchedulerType.CONSTANT),
            total_steps=5,
            steps_per_epoch=5,
        )
        for _ in range(5):
            assert optimiser.param_groups[0]["lr"] == pytest.approx(0.5)
            optimiser.step()
            scheduler.step()

    def test_step_schedule_decays_on_the_interval(self) -> None:
        optimiser = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1.0)
        scheduler = build_scheduler(
            optimiser,
            SchedulerSettings(name=SchedulerType.STEP, step_size_epochs=1, gamma=0.5),
            total_steps=6,
            steps_per_epoch=2,
        )
        rates = []
        for _ in range(6):
            rates.append(optimiser.param_groups[0]["lr"])
            optimiser.step()
            scheduler.step()
        assert rates[0] == pytest.approx(1.0)
        assert rates[2] == pytest.approx(0.5)
        assert rates[4] == pytest.approx(0.25)

    def test_warmup_ramps_from_a_non_zero_value(self) -> None:
        optimiser = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1.0)
        scheduler = build_scheduler(
            optimiser,
            SchedulerSettings(name=SchedulerType.CONSTANT, warmup_steps=4),
            total_steps=8,
            steps_per_epoch=8,
        )
        first = optimiser.param_groups[0]["lr"]
        optimiser.step()
        scheduler.step()
        second = optimiser.param_groups[0]["lr"]
        assert 0 < first < second

    def test_step_counts_are_validated(self) -> None:
        optimiser = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1.0)
        with pytest.raises(ConfigurationError, match="total_steps"):
            build_scheduler(optimiser, SchedulerSettings(), total_steps=0, steps_per_epoch=1)
        with pytest.raises(ConfigurationError, match="steps_per_epoch"):
            build_scheduler(optimiser, SchedulerSettings(), total_steps=1, steps_per_epoch=0)

    @pytest.mark.parametrize(
        "kwargs",
        [{"warmup_steps": -1}, {"min_lr_factor": 2.0}, {"step_size_epochs": 0}, {"gamma": 0.0}],
    )
    def test_hyperparameters_are_validated(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ConfigurationError):
            SchedulerSettings(**kwargs)  # type: ignore[arg-type]


class TestEarlyStopping:
    def test_disabled_by_default(self) -> None:
        stopper = EarlyStopping()
        assert not stopper.enabled
        stopper.update(0.1, epoch=0)
        stopper.update(0.0, epoch=1)
        assert not stopper.should_stop

    def test_stops_after_patience_is_exhausted(self) -> None:
        stopper = EarlyStopping(patience=2, mode=MonitorMode.MAX)
        stopper.update(0.5, epoch=0)
        stopper.update(0.4, epoch=1)
        assert not stopper.should_stop
        stopper.update(0.3, epoch=2)
        assert stopper.should_stop

    def test_improvement_resets_the_counter(self) -> None:
        stopper = EarlyStopping(patience=1, mode=MonitorMode.MAX)
        stopper.update(0.5, epoch=0)
        stopper.update(0.4, epoch=1)
        stopper.update(0.6, epoch=2)
        assert not stopper.should_stop
        assert stopper.best_epoch == 2

    def test_min_delta_ignores_noise(self) -> None:
        stopper = EarlyStopping(patience=1, min_delta=0.05, mode=MonitorMode.MAX)
        stopper.update(0.50, epoch=0)
        assert not stopper.update(0.52, epoch=1)

    def test_minimisation_mode(self) -> None:
        stopper = EarlyStopping(patience=1, mode=MonitorMode.MIN)
        assert stopper.update(1.0, epoch=0)
        assert stopper.update(0.5, epoch=1)
        assert not stopper.update(0.7, epoch=2)

    def test_state_round_trips(self) -> None:
        stopper = EarlyStopping(patience=2, mode=MonitorMode.MAX)
        stopper.update(0.7, epoch=3)
        stopper.update(0.6, epoch=4)
        restored = EarlyStopping(patience=2, mode=MonitorMode.MAX)
        restored.load_state_dict(stopper.state_dict())
        assert restored.best_value == 0.7
        assert restored.best_epoch == 3
        assert restored.epochs_without_improvement == 1

    def test_configuration_is_validated(self) -> None:
        with pytest.raises(ConfigurationError, match="patience"):
            EarlyStopping(patience=-1)
        with pytest.raises(ConfigurationError, match="min_delta"):
            EarlyStopping(min_delta=-1.0)


class TestTrainingCheckpoints:
    def test_round_trips_through_a_file(self, tmp_path: Path) -> None:
        checkpoint = TrainingCheckpoint(
            architecture_hash="abc",
            epoch=2,
            global_step=20,
            model_state={"w": torch.ones(2)},
            history=[{"epoch": 0}],
        )
        path = save_checkpoint(tmp_path / "ck.pt", checkpoint)
        restored = load_checkpoint(path, expected_hash="abc")
        assert restored.epoch == 2
        assert restored.global_step == 20
        assert torch.equal(restored.model_state["w"], torch.ones(2))

    def test_rejects_a_mismatched_architecture(self, tmp_path: Path) -> None:
        path = save_checkpoint(
            tmp_path / "ck.pt",
            TrainingCheckpoint(architecture_hash="abc", epoch=0, global_step=0, model_state={}),
        )
        with pytest.raises(CheckpointError, match="belongs to architecture"):
            load_checkpoint(path, expected_hash="different")

    def test_reports_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="not found"):
            load_checkpoint(tmp_path / "absent.pt")

    def test_reports_a_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.pt"
        path.write_bytes(b"not a torch checkpoint")
        with pytest.raises(CheckpointError, match="could not be read"):
            load_checkpoint(path)

    def test_rejects_a_future_format(self, tmp_path: Path) -> None:
        path = tmp_path / "future.pt"
        torch.save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION + 1,
                "architecture_hash": "abc",
                "epoch": 0,
                "model_state": {},
            },
            path,
        )
        with pytest.raises(CheckpointVersionError, match="newer than the supported"):
            load_checkpoint(path)

    def test_rejects_a_missing_version(self, tmp_path: Path) -> None:
        path = tmp_path / "unversioned.pt"
        torch.save({"architecture_hash": "abc", "epoch": 0, "model_state": {}}, path)
        with pytest.raises(CheckpointError, match="format_version"):
            load_checkpoint(path)

    def test_rejects_missing_required_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "incomplete.pt"
        torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "epoch": 0}, path)
        with pytest.raises(CheckpointError, match="missing required fields"):
            load_checkpoint(path)

    def test_rejects_a_non_dictionary_payload(self, tmp_path: Path) -> None:
        path = tmp_path / "list.pt"
        torch.save([1, 2, 3], path)
        with pytest.raises(CheckpointError, match="dictionary payload"):
            load_checkpoint(path)


class TestTrainer:
    def test_training_reduces_the_loss(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec
    ) -> None:
        loaders = build_dataloaders(synthetic_bundle, LoaderSettings(batch_size=16), seed=1)
        trainer = Trainer(
            TrainingSettings(epochs=3, optimizer=OptimizerSettings(learning_rate=0.01), topk=2)
        )
        outcome = trainer.fit(
            build_model(sample_spec),
            loaders,
            architecture_hash=architecture_hash(sample_spec),
        )
        train_losses = [entry.loss for entry in outcome.history if entry.phase == "train"]
        assert outcome.epochs_completed == 3
        assert train_losses[-1] < train_losses[0]
        assert 0.0 <= outcome.best_validation_accuracy <= 1.0

    def test_outcome_serialises(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec
    ) -> None:
        loaders = build_dataloaders(synthetic_bundle, LoaderSettings(batch_size=32), seed=1)
        trainer = Trainer(TrainingSettings(epochs=1, topk=2))
        payload = trainer.fit(
            build_model(sample_spec),
            loaders,
            architecture_hash=architecture_hash(sample_spec),
        ).to_dict()
        assert payload["epochs_completed"] == 1
        assert isinstance(payload["history"], list)

    def test_resume_continues_from_the_checkpoint(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec, tmp_path: Path
    ) -> None:
        loaders = build_dataloaders(synthetic_bundle, LoaderSettings(batch_size=32), seed=1)
        digest = architecture_hash(sample_spec)
        trainer = Trainer(TrainingSettings(epochs=1, topk=2))
        checkpoint = tmp_path / "training.pt"
        trainer.fit(
            build_model(sample_spec),
            loaders,
            architecture_hash=digest,
            checkpoint_path=checkpoint,
        )
        assert checkpoint.exists()
        outcome = trainer.fit(
            build_model(sample_spec),
            loaders,
            architecture_hash=digest,
            checkpoint_path=checkpoint,
            epochs=2,
        )
        assert outcome.epochs_completed == 2

    def test_early_stopping_ends_the_run(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec
    ) -> None:
        loaders = build_dataloaders(synthetic_bundle, LoaderSettings(batch_size=32), seed=1)
        trainer = Trainer(
            TrainingSettings(
                epochs=20,
                topk=2,
                early_stopping_patience=1,
                early_stopping_min_delta=1.0,  # nothing can ever count as an improvement
                optimizer=OptimizerSettings(learning_rate=1e-6),
            )
        )
        outcome = trainer.fit(
            build_model(sample_spec),
            loaders,
            architecture_hash=architecture_hash(sample_spec),
        )
        assert outcome.stopped_early
        assert outcome.epochs_completed < 20

    def test_divergence_is_a_permanent_failure(self, synthetic_bundle: DatasetBundle) -> None:
        loaders = build_dataloaders(synthetic_bundle, LoaderSettings(batch_size=16), seed=1)
        trainer = Trainer(TrainingSettings(epochs=2, topk=2))
        # A model that reliably emits non-finite logits. Provoking divergence through a
        # large learning rate is possible but flaky — normalisation layers often rescue it
        # — and a flaky test of the divergence path is worse than none.
        with pytest.raises(NonFiniteLossError, match="non-finite") as excinfo:
            trainer.fit(
                _DivergentModel(num_classes=4),
                loaders,
                architecture_hash="divergent",
            )
        assert excinfo.value.retriable is False

    def test_timeout_is_enforced(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec
    ) -> None:
        loaders = build_dataloaders(synthetic_bundle, LoaderSettings(batch_size=4), seed=1)
        trainer = Trainer(TrainingSettings(epochs=50, topk=2, max_seconds=0.001))
        with pytest.raises(EvaluationTimeoutError, match="exceeded its"):
            trainer.fit(
                build_model(sample_spec),
                loaders,
                architecture_hash=architecture_hash(sample_spec),
            )

    def test_empty_loader_is_rejected(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec
    ) -> None:
        loaders = build_dataloaders(
            synthetic_bundle, LoaderSettings(batch_size=1024, drop_last=True), seed=1
        )
        trainer = Trainer(TrainingSettings(epochs=1, topk=2))
        with pytest.raises(ConfigurationError, match="yields no batches"):
            trainer.fit(
                build_model(sample_spec),
                loaders,
                architecture_hash=architecture_hash(sample_spec),
            )

    def test_epoch_override_is_validated(
        self, synthetic_bundle: DatasetBundle, sample_spec: ArchitectureSpec
    ) -> None:
        loaders = build_dataloaders(synthetic_bundle, LoaderSettings(batch_size=32), seed=1)
        trainer = Trainer(TrainingSettings(epochs=1, topk=2))
        with pytest.raises(ConfigurationError, match="epochs must be"):
            trainer.fit(
                build_model(sample_spec),
                loaders,
                architecture_hash=architecture_hash(sample_spec),
                epochs=0,
            )

    def test_settings_are_validated(self) -> None:
        with pytest.raises(ConfigurationError, match="epochs"):
            TrainingSettings(epochs=0)
        with pytest.raises(ConfigurationError, match="gradient_clip_norm"):
            TrainingSettings(gradient_clip_norm=0.0)
        with pytest.raises(ConfigurationError, match="label_smoothing"):
            TrainingSettings(label_smoothing=1.0)
        with pytest.raises(ConfigurationError, match="max_seconds"):
            TrainingSettings(max_seconds=0.0)

    def test_with_epochs_preserves_other_settings(self) -> None:
        settings = TrainingSettings(epochs=3, label_smoothing=0.1, topk=2)
        rebudgeted = settings.with_epochs(9)
        assert rebudgeted.epochs == 9
        assert rebudgeted.label_smoothing == 0.1
        assert rebudgeted.topk == 2

    def test_mixed_precision_falls_back_on_cpu(self) -> None:
        trainer = Trainer(TrainingSettings(epochs=1, mixed_precision=True), device="cpu")
        assert trainer._amp_enabled is False

    def test_evaluation_mode_restores_the_previous_mode(self) -> None:
        model = nn.Linear(2, 2)
        model.train()
        with evaluation_mode(model) as inner:
            assert not inner.training
        assert model.training


class _DivergentModel(nn.Module):
    """A model whose logits are always non-finite, used to exercise divergence handling."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 16 * 16, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return infinite logits regardless of the input."""
        logits: torch.Tensor = self.linear(inputs.flatten(1))
        return logits * float("inf")
