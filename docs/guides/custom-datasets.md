# Custom datasets

Searching over your own data.

## The contract

```python
@runtime_checkable
class DatasetProvider(Protocol):
    @property
    def name(self) -> str: ...
    def build(self) -> DatasetBundle: ...
```

A `Protocol` rather than an abstract base class, so your code can supply a dataset **without
importing from this package** — structural typing keeps the dependency arrow pointing the
right way.

```python
@dataclass(frozen=True)
class DatasetBundle:
    name: str
    train: Dataset[tuple[torch.Tensor, int]]
    validation: Dataset[tuple[torch.Tensor, int]]
    test: Dataset[tuple[torch.Tensor, int]]
    num_classes: int
    input_channels: int
    input_size: int
    description: str = ""
```

Requirements:

- Each split is a map-style `Dataset` returning `(image_tensor, integer_label)`.
- Images are `(channels, height, width)` float tensors, already normalised.
- Images are **square**. The search space assumes it.
- Each split implements `__len__`. Iterable-style datasets are not supported.
- `num_classes >= 2`, `input_channels >= 1`, `input_size >= 4`.

## A minimal provider

```python
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from nas_engine.datasets.base import DatasetBundle
from nas_engine.exceptions import DatasetError


class TensorImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Images and labels held in memory."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if images.shape[0] != labels.shape[0]:
            raise DatasetError(
                f"images and labels disagree: {images.shape[0]} vs {labels.shape[0]}"
            )
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self._images[index], int(self._labels[index].item())


@dataclass(frozen=True)
class MyDatasetProvider:
    """Loads a pre-split dataset from three .pt files."""

    name: str = "my_dataset"
    root: Path = Path("data/mine")
    num_classes: int = 5
    input_size: int = 32

    def build(self) -> DatasetBundle:
        splits = {}
        for split in ("train", "validation", "test"):
            path = self.root / f"{split}.pt"
            if not path.is_file():
                raise DatasetError(
                    f"missing split file: {path}. Expected train.pt, validation.pt, and "
                    f"test.pt under {self.root}.",
                    details={"path": str(path)},
                )
            # weights_only=True: never unpickle arbitrary objects from a data file.
            payload = torch.load(path, map_location="cpu", weights_only=True)
            splits[split] = TensorImageDataset(payload["images"], payload["labels"])

        return DatasetBundle(
            name=self.name,
            train=splits["train"],
            validation=splits["validation"],
            test=splits["test"],
            num_classes=self.num_classes,
            input_channels=3,
            input_size=self.input_size,
            description=f"Custom dataset from {self.root}",
        )
```

## Register and use it

```python
from nas_engine.datasets.registry import register_provider
register_provider("my_dataset", MyDatasetProvider)
```

```yaml
dataset:
  provider: my_dataset
  batch_size: 64
  options:
    root: data/mine
    num_classes: 5
    input_size: 32
```

Everything in `options` is passed to the provider's constructor. A mismatch produces a clear
error rather than a `TypeError`:

```text
dataset provider 'my_dataset' rejected the supplied options ['num_clases', 'root']:
__init__() got an unexpected keyword argument 'num_clases'
```

Registration must happen before the engine builds the dataset. Import your module first:

```python
import my_project.datasets      # calls register_provider at import time
from nas_engine import SearchConfig, SearchEngine

engine = SearchEngine(SearchConfig.from_yaml("configs/my-search.yaml"))
```

There is deliberately no auto-discovery — importing a module named in a configuration file
is arbitrary code execution.

## Or skip the registry entirely

For a script, inject the bundle:

```python
engine = SearchEngine(config, dataset=my_bundle)
```

The provider registry exists so a *configuration file* can name a dataset. If you are
already writing Python, this is simpler.

---

## Getting the splits right

This is the part that matters most, and the part most often got wrong.

### Three splits, three jobs

| Split        | Job                       | Touched                           |
| ------------ | ------------------------- | --------------------------------- |
| `train`      | Fits the weights          | Every epoch of every candidate    |
| `validation` | Ranks architectures       | Once per epoch of every candidate |
| `test`       | The final reported number | **Once**, at the very end         |

Because the search *selects on* validation accuracy, that split stops being an unbiased
estimate of generalisation. Only the test split gives one, and only if it is used once. See
[common pitfalls](../concepts/common-pitfalls.md#4-reusing-the-test-set-during-search-causes-leakage).

### Split deterministically

A split that changes between runs makes results incomparable. Derive it from a seed:

```python
from nas_engine.utilities.seeding import derive_seed

generator = torch.Generator().manual_seed(derive_seed(self.seed, "my_dataset:split"))
permutation = torch.randperm(total, generator=generator).tolist()
validation_indices = permutation[: self.validation_samples]
train_indices = permutation[self.validation_samples :]
```

A *random* permutation rather than a prefix: dataset files are often ordered by class, and
taking the first *N* would silently drop classes from one split.

### Do not leak between splits

Common leakage sources, all of which inflate validation accuracy without inflating test
accuracy — so they look like the search working:

- Augmented copies of the same source image in both train and validation.
- Frames from the same video, or crops from the same scan, split across train and
  validation.
- Normalisation statistics computed over the whole dataset rather than over train alone.
- Duplicate images. Deduplicate before splitting.

### Augment training only

```python
train = MyDataset(root, transform=train_transform)        # crop, flip, jitter
validation = MyDataset(root, transform=eval_transform)    # normalise only
```

The CIFAR-10 provider does this by opening the underlying dataset twice with different
transforms and indexing both through `Subset`, so validation accuracy measures the model
rather than the augmentation lottery.

---

## Sizing the validation split

The validation split size sets your measurement resolution:

$$
\mathrm{SE} = \sqrt{\frac{p(1-p)}{n}}
$$

| Split size | SE at $p = 0.85$ | Smallest detectable difference (≈2 SE) |
| ---------: | ---------------: | -------------------------------------: |
|        256 |             2.2% |                                   4.5% |
|      1 000 |             1.1% |                                   2.3% |
|      5 000 |             0.5% |                                   1.0% |
|     10 000 |            0.36% |                                   0.7% |

**If the differences between your candidates are smaller than that, the search is ranking
noise.** Either enlarge the validation split or accept that only large differences are
detectable.

The trade-off is that validation examples are taken from training. With a small dataset,
5–20% is typical.

---

## Working with a small dataset

NAS needs enough validation data to discriminate, which is awkward when data is scarce.

**Option 1: fewer candidates, more validation data.** Give validation 20–30% and run fewer
evaluations. Better to rank 20 candidates reliably than 200 unreliably.

**Option 2: cross-validation.** Not supported natively — it multiplies every evaluation's
cost by *k*. Implement it in a custom evaluator if you need it.

**Option 3: accept the resolution and search coarsely.** With a small validation split,
search over decisions that produce *large* differences (depth, width) rather than small
ones (activation choice).

---

## Non-image data

The default search space builds 2-D convolutional networks. For other modalities:

**1-D signals** (audio, time series). Reshape to `(channels, 1, length)` and use kernels of
shape $k \times 1$ — or, better, define a search space over `Conv1d` operations, which means
[adding operations](adding-an-operation.md) and a matching builder.

**Tabular data.** The convolutional space does not apply. You would need a new operation
vocabulary (linear layers, embeddings) and a new builder. The genotype, hashing, search
strategies, objectives, persistence, and reporting all carry over unchanged — which is a
reasonable amount of the system.

**Segmentation and detection.** The classifier head assumes global pooling to a fixed-width
vector. A different head, and a different loss, would be needed. Everything upstream of the
head applies.

---

## Testing a provider

```python
def test_my_provider_builds():
    bundle = MyDatasetProvider(root=Path("tests/fixtures/mini")).build()
    sizes = bundle.split_sizes()
    assert sizes["train"] > 0 and sizes["validation"] > 0 and sizes["test"] > 0

    image, label = bundle.train[0]
    assert image.shape == (bundle.input_channels, bundle.input_size, bundle.input_size)
    assert 0 <= label < bundle.num_classes


def test_splits_are_disjoint():
    bundle = MyDatasetProvider().build()
    train_ids = {id_of(bundle.train[i]) for i in range(len(bundle.train))}
    val_ids = {id_of(bundle.validation[i]) for i in range(len(bundle.validation))}
    assert not (train_ids & val_ids)


def test_split_is_deterministic():
    first = MyDatasetProvider(seed=1).build()
    second = MyDatasetProvider(seed=1).build()
    assert torch.equal(first.train[0][0], second.train[0][0])


def test_the_task_is_learnable():
    # If a linear probe cannot beat chance, every downstream training test is measuring
    # nothing. The synthetic provider's own test does exactly this.
    ...
```

That last test is worth writing. It is the difference between "the search found nothing" and
"the search found nothing *because the data has no signal*", and the two look identical in
a report.

**Keep it offline.** A provider that downloads in a test makes CI flaky and slow. The
CIFAR-10 provider requires `download: true` explicitly, and the test suite never sets it.

## See also

- [Training and evaluation](../concepts/training-and-evaluation.md) — how data is used.
- [Running a search](running-a-search.md) — configuring the dataset section.
- [Common pitfalls](../concepts/common-pitfalls.md) — leakage and noise.
