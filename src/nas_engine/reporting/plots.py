"""Matplotlib figures for search reports.

Design decisions
----------------
**Backend.** ``Agg`` is selected before :mod:`matplotlib.pyplot` is imported. Reports are
generated in CI and in containers with no display; importing pyplot first would try to
select an interactive backend and fail or hang.

**Colour.** Three roles, and only three: a neutral for dominated candidates (context, not
identity), one categorical hue for the Pareto front, one for the best candidate. The
palette was checked with a colour-vision-deficiency validator — worst all-pairs CVD
separation ΔE 9.2, normal-vision ΔE 24.0, both above the required floors — so the marks
stay distinguishable for deuteranopic and tritanopic readers. Identity is never carried by
colour alone: every figure has a legend, and the best candidate is directly labelled with
its short hash.

**Marks.** Points carry a light surface ring so overlapping candidates remain countable;
lines are 2px; the grid is recessive and drawn beneath the data.

**Scales.** Parameter counts and MACs span orders of magnitude, so those axes are
logarithmic. A linear axis would compress every small model into one indistinguishable
cluster at the origin — the region a NAS report most needs to show.

**Not shown.** No dual-axis figures, no colour ramp standing in for a third variable, and
no chart where a number would do. Where a figure would have fewer than two usable points
it is skipped rather than drawn empty; :func:`generate_plots` reports which figures were
skipped and why.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Must precede the pyplot import: pyplot binds the backend at import time.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nas_engine.exceptions import ReportingError
from nas_engine.objectives.ranking import RankedCandidate

#: Chart surface and ink. Kept together so a future dark variant is a single substitution.
SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e0"
AXIS = "#c9c8c2"

#: Categorical slots. Validated: all-pairs CVD ΔE 9.2, normal-vision ΔE 24.0 on white.
NEUTRAL_MARK = "#9d9c95"
SERIES_PARETO = "#2a78d6"
SERIES_BEST = "#eb6834"

#: Figure geometry, in inches at 150 dpi.
FIGURE_SIZE = (7.2, 4.4)
FIGURE_DPI = 150


@dataclass(frozen=True)
class PlotResult:
    """The figures a report generated.

    Attributes:
        paths: Figure name to file path.
        skipped: Figure name to the reason it was not drawn.
    """

    paths: dict[str, Path]
    skipped: dict[str, str]

    def to_dict(self) -> dict[str, dict[str, str]]:
        """Return a JSON-serialisable representation."""
        return {
            "paths": {name: str(path) for name, path in self.paths.items()},
            "skipped": dict(self.skipped),
        }


def _style_axes(axes: Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the shared chart styling to one axes object.

    Args:
        axes: Axes to style.
        title: Chart title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """
    axes.set_title(title, color=INK_PRIMARY, fontsize=12, pad=12, loc="left")
    axes.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    axes.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    axes.grid(visible=True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=INK_SECONDARY, labelsize=9)


def _save(figure: Figure, path: Path) -> Path:
    """Write a figure and close it.

    Closing matters: matplotlib keeps every open figure alive, so a report generator that
    forgets grows without bound across a long run.

    Args:
        figure: Figure to save.
        path: Destination file.

    Returns:
        The resolved path.

    Raises:
        ReportingError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE, bbox_inches="tight")
    except OSError as exc:
        msg = f"could not write plot to {path}: {exc}"
        raise ReportingError(msg, details={"path": str(path), "error": str(exc)}) from exc
    finally:
        plt.close(figure)
    return path.resolve()


def _scatter_metric(
    candidates: Sequence[RankedCandidate],
    *,
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    log_x: bool = False,
) -> Path | None:
    """Draw a two-metric scatter with the Pareto front and best candidate highlighted.

    Args:
        candidates: Ranked candidates to plot.
        x_metric: Metric on the x-axis.
        y_metric: Metric on the y-axis.
        title: Chart title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        path: Destination file.
        log_x: Whether the x-axis is logarithmic.

    Returns:
        The written path, or ``None`` when there was not enough data.
    """
    points = [
        candidate
        for candidate in candidates
        if x_metric in candidate.metrics and y_metric in candidate.metrics
    ]
    if len(points) < 2:
        return None

    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    dominated = [candidate for candidate in points if not candidate.on_pareto_front]
    front = [candidate for candidate in points if candidate.on_pareto_front]
    best = points[0] if points[0].rank == 0 else None

    if dominated:
        axes.scatter(
            [candidate.metrics[x_metric] for candidate in dominated],
            [candidate.metrics[y_metric] for candidate in dominated],
            s=55,
            c=NEUTRAL_MARK,
            edgecolors=SURFACE,
            linewidths=1.5,
            label=f"dominated ({len(dominated)})",
            zorder=2,
        )
    if front:
        ordered = sorted(front, key=lambda candidate: candidate.metrics[x_metric])
        axes.plot(
            [candidate.metrics[x_metric] for candidate in ordered],
            [candidate.metrics[y_metric] for candidate in ordered],
            color=SERIES_PARETO,
            linewidth=2.0,
            alpha=0.55,
            zorder=3,
        )
        axes.scatter(
            [candidate.metrics[x_metric] for candidate in front],
            [candidate.metrics[y_metric] for candidate in front],
            s=95,
            c=SERIES_PARETO,
            edgecolors=SURFACE,
            linewidths=1.5,
            label=f"Pareto front ({len(front)})",
            zorder=4,
        )
    if best is not None:
        # An open ring rather than a filled marker: the best candidate is almost always a
        # Pareto member too, and a solid mark would hide the point underneath it, making
        # the legend's front count disagree with the number of visible blue dots.
        axes.scatter(
            [best.metrics[x_metric]],
            [best.metrics[y_metric]],
            s=260,
            marker="o",
            facecolors="none",
            edgecolors=SERIES_BEST,
            linewidths=2.2,
            label="best (ranked first)",
            zorder=5,
        )
        # A direct label so the headline candidate is identified without colour alone.
        axes.annotate(
            best.architecture_hash[:8],
            (best.metrics[x_metric], best.metrics[y_metric]),
            textcoords="offset points",
            xytext=(10, 8),
            color=INK_SECONDARY,
            fontsize=9,
        )

    if log_x:
        axes.set_xscale("log")
    _style_axes(axes, title=title, xlabel=xlabel, ylabel=ylabel)
    legend = axes.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return _save(figure, path)


def plot_accuracy_vs_parameters(candidates: Sequence[RankedCandidate], path: Path) -> Path | None:
    """Plot validation accuracy against trainable parameter count.

    Args:
        candidates: Ranked candidates.
        path: Destination file.

    Returns:
        The written path, or ``None`` when there was not enough data.
    """
    return _scatter_metric(
        candidates,
        x_metric="trainable_parameters",
        y_metric="validation_accuracy",
        title="Accuracy versus model size",
        xlabel="trainable parameters (log scale)",
        ylabel="validation accuracy",
        path=path,
        log_x=True,
    )


def plot_accuracy_vs_latency(candidates: Sequence[RankedCandidate], path: Path) -> Path | None:
    """Plot validation accuracy against measured inference latency.

    Args:
        candidates: Ranked candidates.
        path: Destination file.

    Returns:
        The written path, or ``None`` when latency was not measured.
    """
    return _scatter_metric(
        candidates,
        x_metric="latency_median_ms",
        y_metric="validation_accuracy",
        title="Accuracy versus inference latency (this machine only)",
        xlabel="median latency per batch (ms)",
        ylabel="validation accuracy",
        path=path,
    )


def plot_search_progress(
    history: Sequence[tuple[int, float]], path: Path, *, metric_label: str = "validation accuracy"
) -> Path | None:
    """Plot per-evaluation scores and the running best.

    The running-best line is the honest way to show search progress: individual scores
    bounce around, and a trend line through them would suggest a smoothness the data does
    not have.

    Args:
        history: ``(evaluation_index, value)`` pairs in completion order.
        path: Destination file.
        metric_label: Axis label for the metric being tracked.

    Returns:
        The written path, or ``None`` when there was not enough data.
    """
    if len(history) < 2:
        return None

    indices = [index for index, _ in history]
    values = [value for _, value in history]
    running_best: list[float] = []
    best = float("-inf")
    for value in values:
        best = max(best, value)
        running_best.append(best)

    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    axes.scatter(
        indices,
        values,
        s=55,
        c=NEUTRAL_MARK,
        edgecolors=SURFACE,
        linewidths=1.5,
        label="individual evaluations",
        zorder=2,
    )
    axes.plot(
        indices,
        running_best,
        color=SERIES_PARETO,
        linewidth=2.0,
        label="best so far",
        zorder=3,
    )
    axes.annotate(
        f"{running_best[-1]:.4f}",
        (indices[-1], running_best[-1]),
        textcoords="offset points",
        xytext=(8, -4),
        color=INK_SECONDARY,
        fontsize=9,
    )

    _style_axes(
        axes,
        title="Search progress",
        xlabel="completed evaluations",
        ylabel=metric_label,
    )
    # `loc="best"` lets matplotlib place the legend in the emptiest quadrant, which matters
    # here because the shape of a progress curve is not known in advance.
    legend = axes.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return _save(figure, path)


def plot_population_statistics(
    generations: Sequence[tuple[int, float, float, float]], path: Path
) -> Path | None:
    """Plot evolutionary population fitness over generations.

    Args:
        generations: ``(generation, best, mean, worst)`` tuples.
        path: Destination file.

    Returns:
        The written path, or ``None`` when there was not enough data.
    """
    if len(generations) < 2:
        return None

    axis = [entry[0] for entry in generations]
    best = [entry[1] for entry in generations]
    mean = [entry[2] for entry in generations]
    worst = [entry[3] for entry in generations]

    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    axes.fill_between(axis, worst, best, color=SERIES_PARETO, alpha=0.12, zorder=2)
    axes.plot(axis, mean, color=SERIES_PARETO, linewidth=2.0, label="population mean", zorder=3)
    axes.plot(
        axis,
        best,
        color=SERIES_BEST,
        linewidth=2.0,
        linestyle="--",
        label="population best",
        zorder=4,
    )

    _style_axes(
        axes,
        title="Population fitness over generations",
        xlabel="generation",
        ylabel="objective value (higher is better)",
    )
    legend = axes.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return _save(figure, path)


def generate_plots(
    candidates: Sequence[RankedCandidate],
    history: Sequence[tuple[int, float]],
    output_dir: Path,
    *,
    prefix: str,
    population: Sequence[tuple[int, float, float, float]] | None = None,
) -> PlotResult:
    """Generate every applicable figure for a report.

    Filenames are deterministic: ``<prefix>_<figure>.png``. Regenerating a report
    overwrites the same files rather than accumulating timestamped duplicates.

    Args:
        candidates: Ranked candidates.
        history: ``(evaluation_index, value)`` pairs for the progress figure.
        output_dir: Directory to write into.
        prefix: Filename prefix, normally the search id.
        population: Optional per-generation population statistics.

    Returns:
        A :class:`PlotResult`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    skipped: dict[str, str] = {}

    figures: list[tuple[str, Path | None, str]] = [
        (
            "accuracy_vs_parameters",
            plot_accuracy_vs_parameters(
                candidates, output_dir / f"{prefix}_accuracy_vs_parameters.png"
            ),
            "fewer than two candidates reported both accuracy and parameter count",
        ),
        (
            "accuracy_vs_latency",
            plot_accuracy_vs_latency(candidates, output_dir / f"{prefix}_accuracy_vs_latency.png"),
            "latency was not measured, or fewer than two candidates reported it",
        ),
        (
            "search_progress",
            plot_search_progress(history, output_dir / f"{prefix}_search_progress.png"),
            "fewer than two completed evaluations",
        ),
    ]
    if population:
        figures.append(
            (
                "population_statistics",
                plot_population_statistics(
                    population, output_dir / f"{prefix}_population_statistics.png"
                ),
                "fewer than two recorded generations",
            )
        )

    for name, result, reason in figures:
        if result is None:
            skipped[name] = reason
        else:
            paths[name] = result
    return PlotResult(paths=paths, skipped=skipped)


__all__ = [
    "FIGURE_DPI",
    "FIGURE_SIZE",
    "PlotResult",
    "generate_plots",
    "plot_accuracy_vs_latency",
    "plot_accuracy_vs_parameters",
    "plot_population_statistics",
    "plot_search_progress",
]
