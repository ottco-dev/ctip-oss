"""
analytics/visualization/plotter.py — Matplotlib/Plotly chart generation.

Generates publication-quality plots for:
  - Trichome count distributions
  - Maturity distribution bar/pie charts
  - Precision-recall curves
  - Confidence calibration reliability diagrams
  - Training loss curves
  - Trichome density spatial maps

All functions return the figure object — caller decides whether to show()
or save(). No display-side effects (no plt.show() calls).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend (safe in server context)


# ---------------------------------------------------------------------------
# Maturity distribution
# ---------------------------------------------------------------------------


def plot_maturity_distribution(
    fractions: dict[str, float],
    title: str = "Trichome Maturity Distribution",
    scientific_caveat: bool = True,
) -> plt.Figure:
    """
    Horizontal stacked bar chart for maturity fractions.

    Args:
        fractions: {"clear": 0.2, "cloudy": 0.6, "amber": 0.1, "mixed": 0.1}
        title: Chart title.
        scientific_caveat: Add disclaimer as figure footnote.

    Returns:
        matplotlib Figure.
    """
    colors = {
        "clear": "#a8d8f0",
        "cloudy": "#f5f5f5",
        "amber": "#ffa500",
        "mixed": "#d4c5a9",
    }
    labels = list(fractions.keys())
    values = list(fractions.values())
    bar_colors = [colors.get(k, "#cccccc") for k in labels]

    fig, ax = plt.subplots(figsize=(8, 2.5))
    left = 0.0
    for label, value, color in zip(labels, values, bar_colors):
        if value > 0:
            ax.barh(0, value, left=left, color=color, edgecolor="gray",
                    linewidth=0.5, label=f"{label} ({value:.1%})")
            if value > 0.05:
                ax.text(
                    left + value / 2, 0, f"{value:.0%}",
                    ha="center", va="center", fontsize=9, fontweight="bold"
                )
            left += value

    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel("Fraction", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)

    if scientific_caveat:
        fig.text(
            0.0, -0.15,
            "⚠ Visual maturity does NOT allow cannabinoid quantification.",
            fontsize=7, color="darkred", ha="left",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Precision-Recall curve
# ---------------------------------------------------------------------------


def plot_precision_recall_curve(
    precisions: Sequence[float],
    recalls: Sequence[float],
    auc: Optional[float] = None,
    class_name: str = "all",
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    """Plot PR curve with AUC annotation."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recalls, precisions, color="#2196F3", linewidth=2, label=class_name)
    ax.fill_between(recalls, precisions, alpha=0.1, color="#2196F3")

    if auc is not None:
        ax.text(0.05, 0.05, f"AUC = {auc:.3f}", transform=ax.transAxes,
                fontsize=11, color="#2196F3")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reliability diagram (calibration)
# ---------------------------------------------------------------------------


def plot_reliability_diagram(
    confidences: Sequence[float],
    accuracies: Sequence[float],
    bin_counts: Optional[Sequence[int]] = None,
    ece: Optional[float] = None,
    title: str = "Calibration Reliability Diagram",
) -> plt.Figure:
    """
    Plot reliability diagram (calibration curve).

    A perfectly calibrated model lies on the diagonal.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.plot(confidences, accuracies, color="#E91E63", marker="o", linewidth=2,
             markersize=5, label="Model")
    ax1.fill_between(confidences, confidences, accuracies,
                     alpha=0.2, color="#E91E63", label="Calibration gap")

    if ece is not None:
        ax1.text(0.05, 0.9, f"ECE = {ece:.4f}", transform=ax1.transAxes,
                 fontsize=11, color="#E91E63")
        target_ok = ece < 0.05
        ax1.text(0.05, 0.83, "✓ ECE < 0.05" if target_ok else "✗ ECE ≥ 0.05",
                 transform=ax1.transAxes, fontsize=9,
                 color="green" if target_ok else "red")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax1.set_ylabel("Fraction Correct", fontsize=11)
    ax1.set_title(title, fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bin histogram
    if bin_counts is not None:
        ax2.bar(confidences, bin_counts, width=1.0 / len(bin_counts),
                color="#9C27B0", alpha=0.7, edgecolor="white")
        ax2.set_xlabel("Confidence", fontsize=10)
        ax2.set_ylabel("Sample count", fontsize=10)
        ax2.set_title("Confidence histogram", fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def plot_reliability_diagram_from_bins(
    bins: list[dict],
    ece: float,
    mce: float,
    total_samples: int,
    title: str = "Calibration Reliability Diagram",
    figsize: tuple[float, float] = (10, 4.5),
) -> "plt.Figure":
    """
    Plot a reliability diagram from pre-computed BinStats dicts.

    Designed to work with ``CalibrationResponse.bins`` from
    ``analytics.api.schemas``.

    Each bin dict must contain:
        - ``mean_confidence`` (float)
        - ``accuracy`` (float)
        - ``count`` (int)
        - ``is_overconfident`` (bool)
        - ``is_empty`` (bool)

    Args:
        bins: List of per-bin statistics dicts.
        ece: Expected Calibration Error (0–1).
        mce: Maximum Calibration Error (0–1).
        total_samples: Total number of predictions.
        title: Chart title.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure (Agg backend, save with savefig() or save_figure()).
    """
    non_empty = [b for b in bins if not b.get("is_empty", True)]

    fig, (ax_diag, ax_hist) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # ── Reliability diagram (left panel) ────────────────────────────────────
    # Perfect calibration diagonal
    ax_diag.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.2,
                 label="Perfect calibration", zorder=1)

    # Per-bin bars coloured by over/underconfidence
    for b in non_empty:
        conf = b["mean_confidence"]
        acc = b["accuracy"]
        gap = abs(conf - acc)
        width = 1.0 / len(bins)  # uniform bin width

        # Bar from 0 → accuracy
        bar_color = "#EF6C00" if b.get("is_overconfident") else "#1565C0"
        ax_diag.bar(
            x=conf - width / 2,
            height=acc,
            width=width,
            align="edge",
            color=bar_color,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

        # Gap rectangle (hatched) between bar top and diagonal
        diag_at_conf = conf   # perfect calibration value = conf itself
        if b.get("is_overconfident"):
            # bar below diagonal → fill from acc → conf
            ax_diag.bar(
                x=conf - width / 2,
                height=diag_at_conf - acc,
                bottom=acc,
                width=width,
                align="edge",
                color="#EF6C00",
                alpha=0.25,
                edgecolor="none",
                zorder=3,
            )
        else:
            # bar above diagonal → fill from conf → acc
            ax_diag.bar(
                x=conf - width / 2,
                height=acc - diag_at_conf,
                bottom=diag_at_conf,
                width=width,
                align="edge",
                color="#1565C0",
                alpha=0.25,
                edgecolor="none",
                zorder=3,
            )

    # ECE / MCE annotations
    ece_color = "#2e7d32" if ece < 0.05 else ("#F57F17" if ece < 0.10 else "#c62828")
    ax_diag.text(
        0.04, 0.96,
        f"ECE = {ece:.4f}",
        transform=ax_diag.transAxes,
        fontsize=11, fontweight="bold",
        color=ece_color,
        va="top",
    )
    ax_diag.text(
        0.04, 0.88,
        f"MCE = {mce:.4f}",
        transform=ax_diag.transAxes,
        fontsize=9,
        color="#5D4037",
        va="top",
    )
    quality = (
        "Excellent" if ece < 0.02 else
        "Good" if ece < 0.05 else
        "Moderate" if ece < 0.10 else
        "Poor"
    )
    ax_diag.text(
        0.04, 0.81,
        f"Quality: {quality}",
        transform=ax_diag.transAxes,
        fontsize=9,
        color=ece_color,
        va="top",
    )

    # Legend patches
    from matplotlib.patches import Patch  # noqa: PLC0415
    legend_handles = [
        plt.Line2D([0], [0], color="k", linestyle="--", alpha=0.4, label="Perfect calibration"),
        Patch(facecolor="#EF6C00", alpha=0.75, label="Overconfident bins"),
        Patch(facecolor="#1565C0", alpha=0.75, label="Underconfident bins"),
    ]
    ax_diag.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax_diag.set_xlim(0, 1)
    ax_diag.set_ylim(0, 1.02)
    ax_diag.set_xlabel("Mean Confidence (Bin)", fontsize=10)
    ax_diag.set_ylabel("Fraction Correct", fontsize=10)
    ax_diag.set_title(title, fontsize=11, fontweight="bold")
    ax_diag.grid(True, alpha=0.25)

    # ── Confidence histogram (right panel) ──────────────────────────────────
    confs = [b["mean_confidence"] for b in non_empty]
    counts = [b["count"] for b in non_empty]
    width = 1.0 / len(bins)

    ax_hist.bar(
        [c - width / 2 for c in confs],
        counts,
        width=width,
        align="edge",
        color="#7B1FA2",
        alpha=0.7,
        edgecolor="white",
    )
    ax_hist.set_xlabel("Confidence", fontsize=9)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.set_title(f"Confidence histogram\n(n={total_samples:,})", fontsize=9)
    ax_hist.set_xlim(0, 1)
    ax_hist.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training loss curves
# ---------------------------------------------------------------------------


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    map50s: Optional[Sequence[float]] = None,
    title: str = "Training Progress",
) -> plt.Figure:
    """Plot training + validation loss and optionally mAP50."""
    n_subplots = 2 if map50s is not None else 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(12 if n_subplots == 2 else 6, 4))

    if n_subplots == 1:
        axes = [axes]

    epochs = list(range(1, len(train_losses) + 1))

    ax = axes[0]
    ax.plot(epochs, train_losses, label="Train loss", color="#2196F3", linewidth=2)
    ax.plot(epochs, val_losses, label="Val loss", color="#FF5722", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(f"{title} — Loss", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if map50s is not None and len(axes) > 1:
        ax2 = axes[1]
        ax2.plot(list(range(1, len(map50s) + 1)), map50s,
                 label="mAP50", color="#4CAF50", linewidth=2)
        best_epoch = int(np.argmax(map50s)) + 1
        best_val = max(map50s)
        ax2.axvline(x=best_epoch, color="#4CAF50", linestyle="--", alpha=0.5)
        ax2.text(best_epoch, best_val - 0.02, f"Best: {best_val:.4f}",
                 fontsize=9, color="#4CAF50")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("mAP50", fontsize=11)
        ax2.set_title(f"{title} — mAP50", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Trichome density map
# ---------------------------------------------------------------------------


def plot_density_map(
    centroids: list[tuple[float, float]],
    image_width: int,
    image_height: int,
    bandwidth: float = 30.0,
    colormap: str = "hot",
    title: str = "Trichome Density Map",
) -> plt.Figure:
    """
    Gaussian KDE density map of trichome centroids.

    Args:
        centroids: List of (x, y) centroid coordinates.
        image_width, image_height: Image dimensions.
        bandwidth: Gaussian bandwidth in pixels.
        colormap: Matplotlib colormap name.
        title: Chart title.
    """
    if not centroids:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No trichomes detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return fig

    from scipy.stats import gaussian_kde

    xs = np.array([c[0] for c in centroids])
    ys = np.array([c[1] for c in centroids])

    # Grid for KDE evaluation
    xi = np.linspace(0, image_width, min(200, image_width))
    yi = np.linspace(0, image_height, min(200, image_height))
    xi, yi = np.meshgrid(xi, yi)

    kde = gaussian_kde(np.vstack([xs, ys]), bw_method=bandwidth / image_width)
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(xi, yi, zi, cmap=colormap, shading="auto")
    ax.scatter(xs, ys, color="white", s=5, alpha=0.5, zorder=2)
    fig.colorbar(pcm, ax=ax, label="Density")

    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)  # Image coords: y increases downward
    ax.set_xlabel("X (px)", fontsize=10)
    ax.set_ylabel("Y (px)", fontsize=10)
    ax.set_title(f"{title} (n={len(centroids)})", fontsize=12, fontweight="bold")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> plt.Figure:
    """Plot confusion matrix as a heatmap."""
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(n * 1.2 + 2, n * 1.2 + 1.5))

    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    threshold = 0.5 if normalize else matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text = f"{val:.2f}" if normalize else str(int(val))
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > threshold else "black", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = 150,
    format: str = "png",
) -> Path:
    """Save figure to disk and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, format=format, bbox_inches="tight")
    plt.close(fig)
    return out
