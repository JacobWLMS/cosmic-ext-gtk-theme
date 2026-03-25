"""Plotting utilities for identity signal analysis."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_image_grid(
    images: list,
    titles: list[str],
    save_path: Path,
    ncols: int = 4,
    figsize_per_img: float = 3.0,
    suptitle: Optional[str] = None,
):
    """Plot a grid of images with titles."""
    n = len(images)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_img, nrows * figsize_per_img),
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx < n:
            ax.imshow(images[idx])
            ax.set_title(titles[idx], fontsize=9)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_frequency_heatmaps(
    magnitude: np.ndarray,
    save_path: Path,
    title: str = "Frequency Magnitude per Channel",
    channel_names: Optional[list[str]] = None,
):
    """Plot frequency magnitude heatmaps for each channel.

    Args:
        magnitude: [C, H, W] array of frequency magnitudes
    """
    C = magnitude.shape[0]
    ncols = min(4, C)
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for c in range(C):
        r, col = divmod(c, ncols)
        ax = axes[r, col]
        im = ax.imshow(
            np.log1p(magnitude[c]),
            cmap="viridis",
            aspect="auto",
        )
        name = channel_names[c] if channel_names else f"Ch {c}"
        ax.set_title(name, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Turn off unused axes
    for idx in range(C, nrows * ncols):
        r, col = divmod(idx, ncols)
        axes[r, col].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_difference_heatmaps(
    diff: np.ndarray,
    save_path: Path,
    title: str = "Latent Difference per Channel",
):
    """Plot per-channel difference heatmaps.

    Args:
        diff: [C, H, W] array
    """
    C = diff.shape[0]
    ncols = min(4, C)
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    vmax = np.percentile(np.abs(diff), 99)
    for c in range(C):
        r, col = divmod(c, ncols)
        ax = axes[r, col]
        im = ax.imshow(diff[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Ch {c}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    for idx in range(C, nrows * ncols):
        r, col = divmod(idx, ncols)
        axes[r, col].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_frequency_band_comparison(
    energies_a: np.ndarray,
    energies_b: np.ndarray,
    save_path: Path,
    label_a: str = "Identity A",
    label_b: str = "Identity B",
    title: str = "Frequency Band Energy Comparison",
):
    """Plot frequency band energy comparison per channel.

    Args:
        energies_a, energies_b: [C, n_bands] arrays
    """
    C, n_bands = energies_a.shape
    ncols = min(4, C)
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    bands = np.arange(n_bands)
    for c in range(C):
        r, col = divmod(c, ncols)
        ax = axes[r, col]
        ax.bar(bands - 0.15, energies_a[c], width=0.3, label=label_a, alpha=0.7)
        ax.bar(bands + 0.15, energies_b[c], width=0.3, label=label_b, alpha=0.7)
        ax.set_title(f"Ch {c}", fontsize=10)
        ax.set_xlabel("Frequency Band")
        ax.set_ylabel("Energy")
        if c == 0:
            ax.legend(fontsize=7)

    for idx in range(C, nrows * ncols):
        r, col = divmod(idx, ncols)
        axes[r, col].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_line_chart(
    x: np.ndarray,
    y_dict: dict[str, np.ndarray],
    save_path: Path,
    xlabel: str = "Step",
    ylabel: str = "Value",
    title: str = "",
    yerr_dict: Optional[dict[str, np.ndarray]] = None,
):
    """Plot line chart with multiple series and optional error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, y in y_dict.items():
        yerr = yerr_dict.get(label) if yerr_dict else None
        ax.errorbar(x, y, yerr=yerr, label=label, marker="o", markersize=3, capsize=2)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_channel_importance(
    face_diffs: np.ndarray,
    non_face_diffs: np.ndarray,
    save_path: Path,
    title: str = "Channel Importance: Face vs Non-Face Region Difference",
):
    """Plot per-channel face vs non-face difference magnitudes.

    Args:
        face_diffs: [C] average absolute difference in face region per channel
        non_face_diffs: [C] average absolute difference in non-face region per channel
    """
    C = len(face_diffs)
    fig, ax = plt.subplots(figsize=(max(8, C * 0.8), 5))
    x = np.arange(C)
    ax.bar(x - 0.2, face_diffs, width=0.4, label="Face Region", color="coral")
    ax.bar(x + 0.2, non_face_diffs, width=0.4, label="Non-Face Region", color="steelblue")
    ax.set_xlabel("Channel", fontsize=11)
    ax.set_ylabel("Mean Absolute Difference", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pca_scatter(
    coords: np.ndarray,
    labels: list[str],
    save_path: Path,
    title: str = "PCA of Identity Latents",
):
    """2D/3D PCA scatter plot colored by identity label.

    Args:
        coords: [N, 2] or [N, 3] array of PCA coordinates
        labels: list of identity labels per point
    """
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    if coords.shape[1] >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for lbl in unique_labels:
            mask = [l == lbl for l in labels]
            pts = coords[mask]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label=lbl, color=color_map[lbl], s=30)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        for lbl in unique_labels:
            mask = [l == lbl for l in labels]
            pts = coords[mask]
            ax.scatter(pts[:, 0], pts[:, 1], label=lbl, color=color_map[lbl], s=40)
        ax.set_xlabel("PC1", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
