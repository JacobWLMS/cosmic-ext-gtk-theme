"""
Phase 3 Visualisation Toolkit

Statistical plotting for identity analysis experiments.
Built on seaborn + matplotlib. Designed for small-n comparisons
where individual data points must always be visible.

Usage:
    from phase3.shared.visualisation import *
    fig = plot_injection_bimodal(df, "name_tokens")
    canvas.add_image(fig, title="...")
    plt.close(fig)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional

# ===== Colour Palettes =====

# Categorical: strategies
STRATEGY_COLORS = {
    "all_tokens": "#2c7bb6",
    "all_positions": "#2c7bb6",
    "name_tokens": "#d7191c",
    "name_tokens_only": "#d7191c",
    "name_mean": "#d7191c",
    "high_identity": "#fdae61",
    "high_identity_only": "#fdae61",
    "non_name_mean": "#abd9e9",
    "all_mean": "#2c7bb6",
    "first": "#999999",
    "last": "#666666",
}

# Token types
TOKEN_TYPE_COLORS = {
    "name": "#d7191c",
    "post_name": "#fdae61",
    "pre_name": "#abd9e9",
    "scene": "#2c7bb6",
    "suffix": "#7b3294",
    "system_prefix": "#cccccc",
    "unknown": "#999999",
}

# Sequential: for ranking celebrities by score
SCORE_CMAP = "RdYlGn"

# Style defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


# ===== Core Plot Types =====

def plot_swarm_violin(df, x, y, title="", hue=None, palette=None,
                      threshold=None, threshold_label="threshold",
                      figsize=(10, 6)):
    """Swarm plot with violin overlay. Shows individual points AND distribution.
    Use for any small-n comparison where data might be bimodal."""
    fig, ax = plt.subplots(figsize=figsize)

    if palette is None:
        palette = STRATEGY_COLORS

    # Violin behind
    sns.violinplot(data=df, x=x, y=y, hue=hue, ax=ax, inner=None,
                   alpha=0.3, palette=palette, legend=False)

    # Swarm on top
    sns.swarmplot(data=df, x=x, y=y, hue=hue, ax=ax, size=6,
                  palette=palette, legend=False)

    if threshold is not None:
        ax.axhline(y=threshold, linestyle="--", color="red", alpha=0.7,
                   label="%s (%.2f)" % (threshold_label, threshold))
        ax.legend()

    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_paired_lines(df, x_col, y_col, group_col, title="",
                      highlight_mean=True, figsize=(10, 6)):
    """Paired line plot: one line per group connecting values across x categories.
    Shows whether rankings are consistent per-group or driven by outliers."""
    fig, ax = plt.subplots(figsize=figsize)

    categories = df[x_col].unique()
    x_positions = {cat: i for i, cat in enumerate(categories)}

    # Individual lines (light)
    for group_name, group_df in df.groupby(group_col):
        xs = [x_positions[row[x_col]] for _, row in group_df.iterrows()]
        ys = [row[y_col] for _, row in group_df.iterrows()]
        ax.plot(xs, ys, color="#cccccc", alpha=0.4, linewidth=0.8)
        # Points
        ax.scatter(xs, ys, color="#999999", s=15, alpha=0.5, zorder=3)

    # Mean line (bold)
    if highlight_mean:
        means = df.groupby(x_col)[y_col].mean()
        mean_xs = [x_positions[cat] for cat in categories]
        mean_ys = [means[cat] for cat in categories]
        ax.plot(mean_xs, mean_ys, color="#d7191c", linewidth=3,
                marker="o", markersize=10, zorder=5, label="Mean")
        for xi, yi, cat in zip(mean_xs, mean_ys, categories):
            ax.annotate("%.3f" % yi, (xi, yi), textcoords="offset points",
                        xytext=(8, 5), fontsize=9, fontweight="bold")
        ax.legend()

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_token_heatmap(df, title="Token Position Identity Map", figsize=(14, 5)):
    """Annotated heatmap: token positions x metrics, with token text below.
    For Exp 2 per-position analysis."""
    metrics = ["discrimination_ratio", "silhouette"]
    available = [m for m in metrics if m in df.columns]
    if not available:
        return None

    positions = df["position"].values
    token_labels = df["ref_token"].values if "ref_token" in df.columns else positions

    # Build matrix
    data = np.zeros((len(available), len(positions)))
    for i, metric in enumerate(available):
        data[i] = df[metric].values

    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")

    # Annotations
    for i in range(len(available)):
        for j in range(len(positions)):
            val = data[i, j]
            color = "white" if abs(val) > (data.max() - data.min()) * 0.6 else "black"
            ax.text(j, i, "%.2f" % val, ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([str(p) for p in positions], fontsize=8)
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels(available)

    # Token text below
    ax2 = ax.secondary_xaxis("bottom")
    ax2.set_xticks(range(len(positions)))
    token_strs = [repr(str(t).strip())[:8] for t in token_labels]
    ax2.set_xticklabels(token_strs, fontsize=7, rotation=45, ha="right")
    ax2.tick_params(pad=15)

    # Color by token type
    if "token_type" in df.columns:
        for j, ttype in enumerate(df["token_type"].values):
            color = TOKEN_TYPE_COLORS.get(ttype, "#ffffff")
            rect = plt.Rectangle((j - 0.5, -0.5), 1, len(available),
                                 fill=True, facecolor=color, alpha=0.15,
                                 edgecolor="none")
            ax.add_patch(rect)

        # Legend for token types
        present_types = df["token_type"].unique()
        patches = [mpatches.Patch(color=TOKEN_TYPE_COLORS.get(t, "#999"),
                                  alpha=0.4, label=t) for t in present_types]
        ax.legend(handles=patches, loc="upper right", fontsize=8,
                  title="Token Type")

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_bimodal_strip(df, celebrity_col, score_col, strategy_col,
                       title="", threshold=None, figsize=(12, 6)):
    """Strip plot for bimodal data. Each celebrity is a row, strategies as colours.
    Designed for the Exp 4 injection results."""
    fig, ax = plt.subplots(figsize=figsize)

    strategies = df[strategy_col].unique()

    # Sort celebrities by their best score
    best_scores = df.groupby(celebrity_col)[score_col].max().sort_values(ascending=False)
    celeb_order = best_scores.index.tolist()

    y_positions = {celeb: i for i, celeb in enumerate(celeb_order)}

    for strat in strategies:
        sdf = df[df[strategy_col] == strat]
        ys = [y_positions[c] for c in sdf[celebrity_col]]
        xs = sdf[score_col].values
        color = STRATEGY_COLORS.get(strat, "#999999")
        ax.scatter(xs, ys, color=color, s=80, label=strat, alpha=0.8,
                   edgecolors="white", linewidth=0.5, zorder=3)

    if threshold is not None:
        ax.axvline(x=threshold, linestyle="--", color="red", alpha=0.7,
                   label="threshold (%.1f)" % threshold)

    ax.set_yticks(range(len(celeb_order)))
    ax.set_yticklabels(celeb_order)
    ax.set_xlabel(score_col)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_silhouette_by_config(df, title="Silhouette Scores by Strategy and PCA Dims",
                              figsize=(12, 6)):
    """Line plot: silhouette vs PCA dimensions, one line per strategy.
    For Exp 1 clustering analysis."""
    fig, ax = plt.subplots(figsize=figsize)

    strategies = df["strategy"].unique()

    for strat in strategies:
        sdf = df[df["strategy"] == strat].sort_values("pca_dims")
        color = STRATEGY_COLORS.get(strat, "#999999")
        ax.plot(sdf["pca_dims"].astype(str), sdf["silhouette"],
                marker="o", label=strat, color=color, linewidth=2)

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5,
               label="threshold (0.3)")
    ax.set_xlabel("PCA Dimensions")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bar_by_category(df, x_col, y_col, category_col, title="",
                         threshold=None, figsize=(12, 6)):
    """Grouped bar chart coloured by category.
    Use for any metric broken down by a categorical variable."""
    fig, ax = plt.subplots(figsize=figsize)

    categories = df[category_col].unique()
    x_vals = df[x_col].unique()
    n_cats = len(categories)
    width = 0.8 / max(n_cats, 1)

    for i, cat in enumerate(categories):
        cdf = df[df[category_col] == cat]
        positions = np.arange(len(x_vals)) + i * width - (n_cats - 1) * width / 2
        color = TOKEN_TYPE_COLORS.get(cat, STRATEGY_COLORS.get(cat, "#999999"))
        # Match x values
        vals = []
        for xv in x_vals:
            row = cdf[cdf[x_col] == xv]
            vals.append(row[y_col].values[0] if len(row) > 0 else 0)
        ax.bar(positions, vals, width=width, label=cat, color=color, alpha=0.8)

    if threshold is not None:
        ax.axhline(y=threshold, linestyle="--", color="red", alpha=0.7)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha="right")
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ===== Convenience Functions =====

def save_and_show(fig, path, canvas=None, title=""):
    """Save figure to disk AND optionally add to research-lab canvas."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if canvas is not None:
        canvas.add_image(fig, title=title)
    plt.close(fig)
