"""Experiment 7: PCA on Identity.

Question: Is identity a linear subspace in latent space?
Focus: Compare PCA clustering on Ch 3 (identity fingerprint) vs full latent vs individual channels.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from identity_analysis.pipeline import PipelineWrapper
from identity_analysis.plotting import plot_line_chart, plot_pca_scatter
from identity_analysis.utils import get_celebrity_prompts, get_output_dir


def run(
    model_type: str = "sdxl",
    n_seeds: int = 10,
    save_latents: bool = False,
    output_base: str = "outputs",
    num_steps: int = 20,
):
    """Run Experiment 7: PCA on Identity.

    Args:
        n_seeds: Number of prompts per celebrity (max 20 templates available).
    """
    import torch; torch.cuda.empty_cache(); gc.collect()
    out_dir = get_output_dir("experiment_7", output_base)
    print(f"Experiment 7 output: {out_dir}")

    pipe = PipelineWrapper(model_type)
    celeb_prompts = get_celebrity_prompts()
    n_prompts = min(n_seeds, 20)  # Max 20 templates available

    # Collect latents from generation (not re-encoding)
    all_latents = []  # Raw final latents [1, 4, H, W]
    all_labels = []

    for celeb, prompts in celeb_prompts.items():
        print(f"\nGenerating for: {celeb}")
        prompts_to_use = prompts[:n_prompts]

        for p_idx, prompt in enumerate(tqdm(prompts_to_use, desc=f"  {celeb}")):
            seed = p_idx
            res = pipe.generate(
                prompt, seed,
                num_inference_steps=num_steps,
                save_step_latents=False,
            )

            latent = res["final_latent"]  # Shape: [1, 4, 128, 128] for SDXL 1024x1024
            all_latents.append(latent)
            all_labels.append(celeb)

            if save_latents:
                np.save(
                    out_dir / "latents" / f"{celeb.replace(' ', '_')}_p{p_idx}.npy",
                    latent,
                )

            del res
            gc.collect()

    pipe.cleanup()

    # Stack all latents: [N, 1, C, H, W] -> [N, C, H, W]
    latents = np.array(all_latents)
    if latents.ndim == 5:
        latents = latents[:, 0]  # Remove batch dim
    N, C, H, W = latents.shape
    print(f"\nLatent shape: {latents.shape} ({N} samples, {C} channels, {H}x{W})")

    # === PCA Analysis on Multiple Views ===

    views = {
        "full": latents.reshape(N, -1),          # All 4 channels flattened
        "ch0": latents[:, 0].reshape(N, -1),     # Luminance only
        "ch1": latents[:, 1].reshape(N, -1),     # Red-cyan chrominance
        "ch2": latents[:, 2].reshape(N, -1),     # Warm-cool chrominance
        "ch3": latents[:, 3].reshape(N, -1),     # Pattern/structure (identity fingerprint)
    }

    view_labels = {
        "full": "Full Latent (4ch)",
        "ch0": "Ch 0 (Luminance)",
        "ch1": "Ch 1 (Red-Cyan)",
        "ch2": "Ch 2 (Warm-Cool)",
        "ch3": "Ch 3 (Identity Fingerprint)",
    }

    all_results = []
    unique_labels = sorted(set(all_labels))
    label_ids = [unique_labels.index(l) for l in all_labels]

    for view_name, X in views.items():
        print(f"\n{'='*50}")
        print(f"PCA on: {view_labels[view_name]} — shape {X.shape}")

        n_components = min(30, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        # Explained variance plot
        plot_line_chart(
            np.arange(1, n_components + 1),
            {
                "Individual": pca.explained_variance_ratio_,
                "Cumulative": np.cumsum(pca.explained_variance_ratio_),
            },
            out_dir / "plots" / f"explained_variance_{view_name}.png",
            xlabel="Principal Component",
            ylabel="Explained Variance Ratio",
            title=f"PCA Variance — {view_labels[view_name]}",
        )

        # 2D scatter
        plot_pca_scatter(
            X_pca[:, :2],
            all_labels,
            out_dir / "plots" / f"pca_2d_{view_name}.png",
            title=f"PCA 2D — {view_labels[view_name]}",
        )

        # 3D scatter for key views
        if view_name in ("full", "ch3") and n_components >= 3:
            plot_pca_scatter(
                X_pca[:, :3],
                all_labels,
                out_dir / "plots" / f"pca_3d_{view_name}.png",
                title=f"PCA 3D — {view_labels[view_name]}",
            )

        # Silhouette scores at various dimensions
        for n_dim in [2, 3, 5, 10, 20]:
            if n_dim <= n_components and N > n_dim:
                score = silhouette_score(X_pca[:, :n_dim], label_ids)
                all_results.append({
                    "view": view_name,
                    "view_label": view_labels[view_name],
                    "n_dims": n_dim,
                    "silhouette": score,
                })
                print(f"  Silhouette (top {n_dim} PCs): {score:.4f}")

        # Save PCA coordinates for key views
        if view_name in ("full", "ch3"):
            pca_df = pd.DataFrame(X_pca[:, :min(10, n_components)])
            pca_df.columns = [f"PC{i+1}" for i in range(pca_df.shape[1])]
            pca_df["label"] = all_labels
            pca_df.to_csv(out_dir / f"pca_coordinates_{view_name}.csv", index=False)

        # Top PC channel contribution (full view only)
        if view_name == "full":
            print("\n  Top PC loadings — channel importance:")
            for pc_idx in range(min(5, n_components)):
                component = pca.components_[pc_idx]
                reshaped = component.reshape(C, H, W)
                ch_importance = np.mean(np.abs(reshaped), axis=(1, 2))
                ch_str = ", ".join([f"Ch{i}: {v:.4f}" for i, v in enumerate(ch_importance)])
                print(f"    PC{pc_idx+1}: {ch_str}")

    # === Summary: Compare channels ===
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "results.csv", index=False)

    if not results_df.empty:
        print(f"\n{'='*50}")
        print("CHANNEL COMPARISON — Silhouette Scores")
        print(f"{'='*50}")

        # Plot: silhouette vs dims for each channel view
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for vname in ["full", "ch0", "ch1", "ch2", "ch3"]:
            vdf = results_df[results_df["view"] == vname]
            if not vdf.empty:
                ax.plot(vdf["n_dims"], vdf["silhouette"],
                        marker="o", label=view_labels[vname], linewidth=2)
        ax.set_xlabel("Number of PCA Dimensions", fontsize=11)
        ax.set_ylabel("Silhouette Score", fontsize=11)
        ax.set_title("Identity Clustering by Channel View", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / "channel_comparison_silhouette.png", dpi=150)
        plt.close()

        # Summary table
        pivot = results_df.pivot(index="n_dims", columns="view", values="silhouette")
        print(pivot.to_string(float_format="%.4f"))

        summary = results_df.groupby("view").agg(
            mean_silhouette=("silhouette", "mean"),
            max_silhouette=("silhouette", "max"),
        ).reset_index()
        summary["view_label"] = summary["view"].map(view_labels)
        summary.to_csv(out_dir / "summary.csv", index=False)
        print(f"\nSummary:\n{summary.to_string(index=False)}")

    print(f"\nExperiment 7 complete. Results saved to {out_dir}")
    return out_dir
