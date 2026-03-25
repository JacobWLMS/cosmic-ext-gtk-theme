"""Experiment 7: PCA on Identity.

Question: Is identity a linear subspace in latent space?
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
    n_seeds: int = 50,
    save_latents: bool = True,
    output_base: str = "outputs",
    num_steps: int = 20,
):
    """Run Experiment 7: PCA on Identity."""
    import torch; torch.cuda.empty_cache(); gc.collect()
    out_dir = get_output_dir("experiment_7", output_base)
    print(f"Experiment 7 output: {out_dir}")

    pipe = PipelineWrapper(model_type)
    celeb_prompts = get_celebrity_prompts()
    n_prompts_per_celeb = 20

    all_latents = []
    all_labels = []

    for celeb, prompts in celeb_prompts.items():
        print(f"\nGenerating for: {celeb}")
        prompts_to_use = prompts[:n_prompts_per_celeb]

        for p_idx, prompt in enumerate(tqdm(prompts_to_use, desc=f"  {celeb}")):
            seed = p_idx
            res = pipe.generate(prompt, seed, save_step_latents=False)

            # Encode through VAE for clean latent
            encoded = pipe.encode_image(res["image"])
            all_latents.append(encoded.flatten())
            all_labels.append(celeb)

            if save_latents:
                np.save(
                    out_dir / "latents" / f"{celeb.replace(' ', '_')}_p{p_idx}.npy",
                    encoded,
                )

            del res, encoded
            gc.collect()

    pipe.cleanup()

    # PCA analysis
    X = np.array(all_latents)
    print(f"\nLatent matrix shape: {X.shape}")

    # Full PCA
    n_components = min(50, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Explained variance
    plot_line_chart(
        np.arange(1, n_components + 1),
        {
            "Individual": pca.explained_variance_ratio_,
            "Cumulative": np.cumsum(pca.explained_variance_ratio_),
        },
        out_dir / "plots" / "explained_variance.png",
        xlabel="Principal Component",
        ylabel="Explained Variance Ratio",
        title=f"PCA Explained Variance ({model_type.upper()})",
    )

    # 2D scatter
    plot_pca_scatter(
        X_pca[:, :2],
        all_labels,
        out_dir / "plots" / "pca_2d.png",
        title=f"PCA 2D: Identity Clustering ({model_type.upper()})",
    )

    # 3D scatter
    if n_components >= 3:
        plot_pca_scatter(
            X_pca[:, :3],
            all_labels,
            out_dir / "plots" / "pca_3d.png",
            title=f"PCA 3D: Identity Clustering ({model_type.upper()})",
        )

    # Silhouette score for clustering quality
    unique_labels = list(set(all_labels))
    label_ids = [unique_labels.index(l) for l in all_labels]

    results = {}
    for n_dim in [2, 3, 5, 10, 20]:
        if n_dim <= n_components:
            score = silhouette_score(X_pca[:, :n_dim], label_ids)
            results[n_dim] = score
            print(f"  Silhouette score (top {n_dim} PCs): {score:.4f}")

    plot_line_chart(
        np.array(list(results.keys())),
        {"Silhouette Score": np.array(list(results.values()))},
        out_dir / "plots" / "silhouette_vs_dims.png",
        xlabel="Number of PCA Dimensions",
        ylabel="Silhouette Score",
        title=f"Identity Clustering Quality ({model_type.upper()})",
    )

    # Save results
    results_df = pd.DataFrame({
        "n_dimensions": list(results.keys()),
        "silhouette_score": list(results.values()),
    })
    results_df.to_csv(out_dir / "results.csv", index=False)

    # Save PCA coordinates
    pca_df = pd.DataFrame(X_pca[:, :min(10, n_components)])
    pca_df.columns = [f"PC{i+1}" for i in range(pca_df.shape[1])]
    pca_df["label"] = all_labels
    pca_df.to_csv(out_dir / "pca_coordinates.csv", index=False)

    # Component analysis: which spatial regions contribute most to identity-separating PCs
    n_channels = 4 if model_type == "sdxl" else 16
    latent_h = 64  # 512/8
    latent_w = 64

    print("\nTop PC loadings analysis:")
    for pc_idx in range(min(3, n_components)):
        component = pca.components_[pc_idx]
        reshaped = component.reshape(n_channels, latent_h, latent_w)
        channel_importance = np.mean(np.abs(reshaped), axis=(1, 2))
        print(f"  PC{pc_idx+1} channel importance: {channel_importance}")

    print(f"\nExperiment 7 complete. Results saved to {out_dir}")
    return out_dir
