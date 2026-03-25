"""Experiment 5: Channel-Based Identity Transplant.

Question: Can we transplant identity by swapping Ch 3 between two latents during denoising?

Hypothesis (informed by Exp 1-3, 6):
- Swapping Ch 3 (identity fingerprint) at step 7 during denoising should transfer
  discriminative identity from source to target while preserving target's scene.
- Swapping Ch 1 (style/texture) should NOT transfer identity (negative control).
- Swapping Ch 0 (foundation) should destroy the image (positive control).

Success criteria:
- Ch 3 swap: ArcFace similarity to source identity increases vs unswapped
- Ch 3 swap: scene similarity (SSIM) to target remains high
- Ch 1 swap: ArcFace similarity to source does NOT increase (control)
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from identity_analysis.pipeline import PipelineWrapper
from identity_analysis.plotting import plot_image_grid, plot_line_chart
from identity_analysis.scoring import ArcFaceScorer, compute_mse, compute_ssim
from identity_analysis.utils import get_output_dir


# Source identity (who we want to transplant FROM)
SOURCE_PROMPTS = [
    ("Brad Pitt", "portrait photo of Brad Pitt, detailed face, studio lighting, neutral background"),
    ("Morgan Freeman", "portrait photo of Morgan Freeman, detailed face, studio lighting, neutral background"),
]

# Target scenes (where we want to transplant TO — different people, different contexts)
TARGET_PROMPTS = [
    "portrait photo of a young woman, detailed face, outdoor park, natural lighting",
    "portrait photo of an old man with a beard, detailed face, studio lighting, dark background",
    "portrait photo of a teenage boy, detailed face, urban street, golden hour",
]


def _swap_channel_at_step(pipe, source_prompt, target_prompt, seed, channel, swap_step, num_steps):
    """Generate source and target, swap a specific channel at a specific denoising step.

    Returns dict with swapped_image, source_image, target_image, and all latents.
    """
    import torch

    # Generate source with step latents
    source_res = pipe.generate(
        source_prompt, seed,
        num_inference_steps=num_steps,
        save_step_latents=True,
        output_dir=None,
    )

    # Generate target with step latents
    target_res = pipe.generate(
        target_prompt, seed,
        num_inference_steps=num_steps,
        save_step_latents=True,
        output_dir=None,
    )

    # Get the latent at the swap step from both
    source_step_latent = source_res["step_latents"][swap_step]  # [1, C, H, W]
    target_step_latent = target_res["step_latents"][swap_step]

    # Perform channel swap: replace target's channel with source's channel
    swapped_latent = target_step_latent.copy()
    swapped_latent[0, channel] = source_step_latent[0, channel]

    # Decode the swapped latent to see the result
    swapped_image = pipe.decode_latent(swapped_latent)

    return {
        "swapped_image": swapped_image,
        "swapped_latent": swapped_latent,
        "source_image": source_res["image"],
        "target_image": target_res["image"],
        "source_final_latent": source_res["final_latent"],
        "target_final_latent": target_res["final_latent"],
    }


def run(
    model_type: str = "sdxl",
    n_seeds: int = 50,
    save_latents: bool = True,
    output_base: str = "outputs",
    num_steps: int = 20,
):
    """Run Experiment 5: Channel-Based Identity Transplant."""
    import torch; torch.cuda.empty_cache(); gc.collect()
    out_dir = get_output_dir("experiment_5", output_base)
    print(f"Experiment 5 output: {out_dir}")

    pipe = PipelineWrapper(model_type)
    scorer = ArcFaceScorer()

    seeds = list(range(min(n_seeds, 5)))
    channels_to_test = [0, 1, 2, 3]  # Test all channels to compare
    swap_steps = [num_steps // 4, num_steps // 2, (3 * num_steps) // 4]  # Early, mid, late

    results_rows = []

    for source_name, source_prompt in SOURCE_PROMPTS:
        print(f"\n{'='*60}")
        print(f"Source identity: {source_name}")

        for target_idx, target_prompt in enumerate(TARGET_PROMPTS):
            print(f"\n  Target {target_idx}: {target_prompt[:60]}...")

            for seed in tqdm(seeds, desc=f"  Seeds"):
                # Get source identity embedding from a clean generation
                source_res = pipe.generate(source_prompt, seed, num_inference_steps=num_steps, save_step_latents=False)
                source_emb = scorer.get_embedding(source_res["image"])

                target_res = pipe.generate(target_prompt, seed, num_inference_steps=num_steps, save_step_latents=False)
                target_emb = scorer.get_embedding(target_res["image"])

                # Baseline: ArcFace between unmodified source and target
                baseline_sim = None
                if source_emb is not None and target_emb is not None:
                    baseline_sim = float(np.dot(source_emb, target_emb))

                # Save sample images for first seed
                if seed == 0:
                    source_res["image"].save(out_dir / "plots" / f"{source_name}_target{target_idx}_source.png")
                    target_res["image"].save(out_dir / "plots" / f"{source_name}_target{target_idx}_target.png")

                del source_res, target_res
                gc.collect()

                # Test each channel swap at each step
                for channel in channels_to_test:
                    for swap_step in swap_steps:
                        try:
                            result = _swap_channel_at_step(
                                pipe, source_prompt, target_prompt,
                                seed, channel, swap_step, num_steps,
                            )

                            # ArcFace: does swapped image look more like source?
                            swap_emb = scorer.get_embedding(result["swapped_image"])
                            arcface_to_source = None
                            arcface_to_target = None
                            if swap_emb is not None:
                                if source_emb is not None:
                                    arcface_to_source = float(np.dot(source_emb, swap_emb))
                                if target_emb is not None:
                                    arcface_to_target = float(np.dot(target_emb, swap_emb))

                            # Scene preservation: SSIM between swapped and original target
                            ssim_to_target = compute_ssim(
                                result["target_image"], result["swapped_image"]
                            )

                            results_rows.append({
                                "source": source_name,
                                "target_idx": target_idx,
                                "seed": seed,
                                "channel": channel,
                                "swap_step": swap_step,
                                "swap_step_frac": swap_step / num_steps,
                                "arcface_to_source": arcface_to_source,
                                "arcface_to_target": arcface_to_target,
                                "baseline_arcface": baseline_sim,
                                "identity_transfer": (arcface_to_source - baseline_sim) if (arcface_to_source and baseline_sim) else None,
                                "ssim_to_target": ssim_to_target,
                            })

                            # Save key swap images
                            if seed == 0 and swap_step == num_steps // 2:
                                result["swapped_image"].save(
                                    out_dir / "plots" / f"{source_name}_target{target_idx}_swap_ch{channel}_step{swap_step}.png"
                                )

                            del result
                            gc.collect()
                            torch.cuda.empty_cache()

                        except Exception as e:
                            print(f"    Error: ch{channel} step{swap_step} seed{seed}: {e}")
                            results_rows.append({
                                "source": source_name, "target_idx": target_idx,
                                "seed": seed, "channel": channel,
                                "swap_step": swap_step, "swap_step_frac": swap_step / num_steps,
                                "arcface_to_source": None, "arcface_to_target": None,
                                "baseline_arcface": baseline_sim,
                                "identity_transfer": None, "ssim_to_target": None,
                            })

    # Analysis
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    if not df.empty and df["identity_transfer"].notna().any():
        print(f"\n{'='*60}")
        print("EXPERIMENT 5 RESULTS: Channel-Based Identity Transplant")
        print(f"{'='*60}")

        # Key metric: identity transfer per channel (averaged across steps, seeds, pairs)
        print("\n--- Identity Transfer by Channel (ArcFace gain over baseline) ---")
        for ch in sorted(df["channel"].unique()):
            ch_df = df[(df["channel"] == ch) & df["identity_transfer"].notna()]
            if not ch_df.empty:
                mean_transfer = ch_df["identity_transfer"].mean()
                std_transfer = ch_df["identity_transfer"].std()
                mean_ssim = ch_df["ssim_to_target"].mean()
                print(f"  Ch {ch}: identity_transfer = {mean_transfer:+.3f} ± {std_transfer:.3f}, "
                      f"scene_preservation (SSIM) = {mean_ssim:.3f}")

        # Identity transfer by swap step
        print("\n--- Identity Transfer by Swap Step (Ch 3 only) ---")
        ch3_df = df[(df["channel"] == 3) & df["identity_transfer"].notna()]
        for step in sorted(ch3_df["swap_step"].unique()):
            step_df = ch3_df[ch3_df["swap_step"] == step]
            print(f"  Step {step}/{num_steps}: transfer = {step_df['identity_transfer'].mean():+.3f} ± "
                  f"{step_df['identity_transfer'].std():.3f}")

        # Summary CSV
        summary = df[df["identity_transfer"].notna()].groupby(["channel", "swap_step"]).agg({
            "identity_transfer": ["mean", "std"],
            "ssim_to_target": ["mean"],
            "arcface_to_source": ["mean"],
        }).reset_index()
        summary.columns = ["channel", "swap_step", "transfer_mean", "transfer_std", "ssim_mean", "arcface_mean"]
        summary.to_csv(out_dir / "summary.csv", index=False)

        # Plot: identity transfer by channel
        channels = sorted(df["channel"].unique())
        ch_means = [df[(df["channel"] == c) & df["identity_transfer"].notna()]["identity_transfer"].mean() for c in channels]
        ch_stds = [df[(df["channel"] == c) & df["identity_transfer"].notna()]["identity_transfer"].std() for c in channels]

        plot_line_chart(
            np.array(channels),
            {"Identity Transfer (ArcFace gain)": np.array(ch_means)},
            out_dir / "plots" / "identity_transfer_by_channel.png",
            xlabel="Swapped Channel",
            ylabel="ArcFace Similarity Gain vs Baseline",
            title=f"Identity Transfer by Channel Swap ({model_type.upper()})",
            yerr_dict={"Identity Transfer (ArcFace gain)": np.array(ch_stds)},
        )

        # Plot: transfer vs swap step for each channel
        step_data = {}
        for ch in channels:
            ch_df = df[(df["channel"] == ch) & df["identity_transfer"].notna()]
            steps = sorted(ch_df["swap_step"].unique())
            means = [ch_df[ch_df["swap_step"] == s]["identity_transfer"].mean() for s in steps]
            step_data[f"Ch {ch}"] = np.array(means)

        if step_data:
            plot_line_chart(
                np.array(steps),
                step_data,
                out_dir / "plots" / "transfer_vs_swap_step.png",
                xlabel="Swap Step",
                ylabel="Identity Transfer (ArcFace gain)",
                title=f"Identity Transfer vs Swap Timing ({model_type.upper()})",
            )

        # Plot: identity transfer vs scene preservation (the tradeoff)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 7))
        for ch in channels:
            ch_df = df[(df["channel"] == ch) & df["identity_transfer"].notna() & df["ssim_to_target"].notna()]
            ax.scatter(ch_df["ssim_to_target"], ch_df["identity_transfer"],
                      label=f"Ch {ch}", alpha=0.6, s=40)
        ax.set_xlabel("Scene Preservation (SSIM to target)", fontsize=11)
        ax.set_ylabel("Identity Transfer (ArcFace gain)", fontsize=11)
        ax.set_title(f"Identity Transfer vs Scene Preservation Tradeoff ({model_type.upper()})", fontsize=13)
        ax.legend(fontsize=9)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / "transfer_vs_preservation.png", dpi=150)
        plt.close()

    print(f"\nExperiment 5 complete. Results saved to {out_dir}")
    pipe.cleanup()
    return out_dir
