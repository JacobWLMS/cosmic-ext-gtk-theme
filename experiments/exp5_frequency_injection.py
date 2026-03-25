"""Experiment 5: Channel-Based Identity Transplant.

Question: Can we transplant identity by swapping Ch 3 between two latents during denoising?

Optimized: generates source+target ONCE per seed, caches step latents,
then performs all channel/step swaps from cache (no redundant generation).
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from identity_analysis.pipeline import PipelineWrapper
from identity_analysis.plotting import plot_line_chart
from identity_analysis.scoring import ArcFaceScorer, compute_ssim
from identity_analysis.utils import get_output_dir


SOURCE_PROMPTS = [
    ("Brad Pitt", "portrait photo of Brad Pitt, detailed face, studio lighting, neutral background"),
    ("Morgan Freeman", "portrait photo of Morgan Freeman, detailed face, studio lighting, neutral background"),
]

TARGET_PROMPTS = [
    "portrait photo of a young woman, detailed face, outdoor park, natural lighting",
    "portrait photo of an old man with a beard, detailed face, studio lighting, dark background",
    "portrait photo of a teenage boy, detailed face, urban street, golden hour",
]


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

    seeds = list(range(min(n_seeds, 3)))
    channels_to_test = [0, 1, 2, 3]
    swap_steps = [num_steps // 4, num_steps // 2]  # Early + mid only

    results_rows = []

    for source_name, source_prompt in SOURCE_PROMPTS:
        print(f"\n{'='*60}")
        print(f"Source identity: {source_name}")

        for target_idx, target_prompt in enumerate(TARGET_PROMPTS):
            print(f"\n  Target {target_idx}: {target_prompt[:60]}...")

            for seed in tqdm(seeds, desc=f"  Seeds"):
                # === GENERATE ONCE, CACHE STEP LATENTS ===
                source_res = pipe.generate(
                    source_prompt, seed,
                    num_inference_steps=num_steps,
                    save_step_latents=True,
                    output_dir=None,
                )
                target_res = pipe.generate(
                    target_prompt, seed,
                    num_inference_steps=num_steps,
                    save_step_latents=True,
                    output_dir=None,
                )

                source_emb = scorer.get_embedding(source_res["image"])
                target_emb = scorer.get_embedding(target_res["image"])

                baseline_sim = None
                if source_emb is not None and target_emb is not None:
                    baseline_sim = float(np.dot(source_emb, target_emb))

                if seed == 0:
                    source_res["image"].save(out_dir / "plots" / f"{source_name}_target{target_idx}_source.png")
                    target_res["image"].save(out_dir / "plots" / f"{source_name}_target{target_idx}_target.png")

                # === SWAP FROM CACHE — no regeneration ===
                for channel in channels_to_test:
                    for swap_step in swap_steps:
                        try:
                            source_step_lat = source_res["step_latents"][swap_step]
                            target_step_lat = target_res["step_latents"][swap_step]

                            swapped = target_step_lat.copy()
                            swapped[0, channel] = source_step_lat[0, channel]

                            swapped_image = pipe.decode_latent(swapped)

                            swap_emb = scorer.get_embedding(swapped_image)
                            arcface_to_source = None
                            arcface_to_target = None
                            if swap_emb is not None:
                                if source_emb is not None:
                                    arcface_to_source = float(np.dot(source_emb, swap_emb))
                                if target_emb is not None:
                                    arcface_to_target = float(np.dot(target_emb, swap_emb))

                            ssim_to_target = compute_ssim(target_res["image"], swapped_image)

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
                                "identity_transfer": (arcface_to_source - baseline_sim) if (arcface_to_source is not None and baseline_sim is not None) else None,
                                "ssim_to_target": ssim_to_target,
                            })

                            if seed == 0 and swap_step == num_steps // 2:
                                swapped_image.save(
                                    out_dir / "plots" / f"{source_name}_target{target_idx}_swap_ch{channel}_step{swap_step}.png"
                                )

                            del swapped_image, swapped
                        except Exception as e:
                            print(f"    Error: ch{channel} step{swap_step}: {e}")
                            results_rows.append({
                                "source": source_name, "target_idx": target_idx,
                                "seed": seed, "channel": channel,
                                "swap_step": swap_step, "swap_step_frac": swap_step / num_steps,
                                "arcface_to_source": None, "arcface_to_target": None,
                                "baseline_arcface": baseline_sim,
                                "identity_transfer": None, "ssim_to_target": None,
                            })

                del source_res, target_res
                gc.collect()
                torch.cuda.empty_cache()

    # Analysis
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    if not df.empty and df["identity_transfer"].notna().any():
        print(f"\n{'='*60}")
        print("EXPERIMENT 5 RESULTS: Channel-Based Identity Transplant")
        print(f"{'='*60}")

        print("\n--- Identity Transfer by Channel (ArcFace gain over baseline) ---")
        for ch in sorted(df["channel"].unique()):
            ch_df = df[(df["channel"] == ch) & df["identity_transfer"].notna()]
            if not ch_df.empty:
                print(f"  Ch {ch}: transfer = {ch_df['identity_transfer'].mean():+.3f} ± {ch_df['identity_transfer'].std():.3f}, "
                      f"SSIM = {ch_df['ssim_to_target'].mean():.3f}")

        print("\n--- Identity Transfer by Swap Step (Ch 3 only) ---")
        ch3_df = df[(df["channel"] == 3) & df["identity_transfer"].notna()]
        for step in sorted(ch3_df["swap_step"].unique()):
            step_df = ch3_df[ch3_df["swap_step"] == step]
            print(f"  Step {step}/{num_steps}: transfer = {step_df['identity_transfer'].mean():+.3f}")

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

        # Plot: identity transfer vs scene preservation tradeoff
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 7))
        for ch in channels:
            ch_df = df[(df["channel"] == ch) & df["identity_transfer"].notna() & df["ssim_to_target"].notna()]
            ax.scatter(ch_df["ssim_to_target"], ch_df["identity_transfer"],
                      label=f"Ch {ch}", alpha=0.6, s=40)
        ax.set_xlabel("Scene Preservation (SSIM to target)", fontsize=11)
        ax.set_ylabel("Identity Transfer (ArcFace gain)", fontsize=11)
        ax.set_title(f"Identity Transfer vs Scene Preservation ({model_type.upper()})", fontsize=13)
        ax.legend(fontsize=9)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / "transfer_vs_preservation.png", dpi=150)
        plt.close()

    print(f"\nExperiment 5 complete. Results saved to {out_dir}")
    pipe.cleanup()
    return out_dir
