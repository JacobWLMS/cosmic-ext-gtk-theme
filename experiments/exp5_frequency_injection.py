"""Experiment 5: Channel-Based Identity Transplant with Re-Denoising.

Swaps a channel between source and target latents at a specific denoising step,
then CONTINUES denoising to produce a clean final image (not a garbled intermediate).
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


def _generate_with_channel_swap(pipe, source_prompt, target_prompt, seed, channel, swap_step, num_steps):
    """Generate target image but swap one channel from source at a specific step.

    Uses a callback to inject the source channel during denoising, then
    continues denoising normally — producing a clean final image.
    """
    import torch

    # First generate source with step latent capture to get the source channel values
    source_res = pipe.generate(
        source_prompt, seed,
        num_inference_steps=num_steps,
        save_step_latents=True,
        output_dir=None,
    )
    source_step_latents = source_res["step_latents"]
    source_image = source_res["image"]

    # Now generate target with a callback that swaps the channel at the right step
    swap_done = {"done": False}

    def swap_callback(pipe_obj, step_index, timestep, callback_kwargs):
        if step_index == swap_step and not swap_done["done"]:
            latents = callback_kwargs["latents"]
            # Get the source latent at this same step
            source_lat = torch.from_numpy(source_step_latents[step_index]).to(
                device=latents.device, dtype=latents.dtype
            )
            # Swap the channel
            latents[:, channel] = source_lat[:, channel]
            callback_kwargs["latents"] = latents
            swap_done["done"] = True
        return callback_kwargs

    negative_prompt = pipe.DEFAULT_NEGATIVE_PROMPT
    generator = torch.Generator(device="cpu").manual_seed(seed)

    kwargs = dict(
        prompt=target_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        height=1024,
        width=1024,
        num_inference_steps=num_steps,
        generator=generator,
        output_type="pil",
        callback_on_step_end=swap_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    result = pipe.pipe(**kwargs)
    swapped_image = result.images[0]

    del result
    torch.cuda.empty_cache()

    return {
        "swapped_image": swapped_image,
        "source_image": source_image,
    }


def run(
    model_type: str = "sdxl",
    n_seeds: int = 50,
    save_latents: bool = True,
    output_base: str = "outputs",
    num_steps: int = 20,
):
    """Run Experiment 5: Channel-Based Identity Transplant with Re-Denoising."""
    import torch; torch.cuda.empty_cache(); gc.collect()
    out_dir = get_output_dir("experiment_5", output_base)
    print(f"Experiment 5 output: {out_dir}")

    pipe = PipelineWrapper(model_type)
    scorer = ArcFaceScorer()

    seeds = list(range(min(n_seeds, 3)))
    channels_to_test = [0, 1, 2, 3]
    swap_steps = [num_steps // 4, num_steps // 2]  # Early + mid

    results_rows = []

    for source_name, source_prompt in SOURCE_PROMPTS:
        print(f"\n{'='*60}")
        print(f"Source identity: {source_name}")

        for target_idx, target_prompt in enumerate(TARGET_PROMPTS):
            print(f"\n  Target {target_idx}: {target_prompt[:60]}...")

            for seed in tqdm(seeds, desc=f"  Seeds"):
                # Generate clean unmodified versions for baseline
                source_clean = pipe.generate(source_prompt, seed, num_inference_steps=num_steps, save_step_latents=False)
                target_clean = pipe.generate(target_prompt, seed, num_inference_steps=num_steps, save_step_latents=False)

                source_emb = scorer.get_embedding(source_clean["image"])
                target_emb = scorer.get_embedding(target_clean["image"])

                baseline_sim = None
                if source_emb is not None and target_emb is not None:
                    baseline_sim = float(np.dot(source_emb, target_emb))

                if seed == 0:
                    source_clean["image"].save(out_dir / "plots" / f"{source_name}_target{target_idx}_source.png")
                    target_clean["image"].save(out_dir / "plots" / f"{source_name}_target{target_idx}_target.png")

                del source_clean
                gc.collect()

                # Test each channel swap
                for channel in channels_to_test:
                    for swap_step in swap_steps:
                        try:
                            result = _generate_with_channel_swap(
                                pipe, source_prompt, target_prompt,
                                seed, channel, swap_step, num_steps,
                            )

                            swap_emb = scorer.get_embedding(result["swapped_image"])
                            arcface_to_source = None
                            arcface_to_target = None
                            if swap_emb is not None:
                                if source_emb is not None:
                                    arcface_to_source = float(np.dot(source_emb, swap_emb))
                                if target_emb is not None:
                                    arcface_to_target = float(np.dot(target_emb, swap_emb))

                            ssim_to_target = compute_ssim(target_clean["image"], result["swapped_image"])

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
                                result["swapped_image"].save(
                                    out_dir / "plots" / f"{source_name}_target{target_idx}_swap_ch{channel}_step{swap_step}.png"
                                )

                            del result
                            gc.collect()
                            torch.cuda.empty_cache()

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

                del target_clean
                gc.collect()
                torch.cuda.empty_cache()

    # Analysis
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    if not df.empty and df["identity_transfer"].notna().any():
        print(f"\n{'='*60}")
        print("EXPERIMENT 5 RESULTS: Channel-Based Identity Transplant")
        print(f"{'='*60}")

        print("\n--- Identity Transfer by Channel ---")
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

        # Plots
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

        step_data = {}
        for ch in channels:
            ch_df = df[(df["channel"] == ch) & df["identity_transfer"].notna()]
            steps = sorted(ch_df["swap_step"].unique())
            means = [ch_df[ch_df["swap_step"] == s]["identity_transfer"].mean() for s in steps]
            step_data[f"Ch {ch}"] = np.array(means)

        if step_data and steps:
            plot_line_chart(
                np.array(steps),
                step_data,
                out_dir / "plots" / "transfer_vs_swap_step.png",
                xlabel="Swap Step",
                ylabel="Identity Transfer (ArcFace gain)",
                title=f"Identity Transfer vs Swap Timing ({model_type.upper()})",
            )

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
