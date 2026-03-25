"""Experiment 6: Channel Importance Analysis.

Question: Do specific VAE channels carry more identity information than others?
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from identity_analysis.pipeline import PipelineWrapper
from identity_analysis.plotting import plot_channel_importance, plot_difference_heatmaps
from identity_analysis.utils import (
    detect_face_bbox,
    get_output_dir,
    get_prompt_pairs,
    pixel_bbox_to_latent_bbox,
)


def run(
    model_type: str = "sdxl",
    n_seeds: int = 50,
    save_latents: bool = True,
    output_base: str = "outputs",
    num_steps: int = 20,
):
    """Run Experiment 6: Channel Importance Analysis."""
    import torch; torch.cuda.empty_cache(); gc.collect()
    out_dir = get_output_dir("experiment_6", output_base)
    print(f"Experiment 6 output: {out_dir}")

    pipe = PipelineWrapper(model_type)
    prompt_pairs = get_prompt_pairs()
    seeds = list(range(n_seeds))

    results_rows = []
    all_face_diffs = []
    all_non_face_diffs = []

    for pair_idx, (prompt_a, prompt_b) in enumerate(prompt_pairs):
        print(f"\nPrompt pair {pair_idx + 1}/{len(prompt_pairs)}:")
        print(f"  A: {prompt_a[:60]}...")
        print(f"  B: {prompt_b[:60]}...")

        for seed in tqdm(seeds, desc=f"  Seeds (pair {pair_idx + 1})"):
            res_a = pipe.generate(prompt_a, seed, save_step_latents=False)
            res_b = pipe.generate(prompt_b, seed, save_step_latents=False)

            lat_a = res_a["final_latent"]
            lat_b = res_b["final_latent"]
            img_a = res_a["image"]
            img_b = res_b["image"]

            # Detect faces in both images
            bbox_a = detect_face_bbox(img_a)
            bbox_b = detect_face_bbox(img_b)

            # Per-channel absolute difference
            diff = np.abs(lat_a - lat_b)
            if diff.ndim == 4:
                diff = diff[0]

            C, H, W = diff.shape

            if bbox_a and bbox_b:
                # Use union of both face bboxes for the face region
                lat_bbox_a = pixel_bbox_to_latent_bbox(bbox_a, pipe.vae_scale_factor)
                lat_bbox_b = pixel_bbox_to_latent_bbox(bbox_b, pipe.vae_scale_factor)

                y1 = min(lat_bbox_a[0], lat_bbox_b[0])
                x1 = min(lat_bbox_a[1], lat_bbox_b[1])
                y2 = max(lat_bbox_a[2], lat_bbox_b[2])
                x2 = max(lat_bbox_a[3], lat_bbox_b[3])

                # Clamp
                y1, x1 = max(0, y1), max(0, x1)
                y2, x2 = min(H, y2), min(W, x2)

                face_mask = np.zeros((H, W), dtype=bool)
                face_mask[y1:y2, x1:x2] = True

                face_diffs = np.array([
                    np.mean(diff[c][face_mask]) for c in range(C)
                ])
                non_face_diffs = np.array([
                    np.mean(diff[c][~face_mask]) for c in range(C)
                ])

                all_face_diffs.append(face_diffs)
                all_non_face_diffs.append(non_face_diffs)

                for c in range(C):
                    results_rows.append({
                        "pair_idx": pair_idx,
                        "seed": seed,
                        "channel": c,
                        "face_region_diff": float(face_diffs[c]),
                        "non_face_region_diff": float(non_face_diffs[c]),
                        "ratio": float(face_diffs[c] / (non_face_diffs[c] + 1e-10)),
                        "face_detected": True,
                    })
            else:
                # No face detected - just record overall diff
                for c in range(C):
                    results_rows.append({
                        "pair_idx": pair_idx,
                        "seed": seed,
                        "channel": c,
                        "face_region_diff": float(np.mean(diff[c])),
                        "non_face_region_diff": float(np.mean(diff[c])),
                        "ratio": 1.0,
                        "face_detected": False,
                    })

            # Save example difference heatmaps for first seed
            if seed == 0:
                plot_difference_heatmaps(
                    diff,
                    out_dir / "plots" / f"pair{pair_idx}_diff_heatmap.png",
                    title=f"Pair {pair_idx}: Per-Channel Absolute Difference",
                )

            if save_latents and seed == 0:
                np.save(out_dir / "latents" / f"pair{pair_idx}_diff.npy", diff)

            del res_a, res_b, lat_a, lat_b, diff
            gc.collect()

    # Aggregate analysis
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    if all_face_diffs:
        avg_face = np.mean(all_face_diffs, axis=0)
        avg_non_face = np.mean(all_non_face_diffs, axis=0)

        plot_channel_importance(
            avg_face,
            avg_non_face,
            out_dir / "plots" / "channel_importance.png",
            title=f"Channel Importance: Face vs Non-Face ({model_type.upper()})",
        )

        # Rank channels
        ratio = avg_face / (avg_non_face + 1e-10)
        ranking = np.argsort(ratio)[::-1]
        print(f"\nChannel ranking by face/non-face ratio (most identity-carrying first):")
        for rank, ch in enumerate(ranking):
            print(f"  #{rank + 1}: Channel {ch} (ratio: {ratio[ch]:.3f}, "
                  f"face: {avg_face[ch]:.4f}, non-face: {avg_non_face[ch]:.4f})")

        ranking_df = pd.DataFrame({
            "channel": range(len(avg_face)),
            "face_diff": avg_face,
            "non_face_diff": avg_non_face,
            "ratio": ratio,
            "rank": np.argsort(np.argsort(ratio)[::-1]) + 1,
        })
        ranking_df.to_csv(out_dir / "channel_ranking.csv", index=False)

    print(f"\nExperiment 6 complete. Results saved to {out_dir}")
    pipe.cleanup()
    return out_dir
