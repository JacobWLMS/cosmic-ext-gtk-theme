"""Experiment 6: Channel Importance Analysis.

Question: Do specific VAE channels carry more identity information than others?

Key focus (informed by Exp 1-3):
- Face vs non-face region analysis per channel (spatial localization)
- Channel zeroing: zero each channel, decode, measure ArcFace similarity drop
- Validates the dual-channel model: Ch 2 (magnitude) vs Ch 3 (fingerprint)
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from identity_analysis.pipeline import PipelineWrapper
from identity_analysis.plotting import (
    plot_channel_importance,
    plot_difference_heatmaps,
    plot_line_chart,
)
from identity_analysis.scoring import ArcFaceScorer
from identity_analysis.utils import (
    detect_face_bbox,
    get_output_dir,
    get_prompt_pairs,
    pixel_bbox_to_latent_bbox,
)


CELEB_PROMPTS = [
    ("Brad Pitt", "portrait photo of Brad Pitt, detailed face, studio lighting"),
    ("Taylor Swift", "portrait photo of Taylor Swift, detailed face, studio lighting"),
    ("Morgan Freeman", "portrait photo of Morgan Freeman, detailed face, studio lighting"),
]


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
    scorer = ArcFaceScorer()
    prompt_pairs = get_prompt_pairs()[:6]  # Use first 6 pairs for face/non-face analysis
    seeds = list(range(min(n_seeds, 5)))

    # ====== PART 1: Face vs Non-Face Region Analysis ======
    print("\n=== Part 1: Face vs Non-Face Region Differences ===")
    results_rows = []
    all_face_diffs = []
    all_non_face_diffs = []
    face_detect_count = 0
    total_count = 0

    for pair_idx, (prompt_a, prompt_b) in enumerate(prompt_pairs):
        print(f"\nPair {pair_idx + 1}/{len(prompt_pairs)}: {prompt_a[:50]}...")

        for seed in tqdm(seeds, desc=f"  Seeds"):
            res_a, res_b = pipe.generate_pair(
                prompt_a, prompt_b, seed,
                num_inference_steps=num_steps,
                decode_images=True,
            )

            lat_a = res_a["final_latent"]
            lat_b = res_b["final_latent"]
            img_a = res_a["image"]
            img_b = res_b["image"]

            bbox_a = detect_face_bbox(img_a)
            bbox_b = detect_face_bbox(img_b)
            total_count += 1

            diff = np.abs(lat_a - lat_b)
            if diff.ndim == 4:
                diff = diff[0]
            C, H, W = diff.shape

            if bbox_a and bbox_b:
                face_detect_count += 1
                lat_bbox_a = pixel_bbox_to_latent_bbox(bbox_a, pipe.vae_scale_factor)
                lat_bbox_b = pixel_bbox_to_latent_bbox(bbox_b, pipe.vae_scale_factor)

                y1 = min(lat_bbox_a[0], lat_bbox_b[0])
                x1 = min(lat_bbox_a[1], lat_bbox_b[1])
                y2 = max(lat_bbox_a[2], lat_bbox_b[2])
                x2 = max(lat_bbox_a[3], lat_bbox_b[3])
                y1, x1 = max(0, y1), max(0, x1)
                y2, x2 = min(H, y2), min(W, x2)

                face_mask = np.zeros((H, W), dtype=bool)
                face_mask[y1:y2, x1:x2] = True

                face_diffs = np.array([np.mean(diff[c][face_mask]) for c in range(C)])
                non_face_diffs = np.array([np.mean(diff[c][~face_mask]) for c in range(C)])
                all_face_diffs.append(face_diffs)
                all_non_face_diffs.append(non_face_diffs)

                for c in range(C):
                    results_rows.append({
                        "pair_idx": pair_idx, "seed": seed, "channel": c,
                        "face_diff": float(face_diffs[c]),
                        "non_face_diff": float(non_face_diffs[c]),
                        "face_non_face_ratio": float(face_diffs[c] / (non_face_diffs[c] + 1e-10)),
                        "face_detected": True,
                    })

            if seed == 0:
                plot_difference_heatmaps(
                    diff, out_dir / "plots" / f"pair{pair_idx}_diff_heatmap.png",
                    title=f"Pair {pair_idx}: Per-Channel |latent_A - latent_B|",
                )

            del res_a, res_b, diff
            gc.collect()

    print(f"\n  Face detection rate: {face_detect_count}/{total_count} ({100*face_detect_count/max(1,total_count):.0f}%)")

    # Face vs non-face plots
    if all_face_diffs:
        avg_face = np.mean(all_face_diffs, axis=0)
        avg_non_face = np.mean(all_non_face_diffs, axis=0)

        plot_channel_importance(
            avg_face, avg_non_face,
            out_dir / "plots" / "face_vs_nonface_importance.png",
            title=f"Channel Importance: Face vs Non-Face Region ({model_type.upper()})",
        )

        ratio = avg_face / (avg_non_face + 1e-10)
        print(f"\n  Face/Non-Face ratio per channel:")
        for c in range(len(ratio)):
            print(f"    Ch {c}: {ratio[c]:.3f} (face={avg_face[c]:.4f}, non-face={avg_non_face[c]:.4f})")

    # ====== PART 2: Channel Zeroing (ArcFace Impact) ======
    print("\n=== Part 2: Channel Zeroing — ArcFace Identity Impact ===")
    zeroing_rows = []

    for celeb_name, prompt in tqdm(CELEB_PROMPTS, desc="  Celebrities"):
        for seed in range(3):  # 3 seeds per celebrity
            res = pipe.generate(prompt, seed, num_inference_steps=num_steps, save_step_latents=False)
            original_latent = res["final_latent"]  # [1, C, H, W]
            original_image = res["image"]
            ref_emb = scorer.get_embedding(original_image)

            if ref_emb is None:
                continue

            C = original_latent.shape[1]

            for ch in range(C):
                # Zero out this channel
                zeroed = original_latent.copy()
                zeroed[0, ch] = 0.0
                zeroed_image = pipe.decode_latent(zeroed)

                zeroed_emb = scorer.get_embedding(zeroed_image)
                if zeroed_emb is not None:
                    sim = float(np.dot(ref_emb, zeroed_emb))
                else:
                    sim = 0.0  # No face = identity destroyed

                zeroing_rows.append({
                    "celebrity": celeb_name,
                    "seed": seed,
                    "zeroed_channel": ch,
                    "arcface_similarity": sim,
                    "identity_preserved": sim > 0.3,
                })

                if seed == 0:
                    zeroed_image.save(
                        out_dir / "plots" / f"{celeb_name.replace(' ', '_')}_zeroed_ch{ch}.png"
                    )

            # Also save original
            if seed == 0:
                original_image.save(
                    out_dir / "plots" / f"{celeb_name.replace(' ', '_')}_original.png"
                )

            del res, original_latent, original_image
            gc.collect()

    # Zeroing analysis
    zero_df = pd.DataFrame(zeroing_rows)
    if not zero_df.empty:
        print(f"\n  ArcFace similarity after zeroing each channel (avg across celebs/seeds):")
        for ch in sorted(zero_df['zeroed_channel'].unique()):
            ch_df = zero_df[zero_df['zeroed_channel'] == ch]
            mean_sim = ch_df['arcface_similarity'].mean()
            preserved = ch_df['identity_preserved'].mean()
            print(f"    Ch {ch} zeroed: ArcFace sim = {mean_sim:.3f} (identity preserved: {100*preserved:.0f}%)")

        # Plot: ArcFace similarity when each channel is zeroed
        ch_labels = [f"Ch {c}" for c in sorted(zero_df['zeroed_channel'].unique())]
        ch_sims = [zero_df[zero_df['zeroed_channel'] == c]['arcface_similarity'].mean()
                    for c in sorted(zero_df['zeroed_channel'].unique())]

        plot_line_chart(
            np.arange(len(ch_labels)),
            {"ArcFace Similarity (after zeroing)": np.array(ch_sims)},
            out_dir / "plots" / "zeroing_arcface_impact.png",
            xlabel="Zeroed Channel",
            ylabel="ArcFace Cosine Similarity to Original",
            title=f"Identity Impact of Zeroing Each Channel ({model_type.upper()})",
        )

    # Save all results
    pd.DataFrame(results_rows).to_csv(out_dir / "face_region_results.csv", index=False)
    zero_df.to_csv(out_dir / "zeroing_results.csv", index=False)

    if all_face_diffs:
        pd.DataFrame({
            "channel": range(len(avg_face)),
            "face_diff": avg_face,
            "non_face_diff": avg_non_face,
            "face_non_face_ratio": ratio,
        }).to_csv(out_dir / "channel_ranking.csv", index=False)

    print(f"\nExperiment 6 complete. Results saved to {out_dir}")
    pipe.cleanup()
    return out_dir
