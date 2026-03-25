#!/usr/bin/env python3
"""Main entry point for running identity signal analysis experiments."""

import argparse
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Identity Signal Analysis in Diffusion Model Latent Spaces"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=int,
        nargs="+",
        required=True,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Experiment number(s) to run",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="sdxl",
        choices=["sdxl", "lumina2"],
        help="Model to use (default: sdxl)",
    )
    parser.add_argument(
        "--seeds", "-s",
        type=int,
        default=50,
        help="Number of seeds to use (default: 50)",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: use 5 seeds instead of the specified number",
    )
    parser.add_argument(
        "--no-save-latents",
        action="store_true",
        help="Don't save raw latent tensors to disk",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs",
        help="Base output directory (default: outputs)",
    )

    args = parser.parse_args()

    n_seeds = 5 if args.quick else args.seeds
    save_latents = not args.no_save_latents

    experiment_modules = {
        1: ("experiments.exp1_paired_frequency", "Paired Latent Frequency Analysis"),
        2: ("experiments.exp2_within_identity", "Within-Identity Latent Invariance"),
        3: ("experiments.exp3_identity_emergence", "Step-by-Step Identity Emergence"),
        4: ("experiments.exp4_reference_correlation", "Timestep-Matched Reference Correlation"),
        5: ("experiments.exp5_frequency_injection", "Naive Frequency Injection Test"),
        6: ("experiments.exp6_channel_importance", "Channel Importance Analysis"),
        7: ("experiments.exp7_pca_identity", "PCA on Identity"),
    }

    print("=" * 70)
    print("Identity Signal Analysis in Diffusion Model Latent Spaces")
    print(f"Model: {args.model.upper()}")
    print(f"Seeds: {n_seeds}")
    print(f"Experiments: {args.experiment}")
    print(f"Save latents: {save_latents}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    completed = []
    for exp_num in args.experiment:
        module_name, exp_name = experiment_modules[exp_num]
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT {exp_num}: {exp_name}")
        print(f"{'=' * 70}")

        try:
            module = __import__(module_name, fromlist=["run"])
            out_dir = module.run(
                model_type=args.model,
                n_seeds=n_seeds,
                save_latents=save_latents,
                output_base=args.output_dir,
            )
            completed.append((exp_num, out_dir))
        except Exception as e:
            print(f"\nERROR in Experiment {exp_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary
    if len(completed) > 1:
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        _generate_summary(completed, args.model, args.output_dir)

    print("\nAll experiments complete.")
    for exp_num, out_dir in completed:
        print(f"  Experiment {exp_num}: {out_dir}")


def _generate_summary(completed: list, model_type: str, output_base: str):
    """Generate a summary of completed experiments."""
    summary_dir = Path(output_base) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"Identity Signal Analysis - Summary",
        f"Model: {model_type.upper()}",
        f"Date: {datetime.now().isoformat()}",
        f"Experiments completed: {len(completed)}",
        "",
    ]

    for exp_num, out_dir in completed:
        lines.append(f"Experiment {exp_num}: {out_dir}")
        results_csv = Path(out_dir) / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            lines.append(f"  Rows: {len(df)}")
            lines.append(f"  Columns: {list(df.columns)}")
        lines.append("")

    summary_path = summary_dir / "key_findings.txt"
    summary_path.write_text("\n".join(lines))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
