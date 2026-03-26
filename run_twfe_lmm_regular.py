#!/usr/bin/env python3
"""
Run the regular-data DST TWFE + LMM pipeline end-to-end.

Pipeline order:
1) 03_3_DLST_Adjusted_twfe.py
2) 04_LMM_Regression.r
3) 05_Precompute_Predictions.r
4) 05_LMM_Results.r

Outputs are written under the configured results directory (default: ./results).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"\n[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 03_3_DLST_Adjusted_twfe.py followed by the LMM pipeline "
            "on regular data and write outputs under results/."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base output directory (default: results)",
    )
    parser.add_argument(
        "--twfe-input",
        default="processed_data/ready_for_analysis.parquet",
        help="Expected TWFE input parquet path (used for validation)",
    )
    parser.add_argument(
        "--lmm-input",
        default="processed_data/LMM_analysis.parquet",
        help="Input parquet for 04_LMM_Regression.r (regular data)",
    )
    parser.add_argument(
        "--rscript-bin",
        default=os.getenv("RSCRIPT_BIN", "Rscript"),
        help="Rscript executable to use (default: Rscript)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    twfe_script = repo_root / "03_3_DLST_Adjusted_twfe.py"
    lmm_script = repo_root / "04_LMM_Regression.r"
    precompute_script = repo_root / "05_Precompute_Predictions.r"
    results_script = repo_root / "05_LMM_Results.r"

    twfe_input = (repo_root / args.twfe_input).resolve()
    lmm_input = (repo_root / args.lmm_input).resolve()
    results_dir = (repo_root / args.results_dir).resolve()

    required_files = [twfe_script, lmm_script, precompute_script, results_script, twfe_input, lmm_input]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        print("ERROR: Missing required files:")
        for item in missing:
            print(f"  - {item}")
        return 1

    model_dir = results_dir / "models_04_regular"
    summary_dir = results_dir / "model_summaries_04_regular"
    plot_dir = results_dir / "plots"
    aic_report = results_dir / "model_comparison_aic_04_regular.md"

    if args.dry_run:
        print("Dry run configuration:")
        print(f"  repo_root: {repo_root}")
        print(f"  twfe_input: {twfe_input}")
        print(f"  lmm_input: {lmm_input}")
        print(f"  results_dir: {results_dir}")
        print(f"  model_dir: {model_dir}")
        print(f"  summary_dir: {summary_dir}")
        print(f"  plot_dir: {plot_dir}")
        print(f"  aic_report: {aic_report}")
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    # Ensure regular/full mode behavior.
    env["TEST_MODE"] = "0"
    env["TEST_MODE_04"] = "false"

    # Route LMM/LLM outputs into ./results (regular run).
    env["INPUT_PARQUET_04"] = str(lmm_input)
    env["MODEL_DIR_04"] = str(model_dir)
    env["SUMMARY_DIR_04"] = str(summary_dir)
    env["AIC_REPORT_FILE_04"] = str(aic_report)
    env["OUTPUT_DIR_05"] = str(results_dir)
    env["PLOT_DIR"] = str(plot_dir)

    print("Starting regular-data pipeline...")
    run_cmd([sys.executable, str(twfe_script)], cwd=repo_root, env=env)
    run_cmd([args.rscript_bin, str(lmm_script)], cwd=repo_root, env=env)
    run_cmd([args.rscript_bin, str(precompute_script)], cwd=repo_root, env=env)
    run_cmd([args.rscript_bin, str(results_script)], cwd=repo_root, env=env)

    print("\nPipeline completed successfully.")
    print(f"Results written under: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
