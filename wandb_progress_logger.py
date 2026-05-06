#!/usr/bin/env python3

import argparse
import csv
import json
import time
from pathlib import Path

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Log aggregate PRS pipeline progress to Weights & Biases.")
    parser.add_argument("--progress-file", required=True, help="JSONL progress file written by the pipeline")
    parser.add_argument("--out-dir", required=True, help="Pipeline output directory to upload on completion")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--run-name", required=True, help="W&B run name")
    parser.add_argument("--entity", default="", help="Optional W&B entity")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval for progress file")
    return parser.parse_args()


def read_new_events(progress_path: Path, offset: int):
    if not progress_path.exists():
        return [], offset
    with progress_path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        lines = handle.readlines()
        offset = handle.tell()
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events, offset


def log_event(run, event_index: int, event: dict, chromosome_status: dict):
    stage = event.get("stage", "pipeline")
    event_name = event.get("event", "update")
    metrics = event.get("metrics") or {}
    details = event.get("details") or {}

    payload = {
        "progress/event_index": event_index,
        "progress/event_name": f"{stage}/{event_name}",
    }
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            payload[f"{stage}/{key}"] = value

    if event_name == "chromosome_completed":
        chromosome = details.get("chromosome")
        if chromosome:
            chromosome_status[chromosome] = 1
            chrom_table = wandb.Table(
                data=[[chrom, chromosome_status[chrom]] for chrom in sorted(chromosome_status)],
                columns=["chromosome", "completed"],
            )
            payload["shared_reference/chromosome_progress"] = wandb.plot.bar(
                chrom_table,
                "chromosome",
                "completed",
                title="Shared reference chromosomes completed",
            )

    run.log(payload)

    for key, value in details.items():
        if isinstance(value, (str, int, float)):
            run.summary[f"{stage}/{key}"] = value
    run.summary[f"{stage}/last_event"] = event_name
    run.summary[f"{stage}/status"] = event.get("status", "running")


def maybe_log_csv_table(run, path: Path, key: str, max_rows=None):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return
    header, data = rows[0], rows[1:]
    if max_rows is not None:
        data = data[:max_rows]
    run.log({key: wandb.Table(columns=header, data=data)})


def upload_outputs(run, out_dir: Path):
    summary_path = out_dir / "summary.md"
    if summary_path.exists():
        run.save(str(summary_path), base_path=str(out_dir))

    for plot_path in sorted((out_dir / "plots").glob("*.png")):
        run.log({f"plots/{plot_path.stem}": wandb.Image(str(plot_path))})

    maybe_log_csv_table(
        run,
        out_dir / "tables" / "association_continuous_models.csv",
        "tables/association_continuous_models",
    )
    maybe_log_csv_table(
        run,
        out_dir / "tables" / "association_tertile_models.csv",
        "tables/association_tertile_models",
    )
    maybe_log_csv_table(
        run,
        out_dir / "tables" / "association_tertile_summary.csv",
        "tables/association_tertile_summary",
    )
    maybe_log_csv_table(
        run,
        out_dir / "tables" / "phewas_results.csv",
        "tables/phewas_results_top200",
        max_rows=200,
    )

    artifact = wandb.Artifact("prs-midpoint-results", type="analysis-results")
    for path in out_dir.rglob("*"):
        if path.is_file():
            artifact.add_file(str(path), name=str(path.relative_to(out_dir)))
    run.log_artifact(artifact)


def main():
    args = parse_args()
    progress_path = Path(args.progress_file)
    out_dir = Path(args.out_dir)
    init_kwargs = {"project": args.project, "name": args.run_name, "config": {"output_dir": str(out_dir)}}
    if args.entity:
        init_kwargs["entity"] = args.entity

    run = wandb.init(**init_kwargs)
    offset = 0
    event_index = 0
    chromosome_status = {}

    while True:
        events, offset = read_new_events(progress_path, offset)
        for event in events:
            event_index += 1
            log_event(run, event_index, event, chromosome_status)
            if event.get("event") == "run_finished":
                upload_outputs(run, out_dir)
                run.summary["pipeline/status"] = "completed"
                run.finish()
                return
            if event.get("event") == "run_failed":
                run.summary["pipeline/status"] = "failed"
                run.finish(exit_code=1)
                return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
