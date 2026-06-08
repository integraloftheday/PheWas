#!/usr/bin/env python3
"""
Build an aggregate patient-number flow chart for the Fitbit sleep analysis data.

Outputs:
  - results/patient_flow/patient_flow_counts.csv
  - results/patient_flow/patient_flow_chart.png
  - results/patient_flow/patient_flow_chart.pdf

Optional BigQuery source counts:
  RUN_BIGQUERY_COUNTS=true python 06_Patient_Flow_Chart.py

This script reports aggregate counts only. Counts below 20 are masked in figure
labels to avoid presenting small-cell participant counts.
"""

from __future__ import annotations

import csv
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/matplotlib-cache")))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/private/tmp/font-cache")))

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.patches import Rectangle


OUTPUT_DIR = Path(os.getenv("PATIENT_FLOW_OUTPUT_DIR", "results/patient_flow"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FlowStage:
    source: str
    stage: str
    filter_step: str
    rows: int | None
    persons: int | None
    status: str
    path: str | None = None


def mask_small_n(value: int | None) -> str:
    if value is None or pd.isna(value):
        return "not available"
    value = int(value)
    if value < 20:
        return "<20"
    return f"{value:,}"


def parquet_counts(path: str) -> tuple[int | None, int | None, str]:
    file_path = Path(path)
    if not file_path.exists():
        return None, None, "missing"

    try:
        parquet_file = pq.ParquetFile(file_path)
        rows = int(parquet_file.metadata.num_rows)
        schema_names = parquet_file.schema_arrow.names
        if "person_id" not in schema_names:
            return rows, None, "missing person_id"

        table = pq.read_table(file_path, columns=["person_id"])
        persons = len(table.column("person_id").unique())
        return rows, int(persons), "ok"
    except Exception as exc:  # noqa: BLE001 - keep script diagnostic simple.
        return None, None, f"error: {exc}"


def pgrs_patient_list_count() -> tuple[int | None, str | None]:
    shared_dir = Path("processed_data/PGRS/shared")
    candidates = [
        shared_dir / "patient_list_all.txt",
        shared_dir / "patient_list_eur.txt",
    ]
    if shared_dir.exists():
        candidates.extend(sorted(shared_dir.rglob("patient_list_*.txt")))

    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return sum(1 for _ in handle), str(path)
    return None, None


def run_bigquery_count(sql: str) -> dict[str, Any]:
    from google.cloud import bigquery

    project = os.getenv("GOOGLE_PROJECT")
    dataset = os.getenv("WORKSPACE_CDR")
    if not project or not dataset:
        raise RuntimeError("GOOGLE_PROJECT and WORKSPACE_CDR are required for BigQuery counts.")

    client = bigquery.Client(project=project)
    query = sql.format(dataset=dataset)
    rows = list(client.query(query).result())
    return dict(rows[0])


def source_bigquery_stages() -> list[FlowStage]:
    if os.getenv("RUN_BIGQUERY_COUNTS", "false").lower() not in {"1", "true", "yes"}:
        return []

    queries = [
        (
            "Participants with Fitbit sleep records",
            "Fitbit daily sleep summaries available",
            """
            SELECT COUNT(*) AS rows, COUNT(DISTINCT person_id) AS persons
            FROM `{dataset}.sleep_daily_summary`
            """,
        ),
        (
            "Participants with positive sleep duration",
            "Daily sleep duration >0 minutes",
            """
            SELECT COUNT(*) AS rows, COUNT(DISTINCT person_id) AS persons
            FROM `{dataset}.sleep_daily_summary`
            WHERE minute_asleep > 0
            """,
        ),
        (
            "Eligible sleep episodes",
            "Main sleep episodes with plausible duration",
            """
            SELECT COUNT(*) AS rows, COUNT(DISTINCT sl.person_id) AS persons
            FROM `{dataset}.sleep_level` sl
            INNER JOIN `{dataset}.sleep_daily_summary` sds
              ON sl.person_id = sds.person_id
             AND DATE(sl.sleep_date) = DATE(sds.sleep_date)
            WHERE sds.minute_asleep > 90
              AND sl.duration_in_min < 1080
              AND sl.duration_in_min > 0
              AND sl.duration_in_min != 960
              AND CAST(sl.is_main_sleep AS BOOL) IS TRUE
            """,
        ),
        (
            "Participants with EHR data",
            "At least one EHR-sourced clinical record",
            """
            WITH ehr AS (
              SELECT DISTINCT person_id FROM `{dataset}.measurement` m
              LEFT JOIN `{dataset}.measurement_ext` mm USING (measurement_id)
              WHERE LOWER(mm.src_id) LIKE 'ehr site%'
              UNION DISTINCT
              SELECT DISTINCT person_id FROM `{dataset}.condition_occurrence` m
              LEFT JOIN `{dataset}.condition_occurrence_ext` mm USING (condition_occurrence_id)
              WHERE LOWER(mm.src_id) LIKE 'ehr site%'
              UNION DISTINCT
              SELECT DISTINCT person_id FROM `{dataset}.device_exposure` m
              LEFT JOIN `{dataset}.device_exposure_ext` mm USING (device_exposure_id)
              WHERE LOWER(mm.src_id) LIKE 'ehr site%'
              UNION DISTINCT
              SELECT DISTINCT person_id FROM `{dataset}.drug_exposure` m
              LEFT JOIN `{dataset}.drug_exposure_ext` mm USING (drug_exposure_id)
              WHERE LOWER(mm.src_id) LIKE 'ehr site%'
              UNION DISTINCT
              SELECT DISTINCT person_id FROM `{dataset}.observation` m
              LEFT JOIN `{dataset}.observation_ext` mm USING (observation_id)
              WHERE LOWER(mm.src_id) LIKE 'ehr site%'
              UNION DISTINCT
              SELECT DISTINCT person_id FROM `{dataset}.procedure_occurrence` m
              LEFT JOIN `{dataset}.procedure_occurrence_ext` mm USING (procedure_occurrence_id)
              WHERE LOWER(mm.src_id) LIKE 'ehr site%'
              UNION DISTINCT
              SELECT DISTINCT person_id FROM `{dataset}.visit_occurrence` m
              LEFT JOIN `{dataset}.visit_occurrence_ext` mm USING (visit_occurrence_id)
              WHERE LOWER(mm.src_id) LIKE 'ehr site%'
            )
            SELECT COUNT(*) AS rows, COUNT(DISTINCT person_id) AS persons FROM ehr
            """,
        ),
    ]

    stages: list[FlowStage] = []
    for stage, filter_step, query in queries:
        print(f"Running BigQuery count: {stage}")
        result = run_bigquery_count(query)
        stages.append(
            FlowStage(
                source="bigquery",
                stage=stage,
                filter_step=filter_step,
                rows=int(result["rows"]),
                persons=int(result["persons"]),
                status="ok",
            )
        )
    return stages


def local_stages() -> list[FlowStage]:
    specs = [
        (
            "Valid nightly sleep observations",
            "Main sleep, >3 hours, largest nightly episode, and <15.5 total hours",
            "processed_data/daily_sleep_metrics_enhanced.parquet",
        ),
        (
            "Participants with baseline covariates",
            "Linked demographic, socioeconomic, survey, and BMI covariates",
            "processed_data/fitbit_cohort_covariates.parquet",
        ),
        (
            "Sleep observations with required demographics",
            "Nonmissing date of birth and sex at birth",
            "processed_data/ready_for_analysis.parquet",
        ),
        (
            "Analytic sample for sleep models",
            "Required model variables after sleep timing and geographic preparation",
            "processed_data/LMM_analysis.parquet",
        ),
        (
            "Analytic sample for EHR outcomes",
            "Fitbit baseline, covariates, and mapped phecode outcomes",
            "processed_data/master/master_covariates_only.parquet",
        ),
    ]

    stages: list[FlowStage] = []
    for stage, filter_step, path in specs:
        rows, persons, status = parquet_counts(path)
        stages.append(
            FlowStage(
                source="local parquet",
                stage=stage,
                filter_step=filter_step,
                rows=rows,
                persons=persons,
                status=status,
                path=path,
            )
        )

    pgrs_persons, pgrs_path = pgrs_patient_list_count()
    stages.append(
        FlowStage(
            source="local text",
            stage="Participants with genetic scores",
            filter_step="Overlap with genotype data and scoring eligibility criteria",
            rows=None,
            persons=pgrs_persons,
            status="ok" if pgrs_persons is not None else "missing",
            path=pgrs_path,
        )
    )
    return stages


def write_counts(stages: list[FlowStage], path: Path) -> pd.DataFrame:
    records = []
    for i, stage in enumerate(stages, start=1):
        records.append(
            {
                "step": i,
                "source": stage.source,
                "stage": stage.stage,
                "filter_step": stage.filter_step,
                "rows": stage.rows,
                "persons": stage.persons,
                "status": stage.status,
                "path": stage.path,
            }
        )
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return df


def build_label(row: pd.Series) -> str:
    parts = [
        textwrap.fill(str(row["stage"]), width=27),
        f"Participants: {mask_small_n(row['persons'])}",
    ]
    if pd.notna(row["rows"]):
        parts.append(f"Records: {mask_small_n(int(row['rows']))}")
    parts.append(textwrap.fill(str(row["filter_step"]), width=28))
    return "\n".join(parts)


def draw_flow_chart(df: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    n = len(df)
    box_width = 2.35
    box_height = 1.2
    gap = 0.6
    left_margin = 0.35
    right_margin = 0.35
    x_extent = left_margin + n * box_width + (n - 1) * gap + right_margin

    fig_width = max(11.0, n * 2.85)
    fig, ax = plt.subplots(figsize=(fig_width, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, x_extent)
    ax.set_ylim(0, 2.15)
    ax.axis("off")

    ax.text(
        left_margin,
        1.95,
        "Participant flow for analytic cohort construction",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="normal",
        color="black",
        family="serif",
    )

    for idx, (_, row) in enumerate(df.iterrows()):
        x = left_margin + idx * (box_width + gap)
        y = 0.55
        rect = Rectangle(
            (x, y),
            box_width,
            box_height,
            linewidth=0.8,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(
            x + box_width / 2,
            y + box_height / 2,
            build_label(row),
            ha="center",
            va="center",
            fontsize=7.1,
            color="black",
            family="serif",
            linespacing=1.05,
        )

        if idx < n - 1:
            ax.annotate(
                "",
                xy=(x + box_width + gap - 0.08, y + box_height / 2),
                xytext=(x + box_width + 0.08, y + box_height / 2),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8),
            )

    ax.text(
        left_margin,
        0.2,
        "Counts are aggregate participant and record counts; values <20 are masked.",
        ha="left",
        va="center",
        fontsize=7.5,
        color="black",
        family="serif",
    )

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    stages = source_bigquery_stages() + local_stages()
    counts_path = OUTPUT_DIR / "patient_flow_counts.csv"
    png_path = OUTPUT_DIR / "patient_flow_chart.png"
    pdf_path = OUTPUT_DIR / "patient_flow_chart.pdf"

    df = write_counts(stages, counts_path)
    draw_flow_chart(df, png_path, pdf_path)

    print(f"Wrote counts: {counts_path}")
    print(f"Wrote figure: {png_path}")
    print(f"Wrote figure: {pdf_path}")
    print(df[["step", "stage", "persons", "rows", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
