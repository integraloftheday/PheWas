#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path

import polars as pl


DEFAULT_GCS_PLINK_BASE = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/clinvar/plink_bed"
DEFAULT_GCS_ANCESTRY_PATH = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"


# Notebook-friendly configuration
sleep_parquet = "processed_data/ready_for_analysis.parquet"
phewas_parquet = "processed_data/master/master_phewas_wide.parquet"
person_ids_parquet = "processed_data/person_ids.parquet"
out_dir = Path("results/angus_genetics_overlap_audit")
gcs_plink_base = DEFAULT_GCS_PLINK_BASE
gcs_ancestry_path = DEFAULT_GCS_ANCESTRY_PATH
refresh_downloads = False


def ensure_file(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ensure_gcs_file(dest: Path, gcs_path: str, google_project: str | None, refresh: bool = False):
    if dest.exists() and not refresh:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gsutil"]
    if google_project:
        cmd.extend(["-u", google_project])
    cmd.extend(["cp", gcs_path, str(dest)])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest


def unique_ids_from_parquet(path: Path):
    return set(
        pl.read_parquet(path, columns=["person_id"])
        .select(pl.col("person_id").cast(pl.Utf8))
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )


def unique_ids_from_fam(path: Path):
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) >= 2 and fields[1]:
                ids.add(fields[1])
    return ids


def unique_ids_from_ancestry(path: Path):
    ancestry = pl.read_csv(path, separator="\t", infer_schema_length=0)
    if "research_id" not in ancestry.columns:
        raise ValueError(f"Ancestry TSV missing research_id column: {path}")
    return set(
        ancestry.select(pl.col("research_id").cast(pl.Utf8))
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )


def pct(numerator: int, denominator: int):
    if denominator == 0:
        return None
    return round(100.0 * numerator / denominator, 2)


def add_pairwise_overlap(rows, left_name, left_ids, right_name, right_ids):
    overlap_n = len(left_ids & right_ids)
    rows.append(
        {
            "left_cohort": left_name,
            "right_cohort": right_name,
            "overlap_n": overlap_n,
            "pct_of_left": pct(overlap_n, len(left_ids)),
            "pct_of_right": pct(overlap_n, len(right_ids)),
        }
    )


def run_overlap_audit(
    sleep_parquet=sleep_parquet,
    phewas_parquet=phewas_parquet,
    person_ids_parquet=person_ids_parquet,
    out_dir=out_dir,
    gcs_plink_base=gcs_plink_base,
    gcs_ancestry_path=gcs_ancestry_path,
    refresh_downloads=refresh_downloads,
):
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    google_project = os.getenv("GOOGLE_PROJECT")
    shared_dir = Path("processed_data/PGRS/shared")
    shared_dir.mkdir(parents=True, exist_ok=True)

    sleep_path = ensure_file(sleep_parquet, "Sleep phenotype parquet")
    phewas_path = ensure_file(phewas_parquet, "PheWAS parquet")
    person_ids_path = ensure_file(person_ids_parquet, "person_ids parquet")

    ancestry_path = ensure_gcs_file(
        shared_dir / "ancestry_preds.tsv",
        gcs_ancestry_path,
        google_project,
        refresh=refresh_downloads,
    )
    fam_path = ensure_gcs_file(
        shared_dir / "chr21.fam",
        f"{gcs_plink_base}/chr21.fam",
        google_project,
        refresh=refresh_downloads,
    )

    sleep_ids = unique_ids_from_parquet(sleep_path)
    phewas_ids = unique_ids_from_parquet(phewas_path)
    scorer_ids = unique_ids_from_parquet(person_ids_path)
    fam_ids = unique_ids_from_fam(fam_path)
    ancestry_ids = unique_ids_from_ancestry(ancestry_path)
    genetics_ids = fam_ids & ancestry_ids

    cohort_rows = [
        {"cohort": "sleep_ready_for_analysis", "n": len(sleep_ids)},
        {"cohort": "phewas_master_wide", "n": len(phewas_ids)},
        {"cohort": "existing_person_ids_parquet", "n": len(scorer_ids)},
        {"cohort": "aou_chr21_fam", "n": len(fam_ids)},
        {"cohort": "aou_ancestry_preds", "n": len(ancestry_ids)},
        {"cohort": "aou_scorable_genetics", "n": len(genetics_ids)},
    ]

    overlap_rows = []
    add_pairwise_overlap(overlap_rows, "sleep_ready_for_analysis", sleep_ids, "aou_scorable_genetics", genetics_ids)
    add_pairwise_overlap(overlap_rows, "phewas_master_wide", phewas_ids, "aou_scorable_genetics", genetics_ids)
    add_pairwise_overlap(overlap_rows, "existing_person_ids_parquet", scorer_ids, "aou_scorable_genetics", genetics_ids)
    add_pairwise_overlap(overlap_rows, "sleep_ready_for_analysis", sleep_ids, "phewas_master_wide", phewas_ids)
    add_pairwise_overlap(overlap_rows, "sleep_ready_for_analysis", sleep_ids, "existing_person_ids_parquet", scorer_ids)
    add_pairwise_overlap(overlap_rows, "phewas_master_wide", phewas_ids, "existing_person_ids_parquet", scorer_ids)

    strategy_rows = [
        {
            "analysis_target": "sleep_genetics_association",
            "recommended_n": len(sleep_ids & genetics_ids),
            "definition": "sleep_ready_for_analysis ∩ aou_scorable_genetics",
        },
        {
            "analysis_target": "prs_phewas",
            "recommended_n": len(sleep_ids & phewas_ids & genetics_ids),
            "definition": "sleep_ready_for_analysis ∩ phewas_master_wide ∩ aou_scorable_genetics",
        },
        {
            "analysis_target": "current_scorer_keep_list",
            "recommended_n": len(scorer_ids & genetics_ids),
            "definition": "existing_person_ids_parquet ∩ aou_scorable_genetics",
        },
    ]

    pl.DataFrame(cohort_rows).write_csv(tables_dir / "cohort_sizes.csv")
    pl.DataFrame(overlap_rows).write_csv(tables_dir / "cohort_overlaps.csv")
    pl.DataFrame(strategy_rows).write_csv(tables_dir / "recommended_analysis_counts.csv")

    summary_lines = [
        "# Angus genetics overlap audit",
        "",
        "## Cohort sizes",
        "",
        f"- Sleep phenotype cohort (`ready_for_analysis`): {len(sleep_ids):,}",
        f"- PheWAS cohort (`master_phewas_wide`): {len(phewas_ids):,}",
        f"- Existing scorer keep-list (`person_ids.parquet`): {len(scorer_ids):,}",
        f"- AoU chr21.fam IDs: {len(fam_ids):,}",
        f"- AoU ancestry_preds IDs: {len(ancestry_ids):,}",
        f"- AoU scorable genetics IDs (`chr21.fam ∩ ancestry_preds`): {len(genetics_ids):,}",
        "",
        "## Recommended analysis denominators",
        "",
        f"- Sleep + genetics overlap: {len(sleep_ids & genetics_ids):,}",
        f"- Sleep + genetics + PheWAS overlap: {len(sleep_ids & phewas_ids & genetics_ids):,}",
        f"- Current scorer keep-list + genetics overlap: {len(scorer_ids & genetics_ids):,}",
        "",
        "## Interpretation",
        "",
        "- If sleep + genetics is much larger than the current scorer keep-list overlap, the scorer should use the sleep cohort for association analyses.",
        "- The PheWAS denominator should remain the overlap of sleep, genetics, and PheWAS/EHR coverage.",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\nWrote: {out_dir / 'summary.md'}")
    print(f"Wrote: {tables_dir / 'cohort_sizes.csv'}")
    print(f"Wrote: {tables_dir / 'cohort_overlaps.csv'}")
    print(f"Wrote: {tables_dir / 'recommended_analysis_counts.csv'}")

    return {
        "cohort_rows": cohort_rows,
        "overlap_rows": overlap_rows,
        "strategy_rows": strategy_rows,
        "sleep_ids": sleep_ids,
        "phewas_ids": phewas_ids,
        "scorer_ids": scorer_ids,
        "genetics_ids": genetics_ids,
    }


if __name__ == "__main__":
    run_overlap_audit()
