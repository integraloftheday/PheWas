#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:-results}"
OUT_ZIP="${2:-results_export_$(date +%Y%m%d_%H%M%S).zip}"
REMOVE_RDS="${EXPORT_REMOVE_RDS:-1}"

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "ERROR: Results directory not found: $RESULTS_DIR" >&2
  exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "ERROR: zip command not found." >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d -t phewas_export_XXXXXX)"
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

STAGE_RESULTS="$TMP_ROOT/results"
mkdir -p "$STAGE_RESULTS"

# Stage a copy so the original results directory is untouched.
rsync -a --exclude='.DS_Store' "$RESULTS_DIR/" "$STAGE_RESULTS/"

REPORT_FILE="$TMP_ROOT/removed_artifacts.txt"
: > "$REPORT_FILE"

# 1) Remove known risky files by naming pattern.
while IFS= read -r f; do
  rel="${f#${STAGE_RESULTS}/}"
  echo "$rel" >> "$REPORT_FILE"
  rm -f "$f"
done < <(
  find "$STAGE_RESULTS" -type f \( \
    -iname '*balanced_subset*' -o \
    -iname '*sleep_environment_merged*' -o \
    -iname '*subset_person_profile*' -o \
    -iname '*person_level*' \
  \)
)

# 2) Optional conservative removal of RDS artifacts (models and opaque binaries).
if [[ "$REMOVE_RDS" == "1" ]]; then
  while IFS= read -r f; do
    rel="${f#${STAGE_RESULTS}/}"
    echo "$rel" >> "$REPORT_FILE"
    rm -f "$f"
  done < <(find "$STAGE_RESULTS" -type f -iname '*.rds')
fi

# 3) Header/schema scan for identifier columns in CSV/TSV/Parquet.
python3 - "$STAGE_RESULTS" "$REPORT_FILE" <<'PY'
from pathlib import Path
import csv
import sys

root = Path(sys.argv[1])
report = Path(sys.argv[2])

id_cols = {
    "person_id", "participant_id", "patient_id", "subject_id", "individual_id",
    "mrn", "medical_record_number", "eid", "sample_id"
}

removed = []

# Lazy optional parquet support.
pq = None
try:
    import pyarrow.parquet as _pq  # type: ignore
    pq = _pq
except Exception:
    pq = None

for p in sorted(root.rglob("*")):
    if not p.is_file():
        continue

    lower_name = p.name.lower()
    suffix = p.suffix.lower()

    # CSV/TSV header scan
    if suffix in {".csv", ".tsv"}:
        try:
            with p.open("r", newline="", encoding="utf-8", errors="ignore") as f:
                first = f.readline().strip("\n\r")
            if not first:
                continue
            delim = "\t" if suffix == ".tsv" else ","
            cols = [c.strip().strip('"').lower() for c in first.split(delim)]
            if any(c in id_cols for c in cols):
                removed.append(p)
        except Exception:
            pass

    # Parquet schema scan
    elif suffix == ".parquet" and pq is not None:
        try:
            pf = pq.ParquetFile(p)
            cols = [c.lower() for c in pf.schema.names]
            if any(c in id_cols for c in cols):
                removed.append(p)
        except Exception:
            pass

if removed:
    with report.open("a", encoding="utf-8") as rf:
        for p in removed:
            rel = p.relative_to(root).as_posix()
            rf.write(rel + "\n")
            try:
                p.unlink()
            except Exception:
                pass
PY

# Deduplicate and sort report for readability.
if [[ -s "$REPORT_FILE" ]]; then
  sort -u "$REPORT_FILE" -o "$REPORT_FILE"
fi

# Create zip in current working directory unless absolute path supplied.
if [[ "$OUT_ZIP" = /* ]]; then
  ZIP_PATH="$OUT_ZIP"
else
  ZIP_PATH="$PWD/$OUT_ZIP"
fi

( cd "$TMP_ROOT" && zip -qr "$ZIP_PATH" results )

echo "Sanitized export created: $ZIP_PATH"
if [[ -s "$REPORT_FILE" ]]; then
  echo "Removed potential person-level artifacts:"
  sed 's/^/  - /' "$REPORT_FILE"
else
  echo "No potential person-level artifacts were detected by current rules."
fi

if [[ "$REMOVE_RDS" != "1" ]]; then
  echo "Note: EXPORT_REMOVE_RDS=0 was used; .rds files were retained."
fi
