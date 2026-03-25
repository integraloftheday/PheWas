#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET


ID_COLS = {
    "person_id",
    "participant_id",
    "patient_id",
    "subject_id",
    "individual_id",
    "mrn",
    "medical_record_number",
    "eid",
    "sample_id",
}

RISKY_NAME_PATTERNS = (
    "balanced_subset",
    "sleep_environment_merged",
    "subset_person_profile",
    "person_level",
)

FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg"}
RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    env_remove_rds = os.getenv("EXPORT_REMOVE_RDS", "1") == "1"

    parser = argparse.ArgumentParser(
        description=(
            "Export results as a zip archive. Default mode performs a sanitized "
            "copy; --figures creates a figure-only export with date text added."
        )
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results",
        help="Results directory to export (default: results)",
    )
    parser.add_argument(
        "out_zip",
        nargs="?",
        default=None,
        help="Output zip path (default: timestamped filename in cwd)",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help=(
            "Export only figure files, preserving folder structure, annotate each "
            "figure with current date text, then zip."
        ),
    )
    parser.add_argument(
        "--subpath",
        default=None,
        help=(
            "Relative path inside results_dir to export (file or directory). "
            "Example: --subpath model_summaries_04_5"
        ),
    )
    parser.add_argument(
        "--remove-rds",
        dest="remove_rds",
        action="store_true",
        default=env_remove_rds,
        help="Remove .rds files in sanitized export mode (default: on)",
    )
    parser.add_argument(
        "--no-remove-rds",
        dest="remove_rds",
        action="store_false",
        help="Keep .rds files in sanitized export mode",
    )
    return parser.parse_args()


def build_default_zip_name(figures_only: bool) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "results_figures_export" if figures_only else "results_export"
    return f"{prefix}_{stamp}.zip"


def ensure_results_dir(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"ERROR: Results directory not found: {path}")


def resolve_out_zip(out_zip: str | None, figures_only: bool) -> Path:
    if not out_zip:
        out_zip = build_default_zip_name(figures_only)
    zip_path = Path(out_zip)
    if not zip_path.is_absolute():
        zip_path = Path.cwd() / zip_path
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    return zip_path


def zip_folder(root_dir: Path, zip_path: Path, top_level_dir_name: str) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        base = root_dir / top_level_dir_name
        for f in sorted(base.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(root_dir).as_posix())


def resolve_export_root(results_dir: Path, subpath: str | None) -> Path:
    if not subpath:
        return results_dir

    candidate = (results_dir / subpath).resolve()
    base = results_dir.resolve()

    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise SystemExit(f"ERROR: --subpath must stay within {results_dir}") from exc

    if not candidate.exists():
        raise SystemExit(f"ERROR: Subpath not found under results_dir: {subpath}")

    return candidate


def iter_source_files(export_root: Path) -> list[Path]:
    if export_root.is_file():
        return [export_root]
    return [p for p in sorted(export_root.rglob("*")) if p.is_file()]


def copy_selected_to_stage(results_dir: Path, export_root: Path, stage_results: Path) -> None:
    rel = export_root.relative_to(results_dir)
    dst = stage_results / rel

    if export_root.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(export_root, dst)
    else:
        shutil.copytree(
            export_root,
            dst,
            ignore=shutil.ignore_patterns(".DS_Store"),
            dirs_exist_ok=True,
        )


def annotate_raster_image(path: Path, text: str) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required for --figures raster annotation. "
            "Install with: pip install pillow"
        ) from exc

    with Image.open(path) as img:
        base = img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        font_size = max(24, int(min(base.width, base.height) * 0.035))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w = right - left
        text_h = bottom - top
        pad = max(8, int(font_size * 0.35))
        margin = max(12, int(font_size * 0.6))

        x = max(margin, base.width - text_w - (2 * pad) - margin)
        y = max(margin, base.height - text_h - (2 * pad) - margin)

        draw.rectangle(
            (x, y, x + text_w + (2 * pad), y + text_h + (2 * pad)),
            fill=(255, 255, 255, 185),
        )
        draw.text((x + pad, y + pad), text, fill=(0, 0, 0, 255), font=font)

        out = Image.alpha_composite(base, overlay)
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            out.convert("RGB").save(path, quality=95)
        else:
            out.save(path)


def annotate_svg(path: Path, text: str) -> None:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(path)
    root = tree.getroot()

    if root.tag.startswith("{"):
        ns = root.tag.split("}", 1)[0].strip("{")
        text_tag = f"{{{ns}}}text"
    else:
        text_tag = "text"

    text_el = ET.Element(
        text_tag,
        {
            "x": "98%",
            "y": "98%",
            "text-anchor": "end",
            "font-size": "32",
            "fill": "black",
            "opacity": "0.9",
        },
    )
    text_el.text = text
    root.append(text_el)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def export_figures_only(results_dir: Path, export_root: Path, zip_path: Path) -> None:
    date_label = dt.date.today().isoformat()
    text = f"Exported {date_label}"

    with tempfile.TemporaryDirectory(prefix="phewas_export_") as tmp:
        tmp_root = Path(tmp)
        stage_results = tmp_root / "results"
        stage_results.mkdir(parents=True, exist_ok=True)

        copied = 0
        annotated = 0
        skipped = []

        for src in iter_source_files(export_root):
            if src.name == ".DS_Store":
                continue
            if src.suffix.lower() not in FIGURE_EXTS:
                continue

            rel = src.relative_to(results_dir)
            dst = stage_results / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

            try:
                if dst.suffix.lower() in RASTER_EXTS:
                    annotate_raster_image(dst, text)
                    annotated += 1
                elif dst.suffix.lower() == ".svg":
                    annotate_svg(dst, text)
                    annotated += 1
            except Exception as exc:
                skipped.append((rel.as_posix(), str(exc)))

        zip_folder(tmp_root, zip_path, "results")

    print(f"Figure export created: {zip_path}")
    print(f"Figures copied: {copied}")
    print(f"Figures annotated: {annotated}")
    if skipped:
        print("Figures copied but not annotated:")
        for rel, msg in skipped:
            print(f"  - {rel}: {msg}")


def remove_by_name_patterns(stage_results: Path, report: list[str]) -> None:
    for f in stage_results.rglob("*"):
        if not f.is_file():
            continue
        lname = f.name.lower()
        if any(pattern in lname for pattern in RISKY_NAME_PATTERNS):
            report.append(f.relative_to(stage_results).as_posix())
            f.unlink(missing_ok=True)


def remove_rds_files(stage_results: Path, report: list[str]) -> None:
    for f in stage_results.rglob("*.rds"):
        if f.is_file():
            report.append(f.relative_to(stage_results).as_posix())
            f.unlink(missing_ok=True)


def scan_identifier_columns(stage_results: Path, report: list[str]) -> None:
    pq = None
    try:
        import pyarrow.parquet as _pq  # type: ignore

        pq = _pq
    except Exception:
        pq = None

    for p in sorted(stage_results.rglob("*")):
        if not p.is_file():
            continue

        suffix = p.suffix.lower()

        if suffix in {".csv", ".tsv"}:
            delim = "\t" if suffix == ".tsv" else ","
            try:
                with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                    reader = csv.reader(f, delimiter=delim)
                    first = next(reader, None)
                if not first:
                    continue
                cols = [c.strip().strip('"').lower() for c in first]
                if any(c in ID_COLS for c in cols):
                    report.append(p.relative_to(stage_results).as_posix())
                    p.unlink(missing_ok=True)
            except Exception:
                continue

        elif suffix == ".parquet" and pq is not None:
            try:
                pf = pq.ParquetFile(p)
                cols = [c.lower() for c in pf.schema.names]
                if any(c in ID_COLS for c in cols):
                    report.append(p.relative_to(stage_results).as_posix())
                    p.unlink(missing_ok=True)
            except Exception:
                continue


def sanitized_export(
    results_dir: Path, export_root: Path, zip_path: Path, remove_rds: bool
) -> None:
    with tempfile.TemporaryDirectory(prefix="phewas_export_") as tmp:
        tmp_root = Path(tmp)
        stage_results = tmp_root / "results"

        stage_results.mkdir(parents=True, exist_ok=True)
        copy_selected_to_stage(results_dir, export_root, stage_results)

        report: list[str] = []

        remove_by_name_patterns(stage_results, report)
        if remove_rds:
            remove_rds_files(stage_results, report)
        scan_identifier_columns(stage_results, report)

        report = sorted(set(report))
        zip_folder(tmp_root, zip_path, "results")

    print(f"Sanitized export created: {zip_path}")
    if report:
        print("Removed potential person-level artifacts:")
        for item in report:
            print(f"  - {item}")
    else:
        print("No potential person-level artifacts were detected by current rules.")

    if not remove_rds:
        print("Note: --no-remove-rds was used; .rds files were retained.")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    ensure_results_dir(results_dir)
    export_root = resolve_export_root(results_dir, args.subpath)
    zip_path = resolve_out_zip(args.out_zip, args.figures)

    if args.figures:
        export_figures_only(results_dir, export_root, zip_path)
    else:
        sanitized_export(results_dir, export_root, zip_path, args.remove_rds)


if __name__ == "__main__":
    main()
