#!/usr/bin/env python3
"""
03_4_Chronotype_Clustering.py

Robust chronotype clustering from person-night sleep data.

What this script does:
1) Builds person-level chronotype features from work/free-day sleep behavior:
   - MSW: midpoint of sleep on workdays
   - MSF: midpoint of sleep on free days
   - MSFsc: sleep-corrected midpoint (MCTQ-style)
2) Uses circular-aware encoding (`sin`/`cos`) for midpoint variables.
3) Fits multiple clustering models (KMeans + Gaussian Mixture) across K.
4) Selects a preferred model (default: GMM with minimum BIC).
5) Produces standalone, publication-ready figures and descriptive tables.

Outputs: results/chronotype_clustering/
- tables/
- plots/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Configuration
# -----------------------------
INPUT_PARQUET = Path("processed_data/ready_for_analysis.parquet")
OUTPUT_DIR = Path("results/chronotype_clustering")
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"
for d in [OUTPUT_DIR, PLOT_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TEST_MODE = str(os.getenv("CHRONO_TEST_MODE", "0")).strip() == "1"
TEST_PERSON_MAX = int(str(os.getenv("CHRONO_TEST_PERSON_MAX", "30000")).strip() or "30000")

RANDOM_SEED = int(str(os.getenv("CHRONO_RANDOM_SEED", "2026")).strip() or "2026")
MIN_WORK_NIGHTS = int(str(os.getenv("CHRONO_MIN_WORK_NIGHTS", "3")).strip() or "3")
MIN_FREE_NIGHTS = int(str(os.getenv("CHRONO_MIN_FREE_NIGHTS", "2")).strip() or "2")
MIN_TOTAL_NIGHTS = int(str(os.getenv("CHRONO_MIN_TOTAL_NIGHTS", "10")).strip() or "10")

K_MIN = int(str(os.getenv("CHRONO_K_MIN", "2")).strip() or "2")
K_MAX = int(str(os.getenv("CHRONO_K_MAX", "8")).strip() or "8")
MODEL_SELECTION_RULE = str(os.getenv("CHRONO_MODEL_RULE", "gmm_bic")).strip().lower()

sns.set_style("whitegrid")


# -----------------------------
# Helpers
# -----------------------------
def to_noon_linear_expr(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr < 12).then(expr + 24).otherwise(expr)


def linear_to_clock(arr: np.ndarray) -> np.ndarray:
    return np.mod(arr, 24.0)


def wrap_to_noon_linear(arr: np.ndarray) -> np.ndarray:
    return ((arr - 12.0) % 24.0) + 12.0


def circular_diff_hours(a_linear: np.ndarray, b_linear: np.ndarray) -> np.ndarray:
    """Shortest signed circular difference in hours in [-12, 12)."""
    return ((a_linear - b_linear + 12.0) % 24.0) - 12.0


def circular_mean_clock(hours_clock: np.ndarray) -> float:
    """Circular mean on 24-hour clock, returned in [0,24)."""
    theta = 2.0 * np.pi * (hours_clock / 24.0)
    s = np.nanmean(np.sin(theta))
    c = np.nanmean(np.cos(theta))
    ang = np.arctan2(s, c)
    if ang < 0:
        ang += 2.0 * np.pi
    return float(24.0 * ang / (2.0 * np.pi))


def pick_columns(schema_cols: set[str]) -> Dict[str, str]:
    if "onset_linear" in schema_cols:
        onset_col = "onset_linear"
    elif "daily_start_hour" in schema_cols:
        onset_col = "daily_start_hour"
    else:
        raise ValueError("Missing onset columns: need onset_linear or daily_start_hour.")

    if "midpoint_linear" in schema_cols:
        midpoint_col = "midpoint_linear"
    elif "daily_midpoint_hour" in schema_cols:
        midpoint_col = "daily_midpoint_hour"
    else:
        raise ValueError("Missing midpoint columns: need midpoint_linear or daily_midpoint_hour.")

    if "daily_duration_mins" in schema_cols:
        dur_col = "daily_duration_mins"
    elif "daily_sleep_window_mins" in schema_cols:
        dur_col = "daily_sleep_window_mins"
    else:
        raise ValueError("Missing duration columns: need daily_duration_mins or daily_sleep_window_mins.")

    if "is_weekend_or_holiday" in schema_cols:
        weekend_col = "is_weekend_or_holiday"
    elif "is_weekend" in schema_cols:
        weekend_col = "is_weekend"
    else:
        weekend_col = "__derived_from_date__"

    return {
        "onset_col": onset_col,
        "midpoint_col": midpoint_col,
        "duration_col": dur_col,
        "weekend_col": weekend_col,
    }


def build_person_features(lf: pl.LazyFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    schema_cols = set(lf.collect_schema().names())
    cols = pick_columns(schema_cols)

    onset_col = cols["onset_col"]
    midpoint_col = cols["midpoint_col"]
    duration_col = cols["duration_col"]
    weekend_col = cols["weekend_col"]

    if weekend_col == "__derived_from_date__":
        # ISO weekday in Polars is 1..7 (Mon..Sun), so weekend is 6/7.
        is_weekend_expr = (pl.col("sleep_date").cast(pl.Date).dt.weekday() >= 6)
    else:
        is_weekend_expr = pl.col(weekend_col).cast(pl.Boolean)

    base = (
        lf.select([
            "person_id",
            "sleep_date",
            onset_col,
            midpoint_col,
            duration_col,
        ] + ([weekend_col] if weekend_col != "__derived_from_date__" else []))
        .with_columns([
            pl.col("sleep_date").cast(pl.Date).alias("sleep_date"),
            to_noon_linear_expr(pl.col(onset_col).cast(pl.Float64)).alias("onset_linear_use"),
            to_noon_linear_expr(pl.col(midpoint_col).cast(pl.Float64)).alias("midpoint_linear_use"),
            (pl.col(duration_col).cast(pl.Float64) / 60.0).alias("duration_hours"),
            is_weekend_expr.alias("is_weekend"),
            pl.when(is_weekend_expr).then(pl.lit(0)).otherwise(pl.lit(1)).alias("is_work_day"),
        ])
        .drop_nulls(["person_id", "sleep_date", "onset_linear_use", "midpoint_linear_use", "duration_hours", "is_work_day"])
    )

    # Optional fast dev mode: deterministic person hash downsample.
    if TEST_MODE:
        print(f"CHRONO_TEST_MODE=1: restricting to approx {TEST_PERSON_MAX:,} people via hash filter.")
        people_n = int(base.select(pl.col("person_id").n_unique().alias("n_people")).collect(engine="streaming")["n_people"][0])
        if people_n > TEST_PERSON_MAX:
            mod = int(np.ceil(people_n / TEST_PERSON_MAX))
            base = base.filter((pl.col("person_id").cast(pl.Utf8).hash(seed=RANDOM_SEED) % mod) == 0)

    # Person x daytype means for onset/duration and counts.
    person_daytype = (
        base.group_by(["person_id", "is_work_day"])
        .agg([
            pl.col("onset_linear_use").mean().alias("mean_onset_linear"),
            pl.col("duration_hours").mean().alias("mean_duration_hours"),
            pl.col("midpoint_linear_use").mean().alias("mean_midpoint_linear"),
            pl.col("midpoint_linear_use").std().alias("sd_midpoint_linear"),
            pl.len().alias("n_nights"),
        ])
        .collect(engine="streaming")
        .to_pandas()
    )

    if person_daytype.empty:
        return pd.DataFrame(), {"reason": "No person-daytype rows after preprocessing."}

    work = person_daytype[person_daytype["is_work_day"] == 1].copy().rename(
        columns={
            "mean_onset_linear": "SO_w",
            "mean_duration_hours": "SD_w",
            "mean_midpoint_linear": "MID_w",
            "sd_midpoint_linear": "MID_w_sd",
            "n_nights": "n_work_nights",
        }
    )
    free = person_daytype[person_daytype["is_work_day"] == 0].copy().rename(
        columns={
            "mean_onset_linear": "SO_f",
            "mean_duration_hours": "SD_f",
            "mean_midpoint_linear": "MID_f",
            "sd_midpoint_linear": "MID_f_sd",
            "n_nights": "n_free_nights",
        }
    )

    person = work[["person_id", "SO_w", "SD_w", "MID_w", "MID_w_sd", "n_work_nights"]].merge(
        free[["person_id", "SO_f", "SD_f", "MID_f", "MID_f_sd", "n_free_nights"]],
        on="person_id",
        how="inner",
    )

    if person.empty:
        return pd.DataFrame(), {"reason": "No overlapping persons with both work and free-day data."}

    # Filtering quality thresholds.
    person["n_total_nights"] = person["n_work_nights"] + person["n_free_nights"]
    min_work = 1 if TEST_MODE else MIN_WORK_NIGHTS
    min_free = 1 if TEST_MODE else MIN_FREE_NIGHTS
    min_total = 2 if TEST_MODE else MIN_TOTAL_NIGHTS

    person = person[
        (person["n_work_nights"] >= min_work)
        & (person["n_free_nights"] >= min_free)
        & (person["n_total_nights"] >= min_total)
    ].copy()

    if person.empty:
        return pd.DataFrame(), {"reason": "No persons remain after minimum-night filters."}

    # MCTQ midpoint features.
    person["MSW_linear"] = wrap_to_noon_linear(person["SO_w"].to_numpy(dtype=float) + person["SD_w"].to_numpy(dtype=float) / 2.0)
    person["MSF_linear"] = wrap_to_noon_linear(person["SO_f"].to_numpy(dtype=float) + person["SD_f"].to_numpy(dtype=float) / 2.0)
    person["SD_week"] = ((person["SD_w"] * 5.0) + (person["SD_f"] * 2.0)) / 7.0

    msf_sc = np.where(
        person["SD_f"].to_numpy(dtype=float) > person["SD_w"].to_numpy(dtype=float),
        person["SO_f"].to_numpy(dtype=float) + person["SD_week"].to_numpy(dtype=float) / 2.0,
        person["MSF_linear"].to_numpy(dtype=float),
    )
    person["MSFsc_linear"] = wrap_to_noon_linear(msf_sc)

    # Convert to clock hours for interpretation.
    person["MSW_clock"] = linear_to_clock(person["MSW_linear"].to_numpy(dtype=float))
    person["MSF_clock"] = linear_to_clock(person["MSF_linear"].to_numpy(dtype=float))
    person["MSFsc_clock"] = linear_to_clock(person["MSFsc_linear"].to_numpy(dtype=float))

    # Social jetlag and duration contrasts.
    person["SJL_hours"] = circular_diff_hours(person["MSF_linear"].to_numpy(dtype=float), person["MSW_linear"].to_numpy(dtype=float))
    person["SD_diff_free_minus_work"] = person["SD_f"] - person["SD_w"]
    person["MID_diff_free_minus_work"] = circular_diff_hours(person["MID_f"].to_numpy(dtype=float), person["MID_w"].to_numpy(dtype=float))
    person["midpoint_variability"] = np.nanmean(
        np.column_stack([person["MID_w_sd"].to_numpy(dtype=float), person["MID_f_sd"].to_numpy(dtype=float)]),
        axis=1,
    )

    # Circular encodings (robust for clustering).
    for tag in ["MSW", "MSF", "MSFsc"]:
        theta = 2.0 * np.pi * (person[f"{tag}_clock"].to_numpy(dtype=float) / 24.0)
        person[f"{tag}_sin"] = np.sin(theta)
        person[f"{tag}_cos"] = np.cos(theta)

    person = person.replace([np.inf, -np.inf], np.nan).dropna().copy()

    meta = {
        "onset_col": onset_col,
        "midpoint_col": midpoint_col,
        "duration_col": duration_col,
        "weekend_col": weekend_col,
        "n_people_final": int(person.shape[0]),
        "min_work_used": int(min_work),
        "min_free_used": int(min_free),
        "min_total_used": int(min_total),
    }
    return person, meta


def choose_cluster_names(n_clusters: int) -> List[str]:
    if n_clusters == 2:
        return ["Earlier Chronotype", "Later Chronotype"]
    if n_clusters == 3:
        return ["Lark-like", "Intermediate", "Owl-like"]
    if n_clusters == 4:
        return ["Strong Lark-like", "Mild Lark-like", "Intermediate/Late", "Strong Owl-like"]
    return [f"Chronotype Group {i+1}" for i in range(n_clusters)]


def run_clustering(person: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    feature_cols = [
        "MSW_sin", "MSW_cos",
        "MSF_sin", "MSF_cos",
        "MSFsc_sin", "MSFsc_cos",
        "SJL_hours",
        "SD_w", "SD_f", "SD_diff_free_minus_work", "SD_week",
        "midpoint_variability",
        "MID_diff_free_minus_work",
    ]

    X = person[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_people = int(person.shape[0])
    k_upper = min(K_MAX, max(2, n_people - 1))
    k_lower = min(K_MIN, k_upper)
    k_values = list(range(k_lower, k_upper + 1))
    if not k_values or n_people < 3:
        raise RuntimeError("Insufficient people for clustering. Need at least 3 persons after filtering.")
    metrics_rows: List[Dict[str, object]] = []

    fitted_models: Dict[Tuple[str, int], object] = {}
    labels_store: Dict[Tuple[str, int], np.ndarray] = {}

    for k in k_values:
        # KMeans
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=30)
        lbl_km = km.fit_predict(Xs)
        fitted_models[("kmeans", k)] = km
        labels_store[("kmeans", k)] = lbl_km

        sil_km = silhouette_score(Xs, lbl_km) if k >= 2 else np.nan
        ch_km = calinski_harabasz_score(Xs, lbl_km) if k >= 2 else np.nan
        db_km = davies_bouldin_score(Xs, lbl_km) if k >= 2 else np.nan

        metrics_rows.append(
            {
                "model": "kmeans",
                "k": k,
                "silhouette": float(sil_km),
                "calinski_harabasz": float(ch_km),
                "davies_bouldin": float(db_km),
                "aic": np.nan,
                "bic": np.nan,
                "inertia": float(km.inertia_),
            }
        )

        # GMM
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_SEED, n_init=5)
        gm.fit(Xs)
        lbl_gm = gm.predict(Xs)
        fitted_models[("gmm", k)] = gm
        labels_store[("gmm", k)] = lbl_gm

        sil_gm = silhouette_score(Xs, lbl_gm) if k >= 2 else np.nan
        ch_gm = calinski_harabasz_score(Xs, lbl_gm) if k >= 2 else np.nan
        db_gm = davies_bouldin_score(Xs, lbl_gm) if k >= 2 else np.nan

        metrics_rows.append(
            {
                "model": "gmm",
                "k": k,
                "silhouette": float(sil_gm),
                "calinski_harabasz": float(ch_gm),
                "davies_bouldin": float(db_gm),
                "aic": float(gm.aic(Xs)),
                "bic": float(gm.bic(Xs)),
                "inertia": np.nan,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    # Model selection rule
    if MODEL_SELECTION_RULE == "gmm_bic":
        cands = metrics_df[metrics_df["model"] == "gmm"].dropna(subset=["bic"]).copy()
        best_row = cands.sort_values("bic", ascending=True).iloc[0]
    elif MODEL_SELECTION_RULE == "best_silhouette":
        best_row = metrics_df.sort_values("silhouette", ascending=False).iloc[0]
    else:
        # fallback
        cands = metrics_df[metrics_df["model"] == "gmm"].dropna(subset=["bic"]).copy()
        best_row = cands.sort_values("bic", ascending=True).iloc[0]

    best_model = str(best_row["model"])
    best_k = int(best_row["k"])
    labels_raw = labels_store[(best_model, best_k)].astype(int)

    assign = person.copy()
    assign["cluster_raw"] = labels_raw

    # Order clusters by circular center of MSFsc (early -> late)
    centers = (
        assign.groupby("cluster_raw", as_index=False)
        .agg(msfsc_center_clock=("MSFsc_clock", lambda x: circular_mean_clock(np.asarray(x, dtype=float))))
    )
    centers = centers.sort_values("msfsc_center_clock", ascending=True).reset_index(drop=True)
    centers["cluster_order"] = np.arange(1, len(centers) + 1)

    labels_pretty = choose_cluster_names(best_k)
    if len(labels_pretty) < best_k:
        labels_pretty = [f"Chronotype Group {i+1}" for i in range(best_k)]
    centers["cluster_label"] = labels_pretty[:best_k]

    assign = assign.merge(centers, on="cluster_raw", how="left")
    assign["cluster_id"] = assign["cluster_order"].astype(int)

    # Cluster profile table
    profile = (
        assign.groupby(["cluster_id", "cluster_label"], as_index=False)
        .agg(
            n_people=("person_id", "size"),
            MSW_clock_mean=("MSW_clock", "mean"),
            MSF_clock_mean=("MSF_clock", "mean"),
            MSFsc_clock_mean=("MSFsc_clock", "mean"),
            SJL_hours_mean=("SJL_hours", "mean"),
            SD_w_mean=("SD_w", "mean"),
            SD_f_mean=("SD_f", "mean"),
            midpoint_variability_mean=("midpoint_variability", "mean"),
            msfsc_center_clock=("msfsc_center_clock", "mean"),
        )
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )

    meta = {
        "feature_cols": feature_cols,
        "selected_model": best_model,
        "selected_k": best_k,
        "model_rule": MODEL_SELECTION_RULE,
    }
    return assign, metrics_df, profile, meta


def save_figures(assign: pd.DataFrame, metrics_df: pd.DataFrame, profile: pd.DataFrame, feature_cols: List[str], selected_model: str, selected_k: int) -> None:
    n = int(assign.shape[0])

    # 1) Model selection figure
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    gmm = metrics_df[metrics_df["model"] == "gmm"].sort_values("k")
    km = metrics_df[metrics_df["model"] == "kmeans"].sort_values("k")

    axes[0].plot(gmm["k"], gmm["bic"], marker="o", label="GMM BIC")
    axes[0].plot(gmm["k"], gmm["aic"], marker="o", label="GMM AIC")
    axes[0].set_title("Model selection (GMM information criteria)")
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Criterion value (lower is better)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(km["k"], km["silhouette"], marker="o", label="KMeans silhouette")
    axes[1].plot(gmm["k"], gmm["silhouette"], marker="o", label="GMM silhouette")
    axes[1].set_title("Model selection (silhouette)")
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Silhouette (higher is better)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    axes[2].plot(km["k"], km["davies_bouldin"], marker="o", label="KMeans DB")
    axes[2].plot(gmm["k"], gmm["davies_bouldin"], marker="o", label="GMM DB")
    axes[2].set_title("Model selection (Davies-Bouldin)")
    axes[2].set_xlabel("Number of clusters (K)")
    axes[2].set_ylabel("Davies-Bouldin (lower is better)")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        f"Chronotype clustering model diagnostics | N={n:,} persons | selected={selected_model}, K={selected_k}",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_model_selection_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2) Distribution overlays: MSW/MSF/MSFsc
    fig2, ax2 = plt.subplots(figsize=(10.5, 6.5))
    bins = np.arange(0.0, 24.25, 0.25)
    ax2.hist(assign["MSW_clock"], bins=bins, alpha=0.42, density=True, label="MSW (workday midpoint)")
    ax2.hist(assign["MSF_clock"], bins=bins, alpha=0.42, density=True, label="MSF (free-day midpoint)")
    ax2.hist(assign["MSFsc_clock"], bins=bins, alpha=0.42, density=True, label="MSFsc (sleep-corrected midpoint)")
    ax2.set_xlabel("Clock hour")
    ax2.set_ylabel("Density")
    ax2.set_title(
        "Chronotype midpoint distributions (person-level)\n"
        "MSW = SO_w + SD_w/2; MSF = SO_f + SD_f/2; MSFsc corrected for free-day oversleep"
    )
    ax2.legend(loc="best")
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_midpoint_distributions_msw_msf_msfsc.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # 3) PCA scatter with cluster labels
    pca_features = assign[feature_cols].to_numpy(dtype=float)
    pca_scaled = StandardScaler().fit_transform(pca_features)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pcs = pca.fit_transform(pca_scaled)

    pca_df = assign[["cluster_id", "cluster_label"]].copy()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]

    fig3, ax3 = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10", n_colors=max(3, int(assign["cluster_id"].nunique())))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="cluster_label",
        alpha=0.28,
        s=16,
        linewidth=0,
        ax=ax3,
        palette=palette,
    )
    ax3.set_title(
        "Chronotype clusters in PCA feature space\n"
        f"Model={selected_model}, K={selected_k}, N={n:,}; features include circular midpoint encodings + SJL + duration"
    )
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax3.grid(alpha=0.25)
    ax3.legend(title="Cluster", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_cluster_scatter_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # 4) Cluster profile heatmap (z-scored means)
    profile_cols = [
        "MSW_clock", "MSF_clock", "MSFsc_clock", "SJL_hours",
        "SD_w", "SD_f", "SD_diff_free_minus_work", "midpoint_variability",
    ]
    prof = (
        assign.groupby(["cluster_id", "cluster_label"], as_index=False)[profile_cols]
        .mean()
        .sort_values("cluster_id")
    )
    z = prof[profile_cols].copy()
    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=0).replace(0, np.nan)

    fig4, ax4 = plt.subplots(figsize=(12, 5.8))
    sns.heatmap(
        z.set_index(prof["cluster_label"]),
        cmap="coolwarm",
        center=0,
        ax=ax4,
        cbar_kws={"label": "Standardized cluster mean (z-score)"},
    )
    ax4.set_title(
        "Chronotype cluster profiles (standardized means)\n"
        "Rows = clusters; columns = midpoint, social jetlag, and duration characteristics"
    )
    ax4.set_xlabel("Feature")
    ax4.set_ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_cluster_profile_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    # 5) Standalone boxplots for interpretability
    fig5, axes5 = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    long_cols = [
        ("MSW_clock", "MSW (workday midpoint, clock hour)"),
        ("MSF_clock", "MSF (free-day midpoint, clock hour)"),
        ("MSFsc_clock", "MSFsc (adjusted midpoint, clock hour)"),
        ("SJL_hours", "Social jetlag (hours; MSF-MSW, circular)"),
    ]

    for ax, (col, ttl) in zip(axes5.flatten(), long_cols):
        sns.boxplot(data=assign, x="cluster_label", y=col, ax=ax)
        ax.set_title(ttl)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.2)

    fig5.suptitle(
        "Cluster interpretation panels: work/free/adjusted midpoints and social jetlag\n"
        f"Model={selected_model}, K={selected_k}, N={n:,}",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_cluster_interpretation_panels.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)

    # 6) Cluster size + chronotype center summary
    fig6, ax6 = plt.subplots(figsize=(10.5, 6.0))
    prof_plot = profile.copy().sort_values("cluster_id")
    bars = ax6.bar(prof_plot["cluster_label"], prof_plot["n_people"], alpha=0.85)
    for b, center in zip(bars, prof_plot["msfsc_center_clock"]):
        ax6.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            f"MSFsc center={center:.2f}h",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    ax6.set_title("Cluster sizes with circular MSFsc centers")
    ax6.set_xlabel("Cluster")
    ax6.set_ylabel("Number of people")
    ax6.tick_params(axis="x", rotation=20)
    ax6.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_cluster_sizes_and_centers.png", dpi=300, bbox_inches="tight")
    plt.close(fig6)


def main() -> None:
    print(f"Loading data from {INPUT_PARQUET}...")
    lf = pl.scan_parquet(INPUT_PARQUET)

    person, build_meta = build_person_features(lf)
    if person.empty:
        run_meta = {
            "input_parquet": str(INPUT_PARQUET),
            "status": "no_data",
            "reason": str(build_meta.get("reason", "Unknown")),
            "test_mode": TEST_MODE,
        }
        run_meta.update(build_meta)
        pd.DataFrame([run_meta]).to_csv(TABLE_DIR / "00_run_metadata.csv", index=False)

        # Placeholder figures so downstream docs still have stand-alone assets.
        for figname, title, msg in [
            ("01_model_selection_diagnostics.png", "Model selection diagnostics (not available)", run_meta["reason"]),
            ("02_midpoint_distributions_msw_msf_msfsc.png", "MSW/MSF/MSFsc distributions (not available)", run_meta["reason"]),
            ("03_cluster_scatter_pca.png", "PCA cluster map (not available)", run_meta["reason"]),
            ("04_cluster_profile_heatmap.png", "Cluster profile heatmap (not available)", run_meta["reason"]),
            ("05_cluster_interpretation_panels.png", "Cluster interpretation panels (not available)", run_meta["reason"]),
            ("06_cluster_sizes_and_centers.png", "Cluster sizes and centers (not available)", run_meta["reason"]),
        ]:
            fig = plt.figure(figsize=(10, 5.5))
            plt.text(0.5, 0.55, msg, ha="center", va="center")
            plt.text(0.5, 0.42, "No cluster model fitted.", ha="center", va="center")
            plt.axis("off")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / figname, dpi=220, bbox_inches="tight")
            plt.close(fig)

        print("No person-level features available for clustering. Wrote metadata and placeholder figures.")
        print(f"Tables: {TABLE_DIR}")
        print(f"Plots:  {PLOT_DIR}")
        return

    assign, metrics_df, profile_df, cluster_meta = run_clustering(person)

    # Save tables
    run_meta = {
        "input_parquet": str(INPUT_PARQUET),
        "test_mode": TEST_MODE,
        "test_person_max": TEST_PERSON_MAX,
        "random_seed": RANDOM_SEED,
        "min_work_nights": MIN_WORK_NIGHTS,
        "min_free_nights": MIN_FREE_NIGHTS,
        "min_total_nights": MIN_TOTAL_NIGHTS,
        "k_min": K_MIN,
        "k_max": K_MAX,
        "model_rule": MODEL_SELECTION_RULE,
        "selected_model": cluster_meta["selected_model"],
        "selected_k": cluster_meta["selected_k"],
        "n_people_clustered": int(assign.shape[0]),
        "onset_col": build_meta["onset_col"],
        "midpoint_col": build_meta["midpoint_col"],
        "duration_col": build_meta["duration_col"],
        "weekend_col": build_meta["weekend_col"],
    }

    pd.DataFrame([run_meta]).to_csv(TABLE_DIR / "00_run_metadata.csv", index=False)
    person.to_csv(TABLE_DIR / "01_person_level_features.csv", index=False)
    metrics_df.sort_values(["model", "k"]).to_csv(TABLE_DIR / "02_model_selection_metrics.csv", index=False)
    assign.sort_values(["cluster_id", "person_id"]).to_csv(TABLE_DIR / "03_cluster_assignments.csv", index=False)
    profile_df.sort_values("cluster_id").to_csv(TABLE_DIR / "04_cluster_profiles.csv", index=False)

    # Correlation table for transparency
    corr_cols = [
        "MSW_clock", "MSF_clock", "MSFsc_clock",
        "SJL_hours", "SD_w", "SD_f", "SD_diff_free_minus_work", "midpoint_variability",
    ]
    corr = person[corr_cols].corr(numeric_only=True)
    corr.to_csv(TABLE_DIR / "05_feature_correlations.csv")

    save_figures(
        assign=assign,
        metrics_df=metrics_df,
        profile=profile_df,
        feature_cols=cluster_meta["feature_cols"],
        selected_model=cluster_meta["selected_model"],
        selected_k=int(cluster_meta["selected_k"]),
    )

    print("Done.")
    print(f"Tables: {TABLE_DIR}")
    print(f"Plots:  {PLOT_DIR}")


if __name__ == "__main__":
    main()
