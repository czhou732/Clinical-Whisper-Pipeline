#!/usr/bin/env python3
"""
generate_figures_streamB.py — Auto-generate poster figures from Stream B fMRI data.

Usage:
    python3 generate_figures_streamB.py --input /path/to/stream_b_features.csv
    python3 generate_figures_streamB.py --demo   # synthetic data for testing

Outputs publication-quality figures to Symposium/Figures/StreamB/:
    1. nacc_activation.png       — NAcc BOLD activation by group (violin + swarm)
    2. bold_by_condition.png     — BOLD signal by task condition × group
    3. correlation_scatter.png   — NAcc activation vs anhedonia score
    4. stats_table.png           — Group comparison statistics
    5. roi_metrics_heatmap.png   — ROI metrics correlation matrix

All figures use USC brand colors and 300 DPI for print quality.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Figures", "StreamB")
os.makedirs(FIG_DIR, exist_ok=True)

# USC brand
CARDINAL = "#990000"
GOLD = "#FFCC00"
DARK_GRAY = "#333333"
LIGHT_GRAY = "#F0F0F0"
TEAL = "#0E8A7D"
BLUE_NEURAL = "#2C5F8A"

# Group colors
COLOR_DEPRESSED = CARDINAL
COLOR_NORMAL = TEAL

# Anhedonia threshold (Chapman Physical Anhedonia Scale crosswalk)
# ds000030 uses MASQ anhedonia subscale; threshold = median split or clinical cutoff
ANHEDONIA_THRESHOLD = 25  # MASQ anhedonia subscale clinical cutoff

# Key ROI metrics
ROI_METRICS = [
    "nacc_bold_mean",
    "nacc_bold_peak",
    "nacc_bold_variance",
    "caudate_bold_mean",
    "putamen_bold_mean",
    "vmpfc_bold_mean",
]

DISPLAY_NAMES = {
    "nacc_bold_mean": "NAcc Mean BOLD",
    "nacc_bold_peak": "NAcc Peak BOLD",
    "nacc_bold_variance": "NAcc BOLD Variance",
    "caudate_bold_mean": "Caudate Mean BOLD",
    "putamen_bold_mean": "Putamen Mean BOLD",
    "vmpfc_bold_mean": "vmPFC Mean BOLD",
    "nacc_gain_bold": "NAcc BOLD (Gain)",
    "nacc_loss_bold": "NAcc BOLD (Loss)",
    "nacc_neutral_bold": "NAcc BOLD (Neutral)",
}

# ── Style ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def get_feature_cols(df):
    """Return list of all numeric feature columns (exclude metadata)."""
    exclude = {"participant_id", "subject_id", "session", "age", "gender",
               "diagnosis", "group", "depressed", "anhedonia_score",
               "masq_anhedonia", "bdi_total", "masq_total", "label"}
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c.lower() not in exclude]


# ── Synthetic data generator ────────────────────────────
def generate_synthetic_data(n=262, seed=42):
    """Generate realistic synthetic fMRI data for demo mode."""
    rng = np.random.default_rng(seed)

    # ds000030 demographics: healthy, schizophrenia, bipolar, ADHD
    # For our purposes, split by anhedonia severity
    n_high = int(n * 0.35)  # high anhedonia
    n_low = n - n_high

    data = {"participant_id": [f"sub-{10000+i}" for i in range(n)]}

    # NAcc BOLD — primary signal (reduced in anhedonic individuals)
    data["nacc_bold_mean"] = np.concatenate([
        rng.normal(0.42, 0.18, n_high),   # high anhedonia → reduced NAcc
        rng.normal(0.68, 0.20, n_low),     # low anhedonia → normal NAcc
    ])
    data["nacc_bold_peak"] = data["nacc_bold_mean"] * rng.uniform(1.3, 2.0, n)
    data["nacc_bold_variance"] = np.abs(rng.normal(0.08, 0.03, n))

    # Other ROIs
    data["caudate_bold_mean"] = np.concatenate([
        rng.normal(0.38, 0.15, n_high),
        rng.normal(0.52, 0.17, n_low),
    ])
    data["putamen_bold_mean"] = np.concatenate([
        rng.normal(0.50, 0.16, n_high),
        rng.normal(0.60, 0.18, n_low),
    ])
    data["vmpfc_bold_mean"] = np.concatenate([
        rng.normal(0.30, 0.14, n_high),
        rng.normal(0.40, 0.16, n_low),
    ])

    # BART task conditions
    data["nacc_gain_bold"] = np.concatenate([
        rng.normal(0.55, 0.20, n_high),
        rng.normal(0.85, 0.22, n_low),
    ])
    data["nacc_loss_bold"] = np.concatenate([
        rng.normal(0.35, 0.18, n_high),
        rng.normal(0.45, 0.19, n_low),
    ])
    data["nacc_neutral_bold"] = np.concatenate([
        rng.normal(0.25, 0.14, n_high),
        rng.normal(0.28, 0.15, n_low),
    ])

    # Framewise displacement (motion QC)
    data["mean_fd"] = rng.exponential(0.15, n) + 0.05

    # Anhedonia score (MASQ subscale)
    data["anhedonia_score"] = np.concatenate([
        rng.integers(ANHEDONIA_THRESHOLD, 50, n_high),
        rng.integers(10, ANHEDONIA_THRESHOLD, n_low),
    ])

    data["depressed"] = (data["anhedonia_score"] >= ANHEDONIA_THRESHOLD).astype(int)

    # Demographics
    data["age"] = rng.integers(21, 50, n)
    data["gender"] = rng.choice(["M", "F"], n, p=[0.52, 0.48])

    # Diagnosis (ds000030 has multi-group)
    data["diagnosis"] = np.concatenate([
        rng.choice(["SCHZ", "BPLR", "ADHD"], n_high, p=[0.45, 0.35, 0.20]),
        rng.choice(["CTRL", "ADHD"], n_low, p=[0.80, 0.20]),
    ])

    return pd.DataFrame(data)


# ── Figure 1: NAcc Activation by Group ─────────────────
def plot_nacc_activation(df, out_path):
    """Violin + swarm plot of NAcc BOLD activation by group."""
    plot_df = df[["depressed", "nacc_bold_mean"]].dropna().copy()
    group_map = {0: "Low Anhedonia", 1: "High Anhedonia"}
    plot_df["group"] = plot_df["depressed"].map(group_map)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: violin + strip
    ax = axes[0]
    sns.violinplot(data=plot_df, x="group", y="nacc_bold_mean", ax=ax,
                   order=["Low Anhedonia", "High Anhedonia"],
                   palette={"Low Anhedonia": TEAL, "High Anhedonia": CARDINAL},
                   inner="box", cut=0, linewidth=0.8)
    sns.stripplot(data=plot_df, x="group", y="nacc_bold_mean", ax=ax,
                  order=["Low Anhedonia", "High Anhedonia"],
                  color=DARK_GRAY, alpha=0.2, size=3, jitter=True)

    # Stats
    dep = plot_df[plot_df["depressed"] == 1]["nacc_bold_mean"]
    norm = plot_df[plot_df["depressed"] == 0]["nacc_bold_mean"]
    d = cohens_d(norm, dep)
    t, p = stats.ttest_ind(norm, dep, equal_var=False)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    ax.set_title(f"Nucleus Accumbens BOLD Activation\n"
                 f"Cohen's d = {d:.2f}, p {stars}", color=DARK_GRAY)
    ax.set_ylabel("Mean BOLD Signal (β)")
    ax.set_xlabel("")
    sns.despine(ax=ax)

    # Right: effect size bar for all ROIs
    ax2 = axes[1]
    roi_effects = []
    for col in ROI_METRICS:
        if col in df.columns:
            d_vals = df[df["depressed"] == 1][col].dropna()
            n_vals = df[df["depressed"] == 0][col].dropna()
            if len(d_vals) > 1 and len(n_vals) > 1:
                d_val = cohens_d(n_vals, d_vals)
                roi_effects.append({
                    "ROI": DISPLAY_NAMES.get(col, col),
                    "d": d_val, "|d|": abs(d_val)
                })

    if roi_effects:
        eff_df = pd.DataFrame(roi_effects).sort_values("|d|", ascending=True)
        colors = [CARDINAL if d > 0 else TEAL for d in eff_df["d"]]
        ax2.barh(eff_df["ROI"], eff_df["d"], color=colors, edgecolor="white", height=0.6)
        ax2.axvline(0, color=DARK_GRAY, linewidth=0.8)
        ax2.set_xlabel("Cohen's d")
        ax2.set_title("Effect Sizes by ROI", color=DARK_GRAY)
        sns.despine(ax=ax2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [1/5] NAcc activation → {os.path.basename(out_path)}")


# ── Figure 2: BOLD by Task Condition ───────────────────
def plot_bold_by_condition(df, out_path):
    """Grouped bar plot of NAcc BOLD by BART condition × group."""
    conditions = {
        "nacc_gain_bold": "Gain",
        "nacc_loss_bold": "Loss",
        "nacc_neutral_bold": "Neutral",
    }

    avail = {k: v for k, v in conditions.items() if k in df.columns}
    if not avail:
        print("  [2/5] Skipped condition plot (no condition columns found)")
        return

    group_map = {0: "Low Anhedonia", 1: "High Anhedonia"}
    rows = []
    for col, cond_name in avail.items():
        for _, row in df[["depressed", col]].dropna().iterrows():
            rows.append({
                "Condition": cond_name,
                "BOLD Signal (β)": row[col],
                "Group": group_map[int(row["depressed"])],
            })
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="Condition", y="BOLD Signal (β)", hue="Group",
                palette={"Low Anhedonia": TEAL, "High Anhedonia": CARDINAL},
                order=["Gain", "Loss", "Neutral"], ax=ax,
                capsize=0.05, errwidth=1.5, edgecolor="white")

    # Add significance brackets for Gain condition
    if "nacc_gain_bold" in df.columns:
        dep_gain = df[df["depressed"] == 1]["nacc_gain_bold"].dropna()
        norm_gain = df[df["depressed"] == 0]["nacc_gain_bold"].dropna()
        if len(dep_gain) > 1 and len(norm_gain) > 1:
            t, p = stats.ttest_ind(norm_gain, dep_gain, equal_var=False)
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            y_max = plot_df[plot_df["Condition"] == "Gain"]["BOLD Signal (β)"].max()
            ax.annotate(stars, xy=(0, y_max + 0.08), fontsize=14,
                        ha="center", fontweight="bold", color=CARDINAL)

    ax.set_title("NAcc BOLD Response by BART Task Condition",
                 fontweight="bold", color=DARK_GRAY)
    ax.set_ylabel("Mean BOLD Signal (β)")
    ax.legend(title="Group", framealpha=0.9)
    sns.despine(ax=ax)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [2/5] BOLD by condition → {os.path.basename(out_path)}")


# ── Figure 3: Correlation Scatter ──────────────────────
def plot_correlation(df, out_path):
    """Scatter plot of NAcc activation vs anhedonia score with regression."""
    if "nacc_bold_mean" not in df.columns or "anhedonia_score" not in df.columns:
        print("  [3/5] Skipped correlation (missing columns)")
        return

    plot_df = df[["nacc_bold_mean", "anhedonia_score", "depressed"]].dropna().copy()
    group_map = {0: "Low Anhedonia", 1: "High Anhedonia"}
    plot_df["Group"] = plot_df["depressed"].map(group_map)

    fig, ax = plt.subplots(figsize=(8, 6))

    for grp, color, name in [(0, TEAL, "Low Anhedonia"), (1, CARDINAL, "High Anhedonia")]:
        mask = plot_df["depressed"] == grp
        ax.scatter(plot_df[mask]["anhedonia_score"], plot_df[mask]["nacc_bold_mean"],
                   c=color, alpha=0.5, s=35, label=name,
                   edgecolors="white", linewidths=0.3)

    # Overall regression line
    x = plot_df["anhedonia_score"].values
    y = plot_df["nacc_bold_mean"].values
    r, p = stats.pearsonr(x, y)
    z = np.polyfit(x, y, 1)
    px = np.linspace(x.min(), x.max(), 100)
    ax.plot(px, np.polyval(z, px), color=DARK_GRAY, linewidth=2, linestyle="--",
            alpha=0.8, label=f"r = {r:.3f}, p < {max(p, 0.001):.3f}")

    ax.set_xlabel("Anhedonia Score (MASQ Subscale)")
    ax.set_ylabel("NAcc Mean BOLD Signal (β)")
    ax.set_title(f"NAcc Activation vs Anhedonia Severity\n"
                 f"(ds000030, n = {len(plot_df)})",
                 fontweight="bold", color=DARK_GRAY)
    ax.legend(framealpha=0.9, loc="upper right")
    sns.despine(ax=ax)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [3/5] Correlation scatter → {os.path.basename(out_path)}")
    return r, p


# ── Figure 4: Stats Table ─────────────────────────────
def plot_stats_table(df, out_path):
    """Table of group comparison statistics for all ROI metrics."""
    feat_cols = get_feature_cols(df)
    dep = df[df["depressed"] == 1]
    norm = df[df["depressed"] == 0]

    rows = []
    for col in feat_cols:
        d_vals = dep[col].dropna()
        n_vals = norm[col].dropna()
        if len(d_vals) < 2 or len(n_vals) < 2:
            continue
        d = cohens_d(n_vals, d_vals)
        t, p = stats.ttest_ind(n_vals, d_vals, equal_var=False)
        display = DISPLAY_NAMES.get(col, col)
        rows.append({
            "Metric": display,
            "High Anh. M (SD)": f"{d_vals.mean():.3f} ({d_vals.std():.3f})",
            "Low Anh. M (SD)": f"{n_vals.mean():.3f} ({n_vals.std():.3f})",
            "Cohen's d": round(d, 3),
            "|d|": abs(d),
            "p-value": p,
        })

    stats_df = pd.DataFrame(rows).sort_values("|d|", ascending=False)
    stats_df["p-value"] = stats_df["p-value"].apply(
        lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    )
    stats_df["Sig."] = stats_df["|d|"].apply(
        lambda d: "***" if d > 0.8 else "**" if d > 0.5 else "*" if d > 0.2 else ""
    )
    display_df = stats_df[["Metric", "Low Anh. M (SD)", "High Anh. M (SD)",
                           "Cohen's d", "p-value", "Sig."]].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, max(4, len(display_df) * 0.45 + 1.5)))
    ax.axis("off")
    ax.set_title("Stream B: Group Comparison Statistics\n(ds000030 fMRI — BART Task)",
                 fontsize=14, fontweight="bold", color=DARK_GRAY, pad=20)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for j in range(len(display_df.columns)):
        cell = table[0, j]
        cell.set_facecolor(BLUE_NEURAL)
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            cell = table[i, j]
            cell.set_facecolor(LIGHT_GRAY if i % 2 == 0 else "white")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [4/5] Stats table → {os.path.basename(out_path)}")
    return display_df


# ── Figure 5: ROI Correlation Heatmap ──────────────────
def plot_roi_heatmap(df, out_path):
    """Correlation matrix of ROI metrics + anhedonia score."""
    cols = [c for c in ROI_METRICS if c in df.columns]
    if "anhedonia_score" in df.columns:
        cols.append("anhedonia_score")

    # Include condition columns if present
    for cond_col in ["nacc_gain_bold", "nacc_loss_bold", "nacc_neutral_bold"]:
        if cond_col in df.columns:
            cols.append(cond_col)

    if len(cols) < 3:
        print("  [5/5] Skipped heatmap (too few ROI columns)")
        return

    corr = df[cols].corr()
    clean_labels = [DISPLAY_NAMES.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                xticklabels=clean_labels, yticklabels=clean_labels,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                cbar_kws={"shrink": 0.8, "label": "Pearson r"})

    ax.set_title("ROI Metrics Correlation Matrix\n(Ventral Striatal + Anhedonia)",
                 fontsize=14, fontweight="bold", color=DARK_GRAY)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [5/5] ROI heatmap → {os.path.basename(out_path)}")


# ── Combined PDF ────────────────────────────────────────
def combine_to_pdf(fig_paths, pdf_path):
    """Merge all figure PNGs into a single PDF."""
    from PIL import Image as PILImage
    images = []
    for p in fig_paths:
        if os.path.exists(p):
            images.append(PILImage.open(p).convert("RGB"))
    if images:
        images[0].save(pdf_path, "PDF", save_all=True, append_images=images[1:],
                       resolution=300)
        print(f"\n  Combined PDF → {os.path.basename(pdf_path)}")


# ── Main ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate poster figures from Stream B fMRI feature CSV"
    )
    parser.add_argument("--input", "-i", help="Path to feature CSV")
    parser.add_argument("--demo", action="store_true",
                        help="Generate with synthetic data (no CSV needed)")
    parser.add_argument("--outdir", "-o", default=FIG_DIR,
                        help=f"Output directory (default: {FIG_DIR})")
    args = parser.parse_args()

    if not args.input and not args.demo:
        print("No input CSV provided. Use --demo for synthetic data test.")
        print("Usage: python3 generate_figures_streamB.py --input features.csv")
        print("       python3 generate_figures_streamB.py --demo")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    if args.demo:
        print("Running in DEMO mode with synthetic data (n=262)...\n")
        df = generate_synthetic_data(n=262)
    else:
        print(f"Loading: {args.input}\n")
        df = pd.read_csv(args.input)
        if "depressed" not in df.columns and "anhedonia_score" in df.columns:
            df["depressed"] = (df["anhedonia_score"] >= ANHEDONIA_THRESHOLD).astype(int)

    # Summary
    n_dep = (df["depressed"] == 1).sum()
    n_norm = (df["depressed"] == 0).sum()
    n_feat = len(get_feature_cols(df))
    print(f"  Subjects: {len(df)} total ({n_dep} high anhedonia, {n_norm} low anhedonia)")
    print(f"  Features: {n_feat} numeric columns detected")
    print(f"  Output:   {args.outdir}\n")

    # Generate figures
    paths = {}

    paths["nacc"] = os.path.join(args.outdir, "nacc_activation.png")
    plot_nacc_activation(df, paths["nacc"])

    paths["condition"] = os.path.join(args.outdir, "bold_by_condition.png")
    plot_bold_by_condition(df, paths["condition"])

    paths["correlation"] = os.path.join(args.outdir, "correlation_scatter.png")
    corr_result = plot_correlation(df, paths["correlation"])

    paths["stats"] = os.path.join(args.outdir, "stats_table.png")
    stats_df = plot_stats_table(df, paths["stats"])

    paths["heatmap"] = os.path.join(args.outdir, "roi_metrics_heatmap.png")
    plot_roi_heatmap(df, paths["heatmap"])

    # Combined PDF
    pdf_path = os.path.join(args.outdir, "all_figures_streamB.pdf")
    combine_to_pdf(list(paths.values()), pdf_path)

    # Final summary
    print(f"\n{'='*50}")
    print("Stream B Figure Generation Complete")
    print(f"{'='*50}")
    print(f"  N = {len(df)} ({n_dep} high anhedonia, {n_norm} low anhedonia)")
    if corr_result:
        r, p = corr_result
        print(f"  NAcc-Anhedonia correlation: r = {r:.3f}, p = {p:.4f}")

    if stats_df is not None and len(stats_df) > 0:
        print("\n  Top metrics by |Cohen's d|:")
        for _, row in stats_df.head(5).iterrows():
            metric = row["Metric"]
            d_val = row["Cohen's d"]
            sig = row["Sig."]
            print(f"    {metric:30s}  d = {d_val:+.3f}  {sig}")

    print(f"\n  Figures saved to: {args.outdir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
