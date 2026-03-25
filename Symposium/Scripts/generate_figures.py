#!/usr/bin/env python3
"""
generate_figures.py — Auto-generate poster figures from Stream A feature CSV.

Usage:
    python3 generate_figures.py --input /path/to/stream_a_features.csv
    python3 generate_figures.py --demo   # synthetic data for testing

Outputs 4 publication-quality figures to Symposium/Figures/:
    1. violin_plots.png       — 6 key eGeMAPSv02 features by group
    2. pca_scatter.png        — 2D PCA of all features, colored by group
    3. stats_table.png        — Cohen's d + p-values for top features
    4. correlation_heatmap.png — Top 20 features correlation matrix

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
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

# USC brand
CARDINAL = "#990000"
GOLD = "#FFCC00"
DARK_GRAY = "#333333"
LIGHT_GRAY = "#F0F0F0"
TEAL = "#0E8A7D"

# Group colors
COLOR_DEPRESSED = CARDINAL
COLOR_NORMAL = TEAL

# PHQ-8 threshold (matches OSF pre-reg)
PHQ8_THRESHOLD = 10

# 6 key features for violin plots (eGeMAPSv02 functional names)
KEY_FEATURES = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "loudness_sma3_amean",
    "spectralFlux_sma3_amean",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "HNRdBACF_sma3nz_amean",
]

# Display names for axis labels
DISPLAY_NAMES = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": "F0 (semitones)",
    "loudness_sma3_amean": "Loudness",
    "spectralFlux_sma3_amean": "Spectral Flux",
    "jitterLocal_sma3nz_amean": "Jitter",
    "shimmerLocaldB_sma3nz_amean": "Shimmer (dB)",
    "HNRdBACF_sma3nz_amean": "HNR (dB)",
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


def clean_column_name(col):
    """Strip eGeMAPSv02 prefix and functionals suffix for matching."""
    c = col.strip()
    for prefix in ["egemapsv02_", "eGeMAPSv02_", "gemaps_"]:
        if c.lower().startswith(prefix.lower()):
            c = c[len(prefix):]
    return c


def find_feature_columns(df):
    """Map key feature short names to actual column names in the dataframe."""
    clean_map = {}
    for col in df.columns:
        cleaned = clean_column_name(col)
        clean_map[cleaned.lower()] = col

    mapping = {}
    for feat in KEY_FEATURES:
        key = feat.lower()
        if key in clean_map:
            mapping[feat] = clean_map[key]
        else:
            for cleaned_key, orig_col in clean_map.items():
                if feat.lower().replace("_", "") in cleaned_key.replace("_", ""):
                    mapping[feat] = orig_col
                    break

    return mapping


def get_feature_cols(df):
    """Return list of all numeric feature columns (exclude metadata)."""
    exclude = {"participant_id", "phq8_score", "depressed", "gender", "age",
               "phq_binary", "subject_id", "session", "label", "group",
               "word_count", "duration", "snr"}
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c.lower() not in exclude]


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


# ── Synthetic data generator ────────────────────────────
def generate_synthetic_data(n=150, seed=42):
    """Generate realistic synthetic eGeMAPSv02 data for demo mode."""
    rng = np.random.default_rng(seed)
    n_dep = int(n * 0.4)
    n_norm = n - n_dep

    feature_params = {
        "F0semitoneFrom27.5Hz_sma3nz_amean": (35.0, 6.0, -3.5),
        "loudness_sma3_amean": (0.45, 0.12, -0.08),
        "spectralFlux_sma3_amean": (0.028, 0.008, -0.005),
        "jitterLocal_sma3nz_amean": (0.022, 0.007, 0.003),
        "shimmerLocaldB_sma3nz_amean": (0.75, 0.20, 0.08),
        "HNRdBACF_sma3nz_amean": (8.5, 2.5, -1.2),
    }

    data = {"participant_id": [f"P{i:03d}" for i in range(n)]}

    # Generate the 6 key features with group differences
    for feat, (mu, sigma, delta) in feature_params.items():
        norm_vals = rng.normal(mu, sigma, n_norm)
        dep_vals = rng.normal(mu + delta, sigma * 1.1, n_dep)
        data[feat] = np.concatenate([dep_vals, norm_vals])

    # Generate 82 additional filler features (no real group difference)
    filler_names = [
        "F1frequency_sma3nz_amean", "F2frequency_sma3nz_amean",
        "F3frequency_sma3nz_amean", "F1bandwidth_sma3nz_amean",
        "F2bandwidth_sma3nz_amean", "F3bandwidth_sma3nz_amean",
        "F1amplitudeLogRelF0_sma3nz_amean", "F2amplitudeLogRelF0_sma3nz_amean",
        "F3amplitudeLogRelF0_sma3nz_amean", "logRelF0-H1-H2_sma3nz_amean",
        "logRelF0-H1-A3_sma3nz_amean", "mfcc1_sma3_amean", "mfcc2_sma3_amean",
        "mfcc3_sma3_amean", "mfcc4_sma3_amean",
    ]
    # Pad to 82 filler features
    while len(filler_names) < 82:
        filler_names.append(f"filler_feature_{len(filler_names):02d}")

    for feat in filler_names:
        data[feat] = rng.normal(0, 1, n)

    # PHQ-8 scores
    dep_phq = rng.integers(PHQ8_THRESHOLD, 24, n_dep)
    norm_phq = rng.integers(0, PHQ8_THRESHOLD, n_norm)
    data["phq8_score"] = np.concatenate([dep_phq, norm_phq])
    data["depressed"] = (data["phq8_score"] >= PHQ8_THRESHOLD).astype(int)

    # Gender
    data["gender"] = rng.choice(["Male", "Female"], n, p=[0.54, 0.46])

    return pd.DataFrame(data)


# ── Figure 1: Violin Plots ─────────────────────────────
def plot_violins(df, feat_map, out_path):
    """6-panel violin plots of key features by depression group."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Acoustic Feature Distributions: Depressed vs Normal",
                 fontsize=16, fontweight="bold", color=DARK_GRAY, y=0.98)

    palette = {0: TEAL, 1: CARDINAL}
    group_labels = {0: "Normal", 1: "Depressed"}
    found_features = []

    for idx, feat_key in enumerate(KEY_FEATURES):
        ax = axes[idx // 3][idx % 3]
        col = feat_map.get(feat_key)

        if col is None:
            ax.text(0.5, 0.5, f"{feat_key}\n(not found)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
            ax.set_title(DISPLAY_NAMES.get(feat_key, feat_key), fontsize=11)
            continue

        found_features.append(feat_key)
        plot_df = df[["depressed", col]].dropna().copy()
        plot_df["group"] = plot_df["depressed"].map(group_labels)

        sns.violinplot(data=plot_df, x="group", y=col, ax=ax,
                       order=["Normal", "Depressed"],
                       palette={"Normal": TEAL, "Depressed": CARDINAL},
                       inner="box", cut=0, linewidth=0.8)

        # Stats annotation
        dep = plot_df[plot_df["depressed"] == 1][col]
        norm = plot_df[plot_df["depressed"] == 0][col]
        if len(dep) > 1 and len(norm) > 1:
            t, p = stats.ttest_ind(dep, norm, equal_var=False)
            d = cohens_d(dep, norm)
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.set_title(f"{DISPLAY_NAMES.get(feat_key, feat_key)}\n"
                         f"d={d:.2f}, p{stars}",
                         fontsize=10, color=DARK_GRAY)
        else:
            ax.set_title(DISPLAY_NAMES.get(feat_key, feat_key), fontsize=10)

        ax.set_xlabel("")
        ax.set_xticklabels(["Normal", "Depressed"])
        ax.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [1/4] Violin plots → {os.path.basename(out_path)}")
    return found_features


# ── Figure 2: PCA Scatter ──────────────────────────────
def plot_pca(df, out_path):
    """2D PCA of all numeric features, colored by depression group."""
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].dropna(axis=1, how="all").fillna(0)
    valid_idx = X.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = df.loc[valid_idx, "depressed"].values

    for grp, color, name in [(0, TEAL, "Normal"), (1, CARDINAL, "Depressed")]:
        mask = labels == grp
        ax.scatter(Z[mask, 0], Z[mask, 1], c=color, alpha=0.55,
                   s=40, label=name, edgecolors="white", linewidths=0.3)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title(f"PCA of {len(feat_cols)} eGeMAPSv02 Features\n"
                 f"Depressed vs Normal (DAIC-WOZ, n = {len(valid_idx)})",
                 fontsize=13, fontweight="bold", color=DARK_GRAY)
    ax.legend(framealpha=0.9, loc="upper right")

    sns.despine(ax=ax)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [2/4] PCA scatter  → {os.path.basename(out_path)}")
    return pca.explained_variance_ratio_


# ── Figure 3: Stats Table ──────────────────────────────
def plot_stats_table(df, out_path):
    """Table image of top features ranked by Cohen's d."""
    feat_cols = get_feature_cols(df)
    dep = df[df["depressed"] == 1]
    norm = df[df["depressed"] == 0]

    rows = []
    for col in feat_cols:
        d_vals = dep[col].dropna()
        n_vals = norm[col].dropna()
        if len(d_vals) < 2 or len(n_vals) < 2:
            continue
        d = cohens_d(d_vals, n_vals)
        t, p = stats.ttest_ind(d_vals, n_vals, equal_var=False)
        rows.append({
            "Feature": clean_column_name(col),
            "Cohen's d": round(d, 3),
            "|d|": abs(d),
            "p-value": p,
            "Depressed M": round(d_vals.mean(), 3),
            "Normal M": round(n_vals.mean(), 3),
        })

    stats_df = pd.DataFrame(rows).sort_values("|d|", ascending=False).head(15)
    stats_df["p-value"] = stats_df["p-value"].apply(
        lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    )
    stats_df["Sig."] = stats_df["|d|"].apply(
        lambda d: "***" if d > 0.8 else "**" if d > 0.5 else "*" if d > 0.2 else ""
    )
    display_df = stats_df[["Feature", "Cohen's d", "p-value", "Depressed M",
                           "Normal M", "Sig."]].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(4, len(display_df) * 0.4 + 1.5)))
    ax.axis("off")
    ax.set_title("Top Features by Effect Size (Cohen's d)",
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

    # Style header
    for j in range(len(display_df.columns)):
        cell = table[0, j]
        cell.set_facecolor(CARDINAL)
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            cell = table[i, j]
            cell.set_facecolor(LIGHT_GRAY if i % 2 == 0 else "white")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [3/4] Stats table  → {os.path.basename(out_path)}")
    return display_df


# ── Figure 4: Correlation Heatmap ──────────────────────
def plot_heatmap(df, out_path):
    """Correlation heatmap of top 20 features by effect size."""
    feat_cols = get_feature_cols(df)
    dep = df[df["depressed"] == 1]
    norm = df[df["depressed"] == 0]

    effects = {}
    for col in feat_cols:
        d_vals = dep[col].dropna()
        n_vals = norm[col].dropna()
        if len(d_vals) > 1 and len(n_vals) > 1:
            effects[col] = abs(cohens_d(d_vals, n_vals))

    top20 = sorted(effects, key=effects.get, reverse=True)[:20]
    if len(top20) < 3:
        print("  [4/4] Skipped heatmap (too few features)")
        return

    corr = df[top20].corr()
    clean_labels = [clean_column_name(c)[:25] for c in top20]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                xticklabels=clean_labels, yticklabels=clean_labels,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"})

    ax.set_title("Feature Correlation Matrix (Top 20 by Effect Size)",
                 fontsize=14, fontweight="bold", color=DARK_GRAY)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [4/4] Heatmap      → {os.path.basename(out_path)}")


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
        description="Generate poster figures from Stream A feature CSV"
    )
    parser.add_argument("--input", "-i", help="Path to feature CSV")
    parser.add_argument("--demo", action="store_true",
                        help="Generate with synthetic data (no CSV needed)")
    parser.add_argument("--outdir", "-o", default=FIG_DIR,
                        help=f"Output directory (default: {FIG_DIR})")
    args = parser.parse_args()

    if not args.input and not args.demo:
        print("No input CSV provided. Use --demo for synthetic data test.")
        print("Usage: python3 generate_figures.py --input features.csv")
        print("       python3 generate_figures.py --demo")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    if args.demo:
        print("Running in DEMO mode with synthetic data (n=150)...\n")
        df = generate_synthetic_data(n=150)
    else:
        print(f"Loading: {args.input}\n")
        df = pd.read_csv(args.input)
        if "depressed" not in df.columns and "phq8_score" in df.columns:
            df["depressed"] = (df["phq8_score"] >= PHQ8_THRESHOLD).astype(int)

    # Summary
    n_dep = (df["depressed"] == 1).sum()
    n_norm = (df["depressed"] == 0).sum()
    n_feat = len(get_feature_cols(df))
    print(f"  Subjects: {len(df)} total ({n_dep} depressed, {n_norm} normal)")
    print(f"  Features: {n_feat} numeric columns detected")
    print(f"  Output:   {args.outdir}\n")

    # Find feature column mappings
    feat_map = find_feature_columns(df)
    missing = [f for f in KEY_FEATURES if f not in feat_map]
    if missing:
        print(f"  Warning: {len(missing)} key features not found: {missing}\n")

    # Generate figures
    paths = {}
    paths["violin"] = os.path.join(args.outdir, "violin_plots.png")
    plot_violins(df, feat_map, paths["violin"])

    paths["pca"] = os.path.join(args.outdir, "pca_scatter.png")
    var_explained = plot_pca(df, paths["pca"])

    paths["stats"] = os.path.join(args.outdir, "stats_table.png")
    stats_df = plot_stats_table(df, paths["stats"])

    paths["heatmap"] = os.path.join(args.outdir, "correlation_heatmap.png")
    plot_heatmap(df, paths["heatmap"])

    # Combined PDF
    pdf_path = os.path.join(args.outdir, "all_figures.pdf")
    combine_to_pdf(list(paths.values()), pdf_path)

    # Final summary
    print(f"\n{'='*50}")
    print("Poster Figure Generation Complete")
    print(f"{'='*50}")
    print(f"  N = {len(df)} ({n_dep} depressed, {n_norm} normal)")
    print(f"  PC1 = {var_explained[0]*100:.1f}%, PC2 = {var_explained[1]*100:.1f}%")

    if stats_df is not None and len(stats_df) > 0:
        print("\n  Top 5 features by |Cohen's d|:")
        for _, row in stats_df.head(5).iterrows():
            feat_name = row["Feature"]
            d_val = row["Cohen's d"]
            sig = row["Sig."]
            print(f"    {feat_name:40s}  d = {d_val:+.3f}  {sig}")

    print(f"\n  Figures saved to: {args.outdir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
