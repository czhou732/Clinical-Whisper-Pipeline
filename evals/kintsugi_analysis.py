#!/usr/bin/env python3
"""What Kintsugi's released validation and test scores say about their model.

Data: KintsugiHealth/dam-dataset (Apache-2.0), the DAM 3.1 model's scores on
their validation and test sets with PHQ-9 / GAD-7 items, demographics and
audio-quality estimates (no audio). Four questions:

1. Their published threshold on test: sensitivity and specificity for
   PHQ-9 >= 10.
2. Abstaining: an indeterminate band chosen on validation (20% budget),
   reported on test once.
3. Bias: the same threshold's sensitivity, specificity and AUC by gender,
   ethnicity, age, income and language preference.
4. Construct (exploratory): does the depression score follow PHQ item 1
   (little interest or pleasure: anhedonia) or item 2 (feeling down) more?
   Spearman correlations and partial correlations controlling the other item.

Usage: python evals/kintsugi_analysis.py --out evals/reports/kintsugi_validation.md
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import calibration as cal  # noqa: E402

THRESHOLD = -0.6699  # their depression cut-off between "none" and "mild to moderate"


def load() -> dict[str, pd.DataFrame]:
    from huggingface_hub import snapshot_download

    d = snapshot_download("KintsugiHealth/dam-dataset", repo_type="dataset")
    return {Path(f).stem.split("-")[0]: pd.read_parquet(f) for f in glob.glob(d + "/data/*.parquet")}


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y controlling z (on ranks)."""
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)  # noqa: E731
    return float(np.corrcoef(res(rx, rz), res(ry, rz))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = load()
    val, test = data["validation"], data["test"]
    lines = ["# Kintsugi DAM 3.1 on its own released validation and test scores", "",
             f"Validation n = {len(val)}, test n = {len(test)}. Label: PHQ-9 >= 10. "
             "Source: KintsugiHealth/dam-dataset. Scores only, no audio.", ""]

    y_test = test["phq"] >= 10
    s_test = test["scores_depression"].to_numpy()
    single = cal.Band(THRESHOLD, THRESHOLD)
    m = cal.evaluate(single, s_test, y_test)
    lines += ["## 1. Their threshold on test", "",
              f"AUC {cal.auc(s_test, y_test):.3f}; at {THRESHOLD}: sensitivity {m['sensitivity']}, "
              f"specificity {m['specificity']}, prevalence {y_test.mean():.2f}.", ""]

    band = cal.fit_band(val["scores_depression"].to_numpy(), val["phq"] >= 10, budget=0.2)
    mb = cal.evaluate(band, s_test, y_test)
    lines += ["## 2. Allowing \"can't tell\" (band chosen on validation, 20% budget)", "",
              f"Band {band.low:.3f} to {band.high:.3f}. On test: sensitivity {mb['sensitivity']}, "
              f"specificity {mb['specificity']}, undecided {mb['indeterminate']:.0%}.", ""]

    lines += ["## 3. By group (their threshold, test set)", "",
              "| group | value | n | PHQ>=10 | sensitivity | specificity | AUC |", "|---|---|---|---|---|---|---|"]
    # The survey's labels vary only in line breaks between versions; merge those.
    test = test.assign(ethnicity=test["ethnicity"].str.split().str.join(" "))
    test = test.assign(age_band=pd.cut(test["age"], [0, 29, 44, 59, 120], labels=["18-29", "30-44", "45-59", "60+"]))
    flagged = []
    for col in ("gender", "ethnicity", "age_band", "income", "english_preferred"):
        for value, g in test.groupby(col, observed=True):
            if len(g) < 100:
                continue
            yg = g["phq"] >= 10
            mg = cal.evaluate(single, g["scores_depression"].to_numpy(), yg)
            a = cal.auc(g["scores_depression"].to_numpy(), yg)
            lines.append(f"| {col} | {value} | {len(g)} | {yg.mean():.2f} | {mg['sensitivity']} | "
                         f"{mg['specificity']} | {a:.3f} |" if a is not None else "")
            for k in ("sensitivity", "specificity"):
                if mg[k] is not None and abs(mg[k] - m[k]) >= 0.10:
                    flagged.append(f"{col} = {value}: {k} {mg[k]} vs {m[k]} overall")
    lines += ["", "Groups at least 10 points from the overall sensitivity or specificity:", ""]
    lines += [f"- {f}" for f in flagged] or ["- none"]
    lines.append("")

    both = test.dropna(subset=["phq1", "phq2"])
    s = both["scores_depression"]
    r1, r2 = spearmanr(s, both["phq1"]).statistic, spearmanr(s, both["phq2"]).statistic
    p1, p2 = partial_spearman(s, both["phq1"], both["phq2"]), partial_spearman(s, both["phq2"], both["phq1"])
    lines += ["## 4. Anhedonia or low mood? (exploratory)", "",
              f"Spearman with PHQ item 1 (interest/pleasure) {r1:.3f}; with item 2 (feeling down) {r2:.3f}.",
              f"Partial, controlling the other item: item 1 {p1:.3f}; item 2 {p2:.3f}. n = {len(both)}.", "",
              "Exploratory, not pre-registered; one model on its developer's data.", ""]

    noise = test.dropna(subset=["noi_pred"])
    terc = pd.qcut(noise["noi_pred"], 3, labels=["noisiest third", "middle", "cleanest third"])
    lines += ["## 5. Recording quality (their noise estimate, tertiles)", ""]
    for value, g in noise.groupby(terc, observed=True):
        a = cal.auc(g["scores_depression"].to_numpy(), g["phq"] >= 10)
        lines.append(f"- {value}: AUC {a:.3f} (n {len(g)})")
    args.out.write_text("\n".join(x for x in lines if x is not None) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
