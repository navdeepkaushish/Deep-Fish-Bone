from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "tables" / "table5_br2_presence_performance.tex"

experiments = [
    ("baseline_dice_focal", "Baseline Dice+Focal"),
    ("anatomy_aware_dice_focal", "Anatomy-aware Dice+Focal"),
    ("class_specific_br2", "Class-specific Br2"),
]

structures = ["Br2a", "Br2b"]

data = {}

for folder, label in experiments:
    df = pd.read_csv(
        RESULTS / folder / "pooled_presence_summary.csv"
    )

    data[label] = {}

    for structure in structures:
        row = df[df["structure"] == structure]

        if len(row) != 1:
            raise RuntimeError(
                f"Expected one {structure} row for {folder}"
            )

        data[label][structure] = row.iloc[0]

# Canonical annotation sanity checks.
for label in data:
    assert int(data[label]["Br2a"]["gt_present"]) == 63
    assert int(data[label]["Br2b"]["gt_present"]) == 67

metrics = {
    "fp": "min",
    "fn": "min",
    "precision": "max",
    "recall": "max",
    "f1": "max",
}

best = {}

for structure in structures:
    for metric, direction in metrics.items():

        vals = [
            float(data[label][structure][metric])
            for _, label in experiments
        ]

        best[(structure, metric)] = (
            min(vals) if direction == "min" else max(vals)
        )

def format_value(structure, metric, value, decimals=None):

    value = float(value)

    if decimals is None:
        text = str(int(value))
    else:
        text = f"{value:.{decimals}f}"

    if np.isclose(
        value,
        best[(structure, metric)],
        rtol=0,
        atol=1e-12,
    ):
        return rf"\textbf{{{text}}}"

    return text

lines = [
r"\begin{table}[htbp]",
r"\centering",
r"\caption{Structure-presence identification performance for Br2a and Br2b across the three experimental configurations. TP, FP, and FN denote pooled true-positive, false-positive, and false-negative presence predictions across the five independent fish-level test folds. Precision, recall, and F1 are presence-detection metrics.}",
r"\label{tab:br2_presence}",
r"\small",
r"\begin{tabularx}{\textwidth}{llrrrrrr}",
r"\hline",
r"\textbf{Structure} &",
r"\textbf{Configuration} &",
r"\textbf{TP} &",
r"\textbf{FP} &",
r"\textbf{FN} &",
r"\textbf{Precision} &",
r"\textbf{Recall} &",
r"\textbf{F1} \\",
r"\hline",
"",
]

for structure in structures:

    for i, (_, label) in enumerate(experiments):

        r = data[label][structure]

        structure_cell = structure if i == 0 else ""

        tp = str(int(r["tp"]))
        fp = format_value(structure, "fp", r["fp"])
        fn = format_value(structure, "fn", r["fn"])

        precision = format_value(
            structure, "precision", r["precision"], 3
        )
        recall = format_value(
            structure, "recall", r["recall"], 3
        )
        f1 = format_value(
            structure, "f1", r["f1"], 3
        )

        lines += [
            f"{structure_cell} &",
            f"{label} &",
            f"{tp} & {fp} & {fn} &",
            f"{precision} & {recall} & {f1} \\\\",
            "",
        ]

    lines += [r"\hline", ""]

lines += [
r"\end{tabularx}",
r"\end{table}",
]

OUTPUT.write_text("\n".join(lines) + "\n")

print(f"Saved: {OUTPUT.relative_to(ROOT)}")
print()
print(OUTPUT.read_text())
