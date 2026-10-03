from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "per_structure_dice_comparison.csv"
OUTPUT = ROOT / "tables" / "table4_per_structure_performance.tex"

df = pd.read_csv(INPUT)

methods = ["Baseline", "Anatomy-aware", "Class-specific Br2"]

expected = [
    "Br1a", "Br1b", "Br2a", "Br2b",
    "Cb1", "Cb2", "Ch1", "Ch2",
    "Cl1", "Cl2", "D1", "D2",
    "En1", "En2", "Hm1", "Hm2",
    "M1", "M2", "N",
    "Oc1", "Oc2", "Op1", "Op2",
    "P", "VC",
]

if df["structure"].tolist() != expected:
    raise RuntimeError("Unexpected structure order.")

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Structure-specific Dice scores across the three experimental configurations. The best performance for each structure is highlighted in bold.}",
    r"\label{tab:per_structure_dice}",
    r"\begin{tabular}{lccc}",
    r"\hline",
    r"\textbf{Structure} & \textbf{Baseline} & \textbf{Anatomy-aware} & \textbf{Class-specific Br2} \\",
    r"\hline",
]

for _, row in df.iterrows():
    values = [float(row[m]) for m in methods]
    best = max(values)

    cells = []
    for value in values:
        text = f"{value:.3f}"

        # Bold based on the actual unrounded value.
        if np.isclose(value, best, rtol=0, atol=1e-12):
            text = rf"\textbf{{{text}}}"

        cells.append(text)

    lines.append(
        f"{row['structure']} & "
        f"{cells[0]} & {cells[1]} & {cells[2]} \\\\"
    )

lines += [
    r"\hline",
    r"\end{tabular}",
    r"\end{table}",
]

OUTPUT.write_text("\n".join(lines) + "\n")

print(f"Saved: {OUTPUT.relative_to(ROOT)}")
print()
print(OUTPUT.read_text())
