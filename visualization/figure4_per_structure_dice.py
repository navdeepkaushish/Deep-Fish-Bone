from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "per_structure_dice_comparison.csv"
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

methods = [
    "Baseline",
    "Anatomy-aware",
    "Class-specific Br2",
]

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

y = np.arange(len(df))
height = 0.22

fig, ax = plt.subplots(figsize=(12, 13))

offsets = [-height, 0, height]

for method, offset in zip(methods, offsets):

    values = df[method].to_numpy()

    bars = ax.barh(
        y + offset,
        values,
        height,
        label=method,
    )

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )

ax.set_yticks(y)
ax.set_yticklabels(df["structure"])

# Put Br1a at the top, as in the manuscript figure.
ax.invert_yaxis()

ax.set_xlabel("Dice score")
ax.set_ylabel("Skeletal structure")

ax.set_xlim(0, 1.06)
ax.set_xticks(np.arange(0, 1.01, 0.1))

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.30,
)
ax.set_axisbelow(True)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=3,
    frameon=False,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

pdf = OUTDIR / "figure4_per_structure_dice.pdf"
png = OUTDIR / "figure4_per_structure_dice.png"

fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=600, bbox_inches="tight")

plt.close(fig)

print(f"Saved: {pdf.relative_to(ROOT)}")
print(f"Saved: {png.relative_to(ROOT)}")

print("\nBr2 values used:")
print(
    df[df["structure"].isin(["Br2a", "Br2b"])]
    .to_string(index=False)
)
