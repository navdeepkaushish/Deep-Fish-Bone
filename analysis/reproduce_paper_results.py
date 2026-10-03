from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

EXPERIMENTS = {
    "baseline_dice_focal": "Baseline",
    "anatomy_aware_dice_focal": "Anatomy-aware",
    "class_specific_br2": "Class-specific Br2",
}


def aggregate_dice(exp_dir):
    frames = []

    for fold in range(5):
        path = exp_dir / "folds" / f"fold_{fold}" / "test_per_structure_dice.csv"
        df = pd.read_csv(path)
        df["fold"] = fold
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    rows = []

    for structure, group in all_df.groupby("structure", sort=False):
        count = group["test_count"].sum()

        if count > 0:
            pooled = np.average(
                group["test_dice"],
                weights=group["test_count"],
            )
        else:
            pooled = np.nan

        rows.append({
            "structure": structure,
            "test_dice": pooled,
            "test_count": int(count),
        })

    return pd.DataFrame(rows)


def aggregate_presence(exp_dir):
    frames = []

    for fold in range(5):
        path = exp_dir / "folds" / f"fold_{fold}" / "test_presence_summary.csv"
        df = pd.read_csv(path)
        df["fold"] = fold
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    rows = []

    for structure, group in all_df.groupby("structure", sort=False):

        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())

        gt_present = tp + fn
        pred_present = tp + fp
        total_errors = fp + fn

        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan

        if (
            not np.isnan(precision)
            and not np.isnan(recall)
            and (precision + recall) > 0
        ):
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = np.nan

        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        rows.append({
            "structure": structure,
            "gt_present": gt_present,
            "pred_present": pred_present,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total_errors": total_errors,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
        })

    return pd.DataFrame(rows)


def main():

    print("=" * 72)
    print("REPRODUCING POOLED INDEPENDENT-TEST RESULTS")
    print("=" * 72)

    comparison = []

    for folder, label in EXPERIMENTS.items():

        exp_dir = RESULTS / folder

        dice = aggregate_dice(exp_dir)
        presence = aggregate_presence(exp_dir)

        dice_path = exp_dir / "pooled_per_structure_dice.csv"
        presence_path = exp_dir / "pooled_presence_summary.csv"

        dice.to_csv(dice_path, index=False)
        presence.to_csv(presence_path, index=False)

        print(f"\n{label}")
        print("-" * len(label))
        print(f"Saved: {dice_path.relative_to(ROOT)}")
        print(f"Saved: {presence_path.relative_to(ROOT)}")

        br2 = dice[dice["structure"].isin(["Br2a", "Br2b"])]

        print("\nBr2 Dice:")
        print(br2.to_string(index=False))

        br2_presence = presence[
            presence["structure"].isin(["Br2a", "Br2b"])
        ]

        print("\nBr2 presence:")
        print(
            br2_presence[
                [
                    "structure",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                ]
            ].to_string(index=False)
        )

        temp = dice[["structure", "test_dice"]].copy()
        temp = temp.rename(columns={"test_dice": label})
        comparison.append(temp)

    merged = comparison[0]

    for df in comparison[1:]:
        merged = merged.merge(df, on="structure", how="outer")

    comparison_path = RESULTS / "per_structure_dice_comparison.csv"
    merged.to_csv(comparison_path, index=False)

    print("\n" + "=" * 72)
    print(f"Saved comparison: {comparison_path.relative_to(ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
