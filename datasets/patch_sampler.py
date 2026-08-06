#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, asdict
import random
import numpy as np

@dataclass(frozen=True)
class SamplePlan:
    requested_mode: str
    actual_mode: str
    target_structure: str
    source_index: int
    centre_x: float | None
    centre_y: float | None
    localization_method: str
    fallback: int = 0
    def to_dict(self):
        return asdict(self)

class AnatomyAwarePlanner:
    MODES = ("target_positive","target_hard_negative","general_foreground","random")

    def __init__(self, records, target_structures, probabilities, seed=42):
        self.records = records
        self.targets = list(target_structures)
        self.probabilities = {k: float(probabilities[k]) for k in self.MODES}
        if not np.isclose(sum(self.probabilities.values()), 1.0):
            raise ValueError("Sampler probabilities must sum to 1.")
        self.rng = random.Random(seed)
        self.positive = {
            t: [i for i,r in enumerate(records) if r["targets"][t]["present"]]
            for t in self.targets
        }
        self.negative = {
            t: [i for i,r in enumerate(records) if not r["targets"][t]["present"]]
            for t in self.targets
        }
        self.foreground = [i for i,r in enumerate(records) if r["present_structures"]]
        for t in self.targets:
            if not self.positive[t] or not self.negative[t]:
                raise RuntimeError(f"Target {t} requires positive and negative training images.")

    def _mode(self):
        x, c = self.rng.random(), 0.0
        for mode in self.MODES:
            c += self.probabilities[mode]
            if x <= c:
                return mode
        return "random"

    def make_plan(self):
        mode = self._mode()
        if mode == "target_positive":
            target = self.rng.choice(self.targets)
            idx = self.rng.choice(self.positive[target])
            x, y = self.records[idx]["targets"][target]["centroid"]
            return SamplePlan(mode, mode, target, idx, x, y, "ground_truth_target_centroid")

        if mode == "target_hard_negative":
            target = self.rng.choice(self.targets)
            idx = self.rng.choice(self.negative[target])
            info = self.records[idx]["targets"][target]
            if info["expected_centroid"] is not None:
                x, y = info["expected_centroid"]
                return SamplePlan(mode, mode, target, idx, x, y, info["localization_method"])
            # Final safety fallback only; cache construction should normally prevent this.
            idx = self.rng.choice(self.foreground)
            structure = self.rng.choice(self.records[idx]["present_structures"])
            x, y = self.records[idx]["structure_centroids"][structure]
            return SamplePlan(mode, "general_foreground", structure, idx, x, y,
                              "missing_anatomical_anchors", 1)

        if mode == "general_foreground":
            idx = self.rng.choice(self.foreground)
            structure = self.rng.choice(self.records[idx]["present_structures"])
            x, y = self.records[idx]["structure_centroids"][structure]
            return SamplePlan(mode, mode, structure, idx, x, y,
                              "class_uniform_foreground_centroid")

        idx = self.rng.randrange(len(self.records))
        return SamplePlan(mode, mode, "", idx, None, None, "uniform_random_crop")
