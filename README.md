# Deep-Fish-Bone

Deep learning-based multi-label segmentation of developing craniovertebral skeletal structures in zebrafish larvae from brightfield microscopy images.

This repository contains the source code, experimental configurations, fish-level data splits, evaluation scripts, and result summaries associated with the manuscript:

**Deep Learning-Based Multi-Label Segmentation of Developing Craniovertebral Skeletal Structures in Zebrafish Larvae: A Brightfield Microscopy Benchmark Dataset and Systematic Evaluation**

## Overview

The study addresses simultaneous segmentation of 25 developing craniovertebral skeletal structures in ventral-view brightfield microscopy images of 10-day-post-fertilization (dpf) zebrafish larvae.

The finalized dataset contains 190 annotation records from 34 biological fish, with 25 skeletal structures and WT, HET, and HOM genotypes. Complete fish-level separation is maintained between training, validation, and independent test sets.

Three controlled experimental configurations were evaluated:

1. **Baseline** — foreground-aware patch sampling.
2. **Anatomy-aware** — targeted patch sampling designed to increase exposure to the small and sparsely annotated Br2a and Br2b structures.
3. **Class-specific Br2** — anatomy-aware sampling combined with a class-specific objective for Br2a and Br2b designed to reduce false-positive predictions.

## Model and Training

All three configurations use U-Net++ with a ResNet34 encoder initialized with ImageNet-pretrained weights and 25 output channels.

Common settings are 1288 × 966 resized whole images, 512 × 512 training patches, batch size 2, 200 epochs, AdamW with learning rate 3e-4 and weight decay 1e-4, cosine annealing, and mixed-precision CUDA training.

The Baseline and Anatomy-aware configurations use the same equally weighted Dice and Focal loss. Class-specific Br2 retains the Focal component while using Dice overlap for 23 structures and a Tversky-based overlap term for Br2a and Br2b, with equally weighted regional and Focal components.

## Patch Sampling

Baseline uses 90% foreground-containing and 10% random patches.

Anatomy-aware uses 30% target-positive, 30% target-hard-negative, 30% general foreground, and 10% random patches.

Class-specific Br2 uses the same anatomy-aware sampling, so the Anatomy-aware versus Class-specific Br2 comparison isolates the class-specific loss formulation.

## Experimental Evaluation

Evaluation uses five stratified grouped folds at the biological-fish level. In each rotation, three folds are used for training, one for validation, and one as an independent test set.

The best checkpoint is selected by full-image validation Dice and evaluated once on the corresponding independent test set. Across five rotations, every annotation record appears in an independent test set exactly once, while records from the same biological fish remain in the same partition.

## Whole-Image Inference

Whole-image predictions are reconstructed with overlapping 512 × 512 windows at stride 256. Probabilities are averaged in overlapping regions and thresholded at 0.5. Reported segmentation metrics are calculated from reconstructed whole-image predictions.

## Evaluation Metrics

Structure-specific Dice is calculated on records containing a non-empty reference annotation for the corresponding structure.

A complementary structure-presence analysis evaluates whether structures are correctly identified as present or absent. Predicted maps are thresholded at 0.5 and connected predicted components smaller than 25 pixels are removed before presence-detection precision, recall, and F1 are calculated.

## Main Results

| Configuration | Mean Dice ± SD |
| --- | ---: |
| Baseline | 0.8004 ± 0.0185 |
| Anatomy-aware | 0.8072 ± 0.0194 |
| Class-specific Br2 | 0.7974 ± 0.0237 |

For Br2a, pooled structure-specific Dice increased from 0.511 with Baseline to 0.548 with Anatomy-aware sampling. For Br2b, Dice increased from 0.536 to 0.547.

Class-specific Br2 achieved Br2a and Br2b Dice scores of 0.512 and 0.497. Its main effect was instead observed in structure-presence identification. Relative to Anatomy-aware, false positives decreased from 67 to 22 for Br2a (67.2%) and from 86 to 24 for Br2b (72.1%). This was accompanied by higher presence-detection precision and F1 but lower recall.

Detailed canonical results are in `results/`.

## Repository Structure

```text
analysis/        Result and manuscript-table reproduction scripts
configs/         Final experimental configurations
datasets/        Dataset loading, metadata handling, and patch sampling
engine/          Training, validation, and canonical evaluation pipelines
figures/         Generated manuscript figures
losses/          Loss functions
metadata/        Canonical metadata and final fish-level evaluation splits
models/          Model definitions
results/         Canonical evaluation results
tables/          Generated manuscript tables
utils/           Supporting utilities
visualization/   Figure-generation and visualization scripts
```

Raw microscopy data and trained checkpoints are distributed separately through Zenodo rather than stored in Git.

## Dataset

The finalized dataset contains brightfield microscopy images, anatomical segmentation masks, and supporting full-body masks.

Canonical metadata: `metadata/metadata.csv`

Exact fish-level evaluation partitions: `metadata/final_cv_190/`

**Zenodo dataset:** DOI/link will be added after deposition.

## Trained Models

The model release contains 15 checkpoints: five Baseline, five Anatomy-aware, and five Class-specific Br2 checkpoints corresponding to the five evaluation rotations.

**Zenodo trained models:** DOI/link will be added after deposition.

## Installation

Python 3.8 or later is required.

```bash
pip install -r requirements.txt
```

The repository retains the dependency specification used for the training pipeline.

## Reproducing the Results

Canonical result summaries:

```bash
python analysis/reproduce_paper_results.py
```

Dataset structure statistics:

```bash
python analysis/table2_structure_statistics.py
```

Structure-specific Dice table and figure:

```bash
python analysis/table4_per_structure_dice.py
python visualization/figure4_per_structure_dice.py
```

Br2 structure-presence table:

```bash
python analysis/table5_br2_presence.py
```

`engine/evaluate_canonical_final.py` was used to reevaluate the frozen trained checkpoints against the finalized canonical annotations without retraining.

## Data and Model Availability

GitHub provides the source code, final configurations, canonical metadata, exact evaluation splits, analysis and visualization scripts, lightweight evaluation results, and generated manuscript tables and figures.

Two larger research artifacts are distributed separately through Zenodo:

1. **Dataset record** — microscopy images, anatomical masks, supporting full-body masks, and dataset documentation.
2. **Model record** — 15 final checkpoints with associated configurations, split information, and evaluation metadata.

Zenodo DOI links will be added after the records are finalized.

## Citation

If you use the dataset, code, or trained models, please cite the associated manuscript:

> Navdeep Kumar, Ratish Raman, Marc Muller, Raphael Maree, and Pierre Geurts.  
> *Deep Learning-Based Multi-Label Segmentation of Developing Craniovertebral Skeletal Structures in Zebrafish Larvae: A Brightfield Microscopy Benchmark Dataset and Systematic Evaluation.*

Publication details and DOI will be added after publication.

## Authors

Navdeep Kumar, Ratish Raman, Marc Muller, Raphael Maree, and Pierre Geurts.

University of Liège, Belgium.
