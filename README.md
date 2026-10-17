# IE643 Programming Challenge (2026)

Programming Challenge for **IE643 Deep Learning**: predict one of **11 class labels (0–10)** for every sample in a test set
that mixes clean and deformed (corrupted) samples.

## The challenge

- The training set is a union of **labelled and unlabelled** samples, and each sample is flagged as deformed or not.
- Only **CNNs and/or MLPs** may be used; any other kind of deep network makes the submission invalid.
- Scoring is a **weighted accuracy** with hidden class weights.
- The submission is a CSV with a header and one row per test sample:

  ```
  ID,Label
  1,7
  2,3
  ...
  ```

## Data

The competition data is **not included in this repository**. Download it from the competition page and place it like this:

```
IE643_Programming_Challenge/
├── ie-643-2026-programming-challenge/
│   ├── Train_Data.csv
│   ├── Test_Data.csv
│   ├── Supplementry_Data.csv
│   └── Sample_Submission.csv
└── pca_explained_variance.ipynb
```

| File | Rows | Columns |
|---|---|---|
| `Train_Data.csv` | 2076 | `sample_id`, `e0`–`e767` (768 features), `is_deformed`, `Class_Label` (`-1` = unlabelled) |
| `Test_Data.csv` | 490 | `sample_id`, `f0`–`f767` (768 features) |
| `Supplementry_Data.csv` | 2076 (+2 empty rows) | `sample_id`, `g0`–`g383` (384 features), train samples only |
| `Sample_Submission.csv` | 490 (+ blank rows) | `ID`, `Label` |

**The samples are not images.** Each one is a 768-dimensional continuous vector with no spatial structure: neighbouring features are uncorrelated at every offset. It looks like an embedding from a pretrained encoder.

Composition of the training set:

| | Clean | Deformed | Total |
|---|---|---|---|
| Labelled (D_A) | 136 | 1148 | 1284 |
| Unlabelled (D_B) | 72 | 720 | 792 |

The classes are imbalanced. Class 0 has 450 labelled samples, while class 10 has 40. Among clean samples, class 8 has only 3.

## Notebook: `pca_explained_variance.ipynb`

Everything currently lives in one notebook. Sections:

1. **Split** the training set into D_A (labelled) and D_B (unlabelled), with clean/deformed counts per class.
2. **Keep the clean labelled points** (136 samples).
3. **Normalize** each of the 768 dimensions with the clean points' mean and std, computed across all classes.
4. **PCA** on the clean points. The top 10 PCs explain 48.6% of the variance, and 118 PCs explain 99%.
5. **t-SNE** of the clean points on the top 118 PCs, coloured by class.
6. **t-SNE** with the deformed labelled points projected onto the same PCs.
7. **2-D embedding experiment:** can a non-linear map separate the clean classes?
   - **Models:** a linear map `R^d → R^2` (no hidden layer, no activation) vs an MLP (`d → 64 → 32 → 2`, GELU, dropout 0.3).
   - **Losses:** each model is trained with **NCA** and with a differentiable **soft silhouette**. Both weight every class equally.
   - **Baselines:** k-NN in the full PCA space, and the first 2 PCs.
   - **Cross-validation:** stratified 3-fold, repeated 5 times (15 splits). Train and validation keep the same class proportions.
   - **Leakage control:** normalization and PCA are re-fitted inside each fold on the training part only.
   - **Metrics:** k-NN accuracy and balanced accuracy (k = 1, 5) on held-out points, a paired MLP − linear comparison, learning curves, and 2-D plots.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook pca_explained_variance.ipynb
```

Start Jupyter from the repository root, so the relative data path `ie-643-2026-programming-challenge/` resolves.
The notebook was developed with Python 3.12, numpy 1.26, pandas 2.2, scikit-learn 1.8, PyTorch 2.12 and matplotlib 3.9. Section 7 runs on CPU in about 2 minutes.

## Status

Exploratory stage: understanding how separable the classes are before choosing the final CNN/MLP architecture.
Still to come: using the deformed and unlabelled samples, the supplementary features, and a submission pipeline.
