Here's a "highest-likelihood to move the needle" plan, ordered by expected F1 lift vs effort, based on what you've already built and a couple of things that are *specific to how MALLORN was generated* (which you can exploit).

---

## 1) Turn your current Optuna win into a *stable* gain (reduce fold variance)

Your per-fold F1 spread (0.30 → 0.63) is screaming **variance**. The fastest way to lift private/public score is usually: *bagging + better thresholding + ensembling*, not a brand-new model.

**Do this next:**

* **Repeat CV with multiple random seeds** (e.g., 5× repeated stratified K-fold) and average OOF predictions for threshold selection; average test predictions for submission.
* **Bag LightGBM itself**: run the best Optuna params with 5–10 different seeds (`bagging_seed`, `feature_fraction_seed`, `seed`) and average probabilities.
* **Stop picking one global threshold from one CV run.**
  Build threshold from **OOF predictions pooled across repeats**.

This directly targets your biggest failure mode (fold instability) and usually buys a meaningful, "free" F1 bump.

---

## 2) Make thresholding smarter than a single number (learn a threshold function)

Right now you're optimizing a *single* threshold (~0.16). But F1 is extremely threshold-sensitive under imbalance, and the "right" threshold depends on *how informative the object is* (cadence, SNR, band coverage, redshift).

**Concrete upgrade:** learn a **threshold rule** (t(x)), not a constant (t).

* Bin by 1–2 reliability drivers (start simple):

  * `n_obs_total` (or per-band coverage)
  * `snr_max` or `snr_mean`
  * `redshift`
* For each bin, choose threshold that maximizes OOF F1, but **regularize**:

  * monotone smoothing (e.g., thresholds should generally *decrease* as SNR increases)
  * minimum examples per bin (merge small bins)
* Apply the bin-dependent threshold at inference.

Why it's especially relevant here: MALLORN's usefulness is heavily band/cadence dependent (the paper explicitly notes (g,r,i) are most frequently useful and (u) is important for TDEs but less frequently observed). ([arXiv][1])

---

## 3) Add "generator-aware" features: fit the same stochastic models used in MALLORN production

This is the biggest *new* idea I'd try because it aligns with how the dataset was made.

MALLORN lightcurves were produced by **Gaussian process fitting** (Matérn + constant + white noise) for transients. ([arXiv][1])
And for AGN they use a **damped random walk**-style fit (the paper discusses DRW handling AGN stochasticity). ([arXiv][1])

**Feature block to add (high leverage, low dimension):**

* Per object (maybe per band, or on the "best" band):

  1. Fit a **transient GP** (Matérn kernel) → extract kernel params: amplitude, length-scale, noise, fit quality (MSE/χ²).
  2. Fit a **DRW model** → extract DRW timescale/amplitude + fit quality.
  3. Use **likelihood/χ² ratio features**: "DRW-like vs transient-like".
* Feed these ~10–50 features into your LightGBM alongside ROCKET/domain.

This directly targets the hardest contaminant (AGN) using the exact inductive bias the dataset was built with.

---

## 4) Swap ROCKET for a stronger/faster transform, then *ensemble*

ROCKET is good, but the ROCKET-family has improved since the original:

* **MiniROCKET** (often similar accuracy, *much* faster). ([arXiv][2])
* **MultiROCKET** (usually more accurate than MiniROCKET by adding more pooling ops + transforms like first differences). ([arXiv][3])
* **HYDRA** (different inductive bias; often strong and complementary). ([Springer][4])

**Why this matters for you:** you're spending ~45 min/trial largely due to huge feature extraction + 5-fold CV. If you can cut feature time, you can afford:

* more Optuna trials,
* repeated CV,
* more seeds (bagging),
* and more diverse models.

**Practical move:** build **two** transforms (e.g., ROCKET + MultiROCKET), train separate GBDTs, then average/stack.

---

## 5) Stop throwing away absolute scale information (use dual representations)

You currently normalize flux per object by max abs value. That helps shape-based learning, but it can remove **useful discriminative signal** (e.g., peak-to-error structure, "how close to detection limit", etc.). MALLORN explicitly ties simulated distances to **matching signal-to-noise relative to survey detection limits**, and removes objects peaking below LSST limiting magnitude. ([arXiv][1])

**Try this:**

* Compute ROCKET/MultiROCKET on:

  * (A) your current normalized flux
  * (B) *raw* de-extincted flux (or flux scaled by flux_err)
* Concatenate (A)+(B) (or ensemble models trained on each).

This often helps because "shape" and "scale/SNR regime" are complementary.

---

## 6) Build a 2-stage system focused on the dominant failure mode (AGN)

Because AGN are dominant, many of your FPs are probably AGN-like.

A strong pattern for imbalanced astro problems:

1. **Stage 1:** "AGN-like vs transient-like" (binary), optimized for *high recall on non-AGN*.
2. **Stage 2:** On the "transient-like" subset, train "TDE vs non-TDE" with heavier domain features and tighter thresholding.

Even if you keep one final submission model, the 2-stage decomposition often yields better separability and lets you tune thresholds more safely.

---

## 7) Ensembling/stacking: do it *properly* with OOF and a tiny meta-model

Once you have 3–6 diverse predictors (e.g., ROCKET-LGBM, MultiROCKET-LGBM, GP/DRW-feature-LGBM, maybe a linear model on ROCKET features), do:

* Generate **OOF probabilities** for each base model.
* Train a **simple meta-model** (logistic regression / ridge / small LightGBM) on OOF probs + a few reliability features (n_obs, SNR, redshift).
* Optimize threshold on meta-model OOF.

This is one of the most reliable ways to turn "several 0.5-ish models" into "one 0.6-ish model".

---

## 8) Only after the above: semi-supervised/pseudo-labeling (high risk, can help late)

Pseudo-labeling can work, but it's easy to wreck calibration and F1 if you reinforce biases.

If you try it:

* Only add **extremely high-confidence** pseudo-positives *and* pseudo-negatives
* Weight pseudo-labeled examples low (e.g., 0.1–0.3 of real labels)
* Recompute thresholds from scratch using OOF from the *real-labeled* CV

---

### If I had to pick your next 3 actions

1. **Repeated CV + bagging seeds + pooled-OOF thresholding** (stabilize and lift).
2. **Add GP/DRW fit features + likelihood ratio** (dataset-generation aligned). ([arXiv][1])
3. **MultiROCKET (or HYDRA) as a second transform and ensemble**. ([arXiv][3])


[1]: https://arxiv.org/pdf/2512.04946 "MALLORN: Many Artificial LSST Lightcurves based on Observations of Real Nuclear transients"
[2]: https://arxiv.org/abs/2012.08791?utm_source=chatgpt.com "MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification"
[3]: https://arxiv.org/abs/2102.00457?utm_source=chatgpt.com "MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification"
[4]: https://link.springer.com/article/10.1007/s10618-023-00939-3?utm_source=chatgpt.com "Hydra: competing convolutional kernels for fast and accurate ..."
