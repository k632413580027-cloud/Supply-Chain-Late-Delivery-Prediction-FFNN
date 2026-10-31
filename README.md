# Interpretable FFNN Regression for Shipping-Duration Deviation Prediction

**Leakage-aware evaluation · Order-disjoint validation · Temporal robustness · Explainable deep learning**

A research-oriented deep learning project investigating whether a **Feedforward Neural Network (FFNN)** provides meaningful predictive improvements for shipping-duration deviations in supply chains—and whether those improvements persist under realistic validation protocols and comparisons with simple operational benchmarks.

> **Main finding:** On the DataCo Smart Supply Chain dataset, the FFNN did **not** outperform a training-fitted Shipping-Mode Median benchmark in order-balanced mean absolute error (MAE) under order-grouped or chronological evaluation. The project highlights the importance of leakage-aware testing, transparent baselines, and stable model explanations rather than model complexity alone.

## Research overview

Most delivery-delay prediction studies frame the task as a binary classification problem (late versus on time). This study instead estimates the **signed magnitude of shipping-schedule deviation** in days, enabling distinctions between early, on-schedule, and late shipment durations.

The study addresses five questions:

1. How is signed shipping-duration deviation distributed in the dataset?
2. How accurately can an FFNN estimate the deviation in days?
3. How sensitive are results to random, order-disjoint, and chronological splits?
4. Does the FFNN improve upon training-fitted operational baselines, and how robust are the comparisons?
5. Do feature-importance explanations remain consistent across training seeds and explanation methods?

## Dataset and prediction target

**Source:** [DataCo Smart Supply Chain for Big Data Analysis](https://data.mendeley.com/datasets/8gx2fvg2k6/5)

| Dataset characteristic | Value |
|---|---|
| Order-item observations | 180,519 |
| Unique orders | 65,752 |
| Original variables | 53 |
| Selected source predictors | 21 |
| Encoded model inputs | 39 |
| Target range | −2 to +4 days |
| Task | Supervised regression |

The prediction target is defined as:

```text
delay_duration = Days for shipping (real)
               - Days for shipment (scheduled)
```

- **Positive:** Actual shipping duration exceeds the scheduled duration.
- **Zero:** Actual and scheduled shipping durations match.
- **Negative:** Actual shipping duration is shorter than scheduled.

**Scope clarification:** This target measures *shipping-duration deviation*, not verified final customer-delivery delay. The dataset does not independently establish the final customer receipt timestamp.

## Methodology

### Data preprocessing

- Removed target-derived, directly outcome-revealing, identifying, and unsuitable fields from the predictor matrix.
- Engineered calendar features from order-placement timestamps.
- Applied categorical encoding and numerical standardization.
- Fitted encoders and scalers on the respective **training partition**, then transformed validation and test sets.
- Audited the consequences of shared orders across item-level partitions.

Some retained financial or fulfillment-related variables have unresolved point-in-time availability; this limitation is explicitly discussed below.

### Neural network architecture

| Component | Configuration |
|---|---|
| Input | 39 encoded features |
| Hidden dense layers | 128 → 64 → 32 neurons |
| Hidden activation | ReLU |
| Batch normalization | After first two hidden layers |
| Dropout | 0.20 and 0.10 |
| L2 regularization | 0.0001 |
| Output | 1 neuron, linear activation |
| Optimizer | Adam, learning rate 0.0003 |
| Training loss | Mean squared error (MSE) |
| Maximum epochs | 50 |
| Batch size | 256 |
| Early stopping | Validation loss, patience 8; restore best weights |

### Evaluation protocols

| Protocol | Purpose |
|---|---|
| Random item-level split (70/15/15) | Reproduce the initial model evaluation; diagnose order overlap |
| Order-grouped split | Evaluate unseen orders using disjoint `Order Id` groups |
| Three chronological windows | Assess performance over later historical periods using expanding training windows and a label-availability proxy purge |
| Multiple training seeds | Examine sensitivity to stochastic training |
| Cluster-aware uncertainty analysis | Examine paired FFNN–baseline MAE differences with dependence-aware bootstrap methods |

**Primary non-temporal evaluation:** order-disjoint holdout with order-balanced metrics, so that each order contributes equal total weight regardless of its number of items.

### Operational benchmarks

All benchmark predictions are fitted using the relevant training partition:

- Global Mean
- Global Median
- Shipping-Mode Mean
- **Shipping-Mode Median** (primary MAE comparator)

The principal comparison is `ΔMAE = MAE(FFNN) − MAE(Shipping-Mode Median)`. A **positive** value favors the baseline; a **negative** value favors the FFNN.

## Key results

### Predictive performance across evaluation designs

The table reports **order-balanced MAE (days)**, as presented in the research report.

| Evaluation design | FFNN | Shipping-Mode Median | ΔMAE |
|---|---:|---:|---:|
| Original random split | 0.99387 | 0.97800 | +0.01587 |
| Order-grouped holdout | 1.00345 | **0.98094** | +0.02251 |
| Temporal window 1 | 1.01608 | **0.97867** | +0.03741 |
| Temporal window 2 | 1.00064 | **0.97108** | +0.02956 |
| Temporal window 3 | 1.01503 | **0.98324** | +0.03179 |

**Interpretation:** The Shipping-Mode Median achieves lower order-balanced MAE in each reported evaluation design. These comparisons do not establish that this simple baseline will always outperform neural networks in other datasets or operational environments.

Under the original random item-level split, FFNN test metrics were **MAE = 0.9908 days**, **RMSE = 1.2620 days**, and **R² = 0.2751**. However, **83.17% of original test observations shared an `Order Id` with training observations**, so these results are diagnostic rather than an order-independent generalization estimate.

In the grouped holdout, mean FFNN order-balanced MAE across five training seeds was **1.00345 days** (SD = 0.00146), versus **0.98094 days** for Shipping-Mode Median. The paired difference was **+0.02251 days**, with an order-cluster conditional 95% bootstrap interval of **[+0.019796, +0.025291]** days. These intervals are conditional on the study's models, split, and resampling assumptions.

### Model explainability

The study compares **Expected-Gradients SHAP** and **grouped permutation importance** on independently fitted order-disjoint FFNN models.

| Diagnostic | Reported result |
|---|---|
| Top grouped-permutation feature | Shipping Mode (mean ΔMAE: 0.32465 days) |
| Top SHAP feature | Shipping Mode (mean absolute attribution: 0.61804 days) |
| Second prominent feature in both methods | Days for shipment (scheduled) |
| Mean between-seed Spearman correlation | SHAP: 0.6631; permutation: 0.3160 |
| Mean between-seed top-5 overlap | SHAP: 0.86; permutation: 0.56 |
| Cross-method mean-score rank correlation | 0.3571 |
| Cross-method mean-score top-5 overlap | 0.40 |

Shipping Mode and scheduled shipping duration consistently emerge as prominent signals, but broader importance rankings are sensitive to retraining and attribution methodology. These explanations describe model behavior, **not causal effects**.

## Research contributions

1. **Regression-based target definition:** Retains the direction and magnitude of shipping-duration deviation instead of reducing the outcome to a binary late/on-time label.
2. **Leakage-aware evaluation:** Identifies order overlap in random item-level splits and uses order-disjoint validation.
3. **More realistic stress testing:** Evaluates chronological splits and dependence-aware uncertainty rather than relying solely on a random holdout.
4. **Meaningful operational benchmarking:** Tests incremental FFNN performance against simple, training-fitted prediction rules.
5. **Explanation-stability assessment:** Quantifies reproducibility across training seeds and agreement between SHAP and permutation importance.

## Repository contents

```text
.
├── FFNN_GROUP_4.ipynb    # Model development, evaluation, robustness, and explanations
└── README.md             # Research summary and repository documentation
```

## Using the notebook

1. Download or clone this repository and open `FFNN_GROUP_4.ipynb` in Jupyter Notebook or Google Colab.
2. Obtain the DataCo Smart Supply Chain dataset from the source linked above, subject to its terms of use.
3. Update the dataset file path(s) in the notebook to point to your local or Colab environment.
4. Install any missing dependencies imported by the notebook and run cells in sequence.
5. Review the evaluation splits, benchmarking outputs, robustness checks, and explanation diagnostics.

**Reproducibility note:** Results may vary across environments and stochastic training runs. The report describes specified split seeds and repeated model training; exact reproduction requires matching the code, dataset, package versions, and computational settings.

## Limitations

- The dataset represents a specific historical and operational setting; independent cross-domain validation is still needed.
- Shipping-duration deviation must not be presented as verified last-mile customer delivery delay.
- Some financial or fulfillment-related predictor fields have unverified availability at prediction time.
- Some feature-screening decisions were informed by whole-dataset exploration.
- Grouped splitting removes same-order overlap but does not remove all customer-level dependence.
- The baseline set consists of transparent operational rules, not a comprehensive suite of tuned machine-learning algorithms.
- SHAP and permutation importance are descriptive explanations and do not establish causality.

## Research team

**Nguyen Nam Tang · Nguyen Hai Ly · Le Thanh Huyen · Nguyen Thi Huong Giang**  
Foreign Trade University, Hanoi, Vietnam

## Reference and attribution

This repository accompanies the authors' research report, **“Interpretable FFNN Regression for Shipping-Duration Deviation Prediction.”** Numerical results and methodological descriptions are based on the supplied report. Please cite the authors' report when using this work, and credit the original DataCo dataset.

---

*Research focus: reliable evaluation, interpretable modeling, and evidence-based predictive analytics for supply-chain operations.*
