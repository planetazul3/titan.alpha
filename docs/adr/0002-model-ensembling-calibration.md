# 2. Model Ensemble and Probability Calibration

Date: 2026-01-04

## Status

Accepted

## Context

Raw model outputs (logits or softmax probabilities) are often uncalibrated, meaning a predicted 0.7 probability doesn't necessarily correspond to a 70% win rate. Furthermore, relying on a single model architecture increases variance and the risk of overfitting to specific regimes.

## Decision

We implemented a two-stage post-processing pipeline:
1.  **Calibration**: All raw model outputs are passed through a `ProbabilityCalibrator` (using Isotonic Regression with a robust Binning fallback) to map predictions to realized win rates.
2.  **Ensembling**: Calibrated outputs from multiple models (or checkpoints) are combined using an `EnsembleStrategy` (Average, Voting, or Weighted).

These components are integrated directly into the `DecisionEngine`.

## Consequences

**Positive:**
-   Reliability: Trading decisions are based on probabilities that better reflect reality.
-   Robustness: Ensembling stabilizes predictions and reduces the impact of any single model's failure.
-   Fallback: The binning fallback ensures the system works even without `sklearn`.

**Negative:**
-   Latency: Adding calibration and ensembling steps increases the inference path latency slightly.
-   Data Requirement: Calibration requires a history of predictions vs. outcomes to "train" the calibrator. Cold start is an issue.
