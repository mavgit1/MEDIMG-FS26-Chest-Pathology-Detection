from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eval import evaluate_classifier_probs


@dataclass(frozen=True)
class BaselineResult:
    auc: float
    acc: float
    f1: float


def distribution_matched_random_baseline(y_true: np.ndarray, train_prevalence: float, seed: int = 0) -> BaselineResult:
    """
    Baseline explicitly allowed by course staff:
    - Predict a random label with P(PNEUMONIA)=pi_train, P(NORMAL)=1-pi_train.
    For AUC we use random probabilities in [0,1] (independent of y), which should yield ~0.5.
    """
    rng = np.random.default_rng(seed)
    # Draw random predictions with the same positive rate as the training set.
    y_pred = rng.binomial(n=1, p=float(train_prevalence), size=len(y_true)).astype(np.int64)
    # Use the sampled label as a degenerate probability (0/1). This matches the stated baseline.
    y_prob = y_pred.astype(np.float64)
    m = evaluate_classifier_probs(y_true=y_true, y_prob=y_prob, threshold=0.5)
    return BaselineResult(auc=m["auc"], acc=m["acc"], f1=m["f1"])


def majority_class_baseline(y_true: np.ndarray) -> BaselineResult:
    # Always predict the most frequent class in y_true (usually NORMAL).
    counts = np.bincount(y_true.astype(np.int64), minlength=2)
    pred = int(np.argmax(counts))
    y_prob = np.full(shape=(len(y_true),), fill_value=float(pred), dtype=np.float64)
    # AUC is not meaningful for constant scores if both classes exist -> roc_auc_score would error;
    # reuse evaluate which returns nan if only one class in truth (otherwise sklearn would throw).
    m = evaluate_classifier_probs(y_true=y_true, y_prob=y_prob, threshold=0.5)
    return BaselineResult(auc=m["auc"], acc=m["acc"], f1=m["f1"])

