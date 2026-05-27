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


def majority_class_baseline(y_true: np.ndarray, train_majority_label: int) -> BaselineResult:
    # Deployable majority rule: always predict the most frequent class on TRAIN.
    pred = int(train_majority_label)
    y_prob = np.full(shape=(len(y_true),), fill_value=float(pred), dtype=np.float64)
    # AUC is not meaningful for constant scores if both classes exist -> roc_auc_score would error;
    # reuse evaluate which returns nan if only one class in truth (otherwise sklearn would throw).
    m = evaluate_classifier_probs(y_true=y_true, y_prob=y_prob, threshold=0.5)
    return BaselineResult(auc=m["auc"], acc=m["acc"], f1=m["f1"])


def frozen_resnet18_logistic_baseline(
    train_loader,
    test_loader,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[BaselineResult, np.ndarray]:
    """Frozen ImageNet ResNet18 features + sklearn logistic regression."""
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression

    from .models import build_resnet18

    model = build_resnet18(num_classes=2, pretrained=True).to(device)
    model.fc = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    def _features(loader) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                f = model(x).flatten(1).cpu().numpy()
                xs.append(f)
                ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys).astype(np.int64)

    x_train, y_train = _features(train_loader)
    x_test, y_test = _features(test_loader)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(x_train, y_train)
    y_prob = clf.predict_proba(x_test)[:, 1]
    m = evaluate_classifier_probs(y_true=y_test, y_prob=y_prob, threshold=0.5)
    return BaselineResult(auc=m["auc"], acc=m["acc"], f1=m["f1"]), y_prob

