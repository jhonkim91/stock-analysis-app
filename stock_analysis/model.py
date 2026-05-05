from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


class ProbabilityModel(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "ProbabilityModel":
        ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        ...

    def predict(self, x: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        ...


@dataclass
class Metrics:
    accuracy: float
    balanced_accuracy: float
    baseline_accuracy: float
    edge_vs_baseline: float
    precision_up: float
    recall_up: float
    train_rows: int
    test_rows: int
    positive_rate_test: float
    walk_forward_accuracy: float | None = None
    walk_forward_balanced_accuracy: float | None = None
    walk_forward_baseline_accuracy: float | None = None
    walk_forward_edge_vs_baseline: float | None = None
    walk_forward_test_rows: int = 0
    recommended_threshold: float | None = None
    recommended_threshold_basis: str | None = None
    recommended_threshold_edge: float | None = None
    recommended_threshold_balanced_accuracy: float | None = None


@dataclass
class ModelComparison:
    name: str
    label: str
    metrics: Metrics
    selected: bool = False


@dataclass
class TrainingResult:
    model: ProbabilityModel
    model_name: str
    model_label: str
    model_selection_basis: str
    metrics: Metrics
    compared_models: list[ModelComparison]
    walk_forward_actuals: np.ndarray | None = None
    walk_forward_probabilities: np.ndarray | None = None
    walk_forward_start_index: int | None = None


@dataclass(frozen=True)
class _CandidateSpec:
    name: str
    label: str
    factory: Callable[[], ProbabilityModel]


@dataclass
class _EvaluatedCandidate:
    name: str
    label: str
    model: ProbabilityModel
    metrics: Metrics
    walk_forward_actuals: np.ndarray | None = None
    walk_forward_probabilities: np.ndarray | None = None
    walk_forward_start_index: int | None = None


class LogisticDirectionModel:
    """Small logistic regression classifier implemented with numpy."""

    def __init__(self, *, learning_rate: float = 0.05, epochs: int = 2500, l2: float = 0.001):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticDirectionModel":
        if x.ndim != 2:
            raise ValueError("x must be a 2D array.")
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of rows.")

        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0
        x_scaled = (x - self.mean) / self.std

        self.weights = np.zeros(x_scaled.shape[1], dtype=float)
        self.bias = 0.0
        n_rows = len(y)

        for _ in range(self.epochs):
            logits = x_scaled @ self.weights + self.bias
            probabilities = _sigmoid(logits)
            error = probabilities - y

            grad_w = (x_scaled.T @ error) / n_rows + self.l2 * self.weights
            grad_b = float(error.mean())

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None or self.mean is None or self.std is None:
            raise RuntimeError("Model has not been fitted yet.")
        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_scaled = (x - self.mean) / self.std
        return _sigmoid(x_scaled @ self.weights + self.bias)

    def predict(self, x: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


class HistGradientTreeModel:
    """Free tree-based classifier using scikit-learn's histogram gradient boosting."""

    def __init__(self) -> None:
        self.model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=3,
            max_iter=180,
            min_samples_leaf=20,
            l2_regularization=0.05,
            random_state=42,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HistGradientTreeModel":
        self.model.fit(x, y.astype(int))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        probabilities = self.model.predict_proba(x)
        return _ensure_probability_vector(probabilities)

    def predict(self, x: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


def train_direction_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    test_size: float = 0.2,
    threshold: float = 0.5,
    compute_walk_forward_metrics: bool = False,
    optimize_threshold: bool = False,
    threshold_candidates: list[float] | None = None,
    compare_tree_model: bool = False,
) -> TrainingResult:
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be greater than 0 and smaller than 0.5.")

    x = frame.loc[:, feature_columns].to_numpy(dtype=float)
    y = frame["target_up"].to_numpy(dtype=float)

    n_rows = len(frame)
    if n_rows < 200:
        raise ValueError(f"Only {n_rows} training rows are available. More history is needed.")

    test_rows = max(30, int(n_rows * test_size))
    train_rows = n_rows - test_rows
    if train_rows < 120:
        raise ValueError("Not enough rows remain for training after the test split.")

    candidate_specs = [
        _CandidateSpec("logistic", "로지스틱 회귀", LogisticDirectionModel),
    ]
    if compare_tree_model:
        candidate_specs.append(
            _CandidateSpec("tree", "무료 트리 모델", HistGradientTreeModel),
        )

    evaluated_candidates = [
        _evaluate_candidate(
            spec=spec,
            x=x,
            y=y,
            train_rows=train_rows,
            test_rows=test_rows,
            threshold=threshold,
            compute_walk_forward_metrics=compute_walk_forward_metrics,
            optimize_threshold=optimize_threshold,
            threshold_candidates=threshold_candidates,
        )
        for spec in candidate_specs
    ]

    selected_candidate = max(evaluated_candidates, key=_candidate_selection_key)
    selection_basis = _selection_basis(selected_candidate.metrics)

    compared_models = [
        ModelComparison(
            name=candidate.name,
            label=candidate.label,
            metrics=candidate.metrics,
            selected=candidate.name == selected_candidate.name,
        )
        for candidate in evaluated_candidates
    ]

    return TrainingResult(
        model=selected_candidate.model,
        model_name=selected_candidate.name,
        model_label=selected_candidate.label,
        model_selection_basis=selection_basis,
        metrics=selected_candidate.metrics,
        compared_models=compared_models,
        walk_forward_actuals=selected_candidate.walk_forward_actuals,
        walk_forward_probabilities=selected_candidate.walk_forward_probabilities,
        walk_forward_start_index=selected_candidate.walk_forward_start_index,
    )


def _evaluate_candidate(
    *,
    spec: _CandidateSpec,
    x: np.ndarray,
    y: np.ndarray,
    train_rows: int,
    test_rows: int,
    threshold: float,
    compute_walk_forward_metrics: bool,
    optimize_threshold: bool,
    threshold_candidates: list[float] | None,
) -> _EvaluatedCandidate:
    x_train, x_test = x[:train_rows], x[train_rows:]
    y_train, y_test = y[:train_rows], y[train_rows:]

    evaluation_model = spec.factory().fit(x_train, y_train)
    holdout_probabilities = _ensure_probability_vector(evaluation_model.predict_proba(x_test))
    if not np.isfinite(holdout_probabilities).all():
        raise RuntimeError(f"{spec.label} produced non-finite probabilities.")

    walk_forward_probabilities: np.ndarray | None = None
    walk_forward_actuals: np.ndarray | None = None
    if compute_walk_forward_metrics or optimize_threshold:
        walk_forward_actuals, walk_forward_probabilities = _walk_forward_probabilities(
            x=x,
            y=y,
            start_index=train_rows,
            model_factory=spec.factory,
        )

    recommendation = None
    threshold_to_use = float(threshold)
    if optimize_threshold:
        recommendation = _recommend_threshold(
            y_holdout=y_test,
            holdout_probabilities=holdout_probabilities,
            y_walk_forward=walk_forward_actuals,
            walk_forward_probabilities=walk_forward_probabilities,
            threshold_candidates=threshold_candidates,
            default_threshold=threshold,
        )
        threshold_to_use = recommendation["threshold"]

    holdout_predicted = (holdout_probabilities >= threshold_to_use).astype(int)
    holdout_metrics = _classification_metrics(y_test, holdout_predicted)

    walk_forward_metrics = None
    if compute_walk_forward_metrics and walk_forward_actuals is not None and walk_forward_probabilities is not None:
        walk_forward_predicted = (walk_forward_probabilities >= threshold_to_use).astype(int)
        walk_forward_metrics = _classification_metrics(walk_forward_actuals, walk_forward_predicted)
        walk_forward_metrics["rows"] = float(len(walk_forward_actuals))

    final_model = spec.factory().fit(x, y)

    return _EvaluatedCandidate(
        name=spec.name,
        label=spec.label,
        model=final_model,
        metrics=Metrics(
            accuracy=holdout_metrics["accuracy"],
            balanced_accuracy=holdout_metrics["balanced_accuracy"],
            baseline_accuracy=holdout_metrics["baseline_accuracy"],
            edge_vs_baseline=holdout_metrics["edge_vs_baseline"],
            precision_up=holdout_metrics["precision_up"],
            recall_up=holdout_metrics["recall_up"],
            train_rows=train_rows,
            test_rows=test_rows,
            positive_rate_test=holdout_metrics["positive_rate"],
            walk_forward_accuracy=None if walk_forward_metrics is None else walk_forward_metrics["accuracy"],
            walk_forward_balanced_accuracy=None
            if walk_forward_metrics is None
            else walk_forward_metrics["balanced_accuracy"],
            walk_forward_baseline_accuracy=None
            if walk_forward_metrics is None
            else walk_forward_metrics["baseline_accuracy"],
            walk_forward_edge_vs_baseline=None
            if walk_forward_metrics is None
            else walk_forward_metrics["edge_vs_baseline"],
            walk_forward_test_rows=0 if walk_forward_metrics is None else int(walk_forward_metrics["rows"]),
            recommended_threshold=None if recommendation is None else recommendation["threshold"],
            recommended_threshold_basis=None if recommendation is None else recommendation["basis"],
            recommended_threshold_edge=None if recommendation is None else recommendation["edge_vs_baseline"],
            recommended_threshold_balanced_accuracy=None
            if recommendation is None
            else recommendation["balanced_accuracy"],
        ),
        walk_forward_actuals=walk_forward_actuals,
        walk_forward_probabilities=walk_forward_probabilities,
        walk_forward_start_index=train_rows if walk_forward_actuals is not None else None,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -500, 500)
    return 1 / (1 + np.exp(-clipped))


def _ensure_probability_vector(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim == 2:
        if values.shape[1] < 2:
            raise ValueError("predict_proba must expose at least two class columns.")
        return values[:, 1]
    return values.reshape(-1)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    accuracy = float((y_pred == y_true).mean())
    positive_rate = float(y_true.mean())
    baseline_accuracy = float(max(positive_rate, 1 - positive_rate))

    true_positive = float(((y_pred == 1) & (y_true == 1)).sum())
    false_positive = float(((y_pred == 1) & (y_true == 0)).sum())
    false_negative = float(((y_pred == 0) & (y_true == 1)).sum())
    true_negative = float(((y_pred == 0) & (y_true == 0)).sum())

    precision_up = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall_up = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    recall_down = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    balanced_accuracy = float((recall_up + recall_down) / 2)
    edge_vs_baseline = float(accuracy - baseline_accuracy)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "edge_vs_baseline": edge_vs_baseline,
        "precision_up": float(precision_up),
        "recall_up": float(recall_up),
        "positive_rate": positive_rate,
    }


def _walk_forward_probabilities(
    *,
    x: np.ndarray,
    y: np.ndarray,
    start_index: int,
    model_factory: Callable[[], ProbabilityModel],
) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[float] = []
    actuals: list[float] = []

    for index in range(start_index, len(x)):
        model = model_factory().fit(x[:index], y[:index])
        probability = float(_ensure_probability_vector(model.predict_proba(x[index]))[0])
        probabilities.append(probability)
        actuals.append(float(y[index]))

    return np.asarray(actuals, dtype=float), np.asarray(probabilities, dtype=float)


def _recommend_threshold(
    *,
    y_holdout: np.ndarray,
    holdout_probabilities: np.ndarray,
    y_walk_forward: np.ndarray | None,
    walk_forward_probabilities: np.ndarray | None,
    threshold_candidates: list[float] | None,
    default_threshold: float,
) -> dict[str, float | str]:
    candidates = _normalize_threshold_candidates(threshold_candidates, default_threshold)
    basis = "walk_forward" if y_walk_forward is not None and walk_forward_probabilities is not None else "holdout"
    best: dict[str, float | str] | None = None

    for candidate in candidates:
        if basis == "walk_forward":
            predicted = (walk_forward_probabilities >= candidate).astype(int)
            metrics = _classification_metrics(y_walk_forward, predicted)
        else:
            predicted = (holdout_probabilities >= candidate).astype(int)
            metrics = _classification_metrics(y_holdout, predicted)

        current = {
            "threshold": float(candidate),
            "basis": basis,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "edge_vs_baseline": metrics["edge_vs_baseline"],
        }
        if best is None or _recommendation_key(current) > _recommendation_key(best):
            best = current

    assert best is not None
    return best


def _normalize_threshold_candidates(
    threshold_candidates: list[float] | None,
    default_threshold: float,
) -> list[float]:
    if threshold_candidates:
        values = threshold_candidates
    else:
        values = [0.40, 0.45, 0.50, 0.55, 0.60]

    values = [float(value) for value in values if 0.05 <= float(value) <= 0.95]
    values.append(float(default_threshold))
    return sorted({round(value, 4) for value in values})


def _recommendation_key(payload: dict[str, float | str]) -> tuple[float, float, float, float]:
    threshold = float(payload["threshold"])
    return (
        float(payload["edge_vs_baseline"]),
        float(payload["balanced_accuracy"]),
        float(payload["accuracy"]),
        -abs(threshold - 0.5),
    )


def _selection_basis(metrics: Metrics) -> str:
    if metrics.walk_forward_accuracy is not None:
        return "walk_forward"
    return "holdout"


def _candidate_selection_key(candidate: _EvaluatedCandidate) -> tuple[float, float, float, float, float, float]:
    metrics = candidate.metrics
    if metrics.walk_forward_accuracy is not None:
        primary_edge = float(metrics.walk_forward_edge_vs_baseline or 0.0)
        primary_balanced = float(metrics.walk_forward_balanced_accuracy or 0.0)
        primary_accuracy = float(metrics.walk_forward_accuracy)
        primary_basis = 1.0
    else:
        primary_edge = float(metrics.edge_vs_baseline)
        primary_balanced = float(metrics.balanced_accuracy)
        primary_accuracy = float(metrics.accuracy)
        primary_basis = 0.0

    simplicity_bonus = 1.0 if candidate.name == "logistic" else 0.0
    return (
        primary_basis,
        primary_edge,
        primary_balanced,
        primary_accuracy,
        float(metrics.edge_vs_baseline),
        simplicity_bonus,
    )
