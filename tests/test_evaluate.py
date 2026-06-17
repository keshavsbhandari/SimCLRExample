import matplotlib

matplotlib.use("Agg")

import numpy as np

from simclr_fraud.evaluate.plots import (
    find_best_f1_threshold,
    plot_pr_curve,
    plot_roc_curve,
    threshold_metrics_table,
)


def _synthetic_scores(n: int = 500, fraud_rate: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < fraud_rate).astype(int)
    y_score = rng.random(n)
    y_score[y_true == 1] += 0.4
    y_score = np.clip(y_score, 0, 1)
    return y_true, y_score


def test_threshold_metrics_table_shape():
    y_true, y_score = _synthetic_scores()
    df = threshold_metrics_table(y_true, y_score, [0.5, 0.9])
    assert len(df) == 2
    assert set(df.columns) >= {"threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"}


def test_find_best_f1_threshold_returns_floats():
    y_true, y_score = _synthetic_scores()
    t, f1 = find_best_f1_threshold(y_true, y_score)
    assert 0.0 <= t <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_plot_roc_and_pr_return_figures():
    y_true, y_score = _synthetic_scores()
    roc_fig = plot_roc_curve(y_true, y_score, [0.5, 0.9])
    pr_fig = plot_pr_curve(y_true, y_score, [0.5, 0.9])
    assert roc_fig.axes
    assert pr_fig.axes
    import matplotlib.pyplot as plt

    plt.close(roc_fig)
    plt.close(pr_fig)
