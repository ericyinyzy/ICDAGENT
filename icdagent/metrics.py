"""Macro / micro precision, recall and F1 for multi-label ICD coding.

Macro metrics are averaged only over codes that occur at least once in the
gold label set of the evaluated split.
"""
import numpy as np

from .utils import normalize

# Predicted codes outside the benchmark label space are real false positives, so
# they get parked in spare columns instead of being silently dropped.
OOV_SLOTS = 64


def build_matrices(records, label_space, pred_key='extracted_codes',
                   gold_key='label', strip_dot=False):
    """Turn prediction records into the binary (yhat, y) matrices."""
    columns = list(label_space) + ['|__oov_%d__|' % i for i in range(OOV_SLOTS)]
    index = {code: i for i, code in enumerate(columns)}
    y = np.zeros((len(records), len(columns)), dtype=int)
    yhat = np.zeros((len(records), len(columns)), dtype=int)
    for row, sample in enumerate(records):
        gold = sample[gold_key]
        if isinstance(gold, str):
            gold = gold.split(';')
        for code in (normalize(c) for c in gold):
            if code in index:
                y[row, index[code]] = 1
        oov = -1
        for code in (normalize(c, strip_dot) for c in set(sample[pred_key])):
            if code in index:
                yhat[row, index[code]] = 1
            else:
                yhat[row, oov] = 1
                oov -= 1
    return yhat, y


def _union_size(yhat, y, axis):
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)


def _intersect_size(yhat, y, axis):
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


def macro_accuracy(yhat, y):
    m = y.sum(axis=0) > 0
    return np.mean(_intersect_size(yhat[:, m], y[:, m], 0) / (_union_size(yhat[:, m], y[:, m], 0) + 1e-10))


def macro_precision(yhat, y):
    m = y.sum(axis=0) > 0
    return np.mean(_intersect_size(yhat[:, m], y[:, m], 0) / (yhat[:, m].sum(axis=0) + 1e-10))


def macro_recall(yhat, y):
    m = y.sum(axis=0) > 0
    return np.mean(_intersect_size(yhat[:, m], y[:, m], 0) / (y[:, m].sum(axis=0) + 1e-10))


def macro_f1(yhat, y):
    if not np.any(y.sum(axis=0) > 0):
        return 0.0
    p, r = macro_precision(yhat, y), macro_recall(yhat, y)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def micro_accuracy(yhat, y):
    return _intersect_size(yhat, y, 0) / (1e-10 + _union_size(yhat, y, 0))


def micro_precision(yhat, y):
    return _intersect_size(yhat, y, 0) / (1e-10 + yhat.sum(axis=0))


def micro_recall(yhat, y):
    return _intersect_size(yhat, y, 0) / (1e-10 + y.sum(axis=0))


def micro_f1(yhat, y):
    p, r = micro_precision(yhat, y), micro_recall(yhat, y)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def average_recall(yhat, y):
    out = []
    for i in range(y.shape[0]):
        total = y[i].sum()
        out.append(0.0 if total == 0 else np.logical_and(yhat[i], y[i]).sum() / total)
    return float(np.mean(out))


def all_metrics(yhat, y):
    ymic, yhatmic = y.ravel(), yhat.ravel()
    return {
        'acc_macro': macro_accuracy(yhat, y),
        'prec_macro': macro_precision(yhat, y),
        'rec_macro': macro_recall(yhat, y),
        'f1_macro': macro_f1(yhat, y),
        'acc_micro': micro_accuracy(yhatmic, ymic),
        'prec_micro': micro_precision(yhatmic, ymic),
        'rec_micro': micro_recall(yhatmic, ymic),
        'f1_micro': micro_f1(yhatmic, ymic),
        'avg_recall': average_recall(yhat, y),
    }
