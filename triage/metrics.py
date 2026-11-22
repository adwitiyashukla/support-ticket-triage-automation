import numpy as np


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels) if labels is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    scores = []
    for label in labels:
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        false_positive = int(((y_true != label) & (y_pred == label)).sum())
        false_negative = int(((y_true == label) & (y_pred != label)).sum())
        if true_positive == 0:
            scores.append(0.0)
            continue
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def per_label_report(y_true, y_pred, labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for label in labels:
        support = int((y_true == label).sum())
        if support == 0:
            continue
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        false_positive = int(((y_true != label) & (y_pred == label)).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": label, "support": support, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)})
    return sorted(rows, key=lambda row: row["f1"])


def top_confusions(y_true, y_pred, limit=10):
    counts = {}
    for true_label, predicted in zip(y_true, y_pred):
        if true_label == predicted:
            continue
        key = (true_label, predicted)
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: -item[1])[:limit]
    return [{"gold": key[0], "predicted": key[1], "count": value} for key, value in ordered]


def expected_calibration_error(confidences, correct, bins=15):
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidences.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        selected = (confidences > start) & (confidences <= end)
        if not selected.any():
            continue
        weight = selected.mean()
        total += weight * abs(correct[selected].mean() - confidences[selected].mean())
    return float(total)


def reliability_table(confidences, correct, bins=15):
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for start, end in zip(edges[:-1], edges[1:]):
        selected = (confidences > start) & (confidences <= end)
        if not selected.any():
            continue
        rows.append({
            "bin_low": round(float(start), 4),
            "bin_high": round(float(end), 4),
            "count": int(selected.sum()),
            "mean_confidence": round(float(confidences[selected].mean()), 4),
            "accuracy": round(float(correct[selected].mean()), 4),
        })
    return rows


def roc_auc(scores, positives):
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives).astype(bool)
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(sorted_scores):
        end = start
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        if end > start:
            average = (start + end + 2) / 2.0
            ranks[order[start:end + 1]] = average
        start = end + 1
    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def recall_at_precision(scores, positives, target_precision):
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives).astype(bool)
    order = np.argsort(-scores, kind="mergesort")
    sorted_positives = positives[order]
    true_positive = np.cumsum(sorted_positives)
    predicted = np.arange(1, len(scores) + 1)
    precision = true_positive / predicted
    total_positive = max(int(positives.sum()), 1)
    recall = true_positive / total_positive
    eligible = precision >= target_precision
    if not eligible.any():
        return 0.0, 1.0
    index = int(np.argmax(recall * eligible))
    return float(recall[index]), float(scores[order][index])


def percentiles(values, points=(50, 90, 95, 99)):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {f"p{point}": 0.0 for point in points}
    return {f"p{point}": round(float(np.percentile(values, point)), 3) for point in points}


def bootstrap_interval(y_true, y_pred, metric, samples=500, seed=13, level=0.95):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        index = rng.integers(0, y_true.size, y_true.size)
        values.append(metric(y_true[index], y_pred[index]))
    lower = float(np.percentile(values, (1 - level) / 2 * 100))
    upper = float(np.percentile(values, (1 + level) / 2 * 100))
    return (round(lower, 4), round(upper, 4))
