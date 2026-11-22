import json
import pickle
import time
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from .config import load_config, load_spec
from .data import load_split
from .metrics import accuracy, bootstrap_interval, macro_f1


def build_vectorizer(settings):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, settings.word_ngram_max), min_df=settings.min_df, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(settings.char_ngram_min, settings.char_ngram_max), min_df=settings.min_df, sublinear_tf=True)
    return FeatureUnion([("word", word), ("char", char)])


def fit_tfidf(config, log=print):
    settings = config.baselines
    train_rows = load_split(config, "train")
    texts = [row["text"] for row in train_rows]
    labels = [row["intent"] for row in train_rows]
    vectorizer = build_vectorizer(settings)
    started = time.perf_counter()
    features = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(C=settings.logreg_c, max_iter=settings.logreg_max_iter)
    classifier.fit(features, labels)
    elapsed = time.perf_counter() - started
    payload = {"vectorizer": vectorizer, "classifier": classifier, "train_texts": texts, "train_labels": labels, "features": features}
    path = config.paths.artifacts_dir / "baseline.pkl"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    log(f"tfidf baseline fitted in {elapsed:.1f}s features {features.shape[1]}")
    return payload


def load_tfidf(config):
    path = config.paths.artifacts_dir / "baseline.pkl"
    if not path.exists():
        return fit_tfidf(config)
    with path.open("rb") as handle:
        return pickle.load(handle)


def nearest_neighbours(payload, texts, k):
    query = payload["vectorizer"].transform(texts)
    similarity = query @ payload["features"].T
    dense = similarity.toarray() if hasattr(similarity, "toarray") else np.asarray(similarity)
    order = np.argsort(-dense, axis=1)[:, :k]
    return order, dense


def majority_baseline(train_rows, test_rows):
    label = Counter(row["intent"] for row in train_rows).most_common(1)[0][0]
    gold = [row["intent"] for row in test_rows]
    predicted = [label] * len(test_rows)
    return {"name": "majority class", "accuracy": round(accuracy(gold, predicted), 4), "macro_f1": round(macro_f1(gold, predicted), 4)}


def tfidf_results(config, payload, test_rows, labels):
    texts = [row["text"] for row in test_rows]
    gold = [row["intent"] for row in test_rows]
    started = time.perf_counter()
    features = payload["vectorizer"].transform(texts)
    predicted = payload["classifier"].predict(features)
    elapsed = (time.perf_counter() - started) * 1000 / max(len(texts), 1)
    probabilities = payload["classifier"].predict_proba(features)
    confidence = probabilities.max(axis=1)
    order, _ = nearest_neighbours(payload, texts, config.baselines.neighbours)
    neighbour_labels = [Counter(payload["train_labels"][index] for index in row).most_common(1)[0][0] for row in order]
    return [
        {
            "name": "tfidf logistic regression",
            "accuracy": round(accuracy(gold, predicted), 4),
            "accuracy_ci": bootstrap_interval(np.array(gold), np.array(predicted), accuracy),
            "macro_f1": round(macro_f1(gold, predicted, labels), 4),
            "mean_confidence": round(float(confidence.mean()), 4),
            "latency_ms_per_ticket": round(elapsed, 3),
        },
        {
            "name": f"tfidf nearest neighbour k={config.baselines.neighbours}",
            "accuracy": round(accuracy(gold, neighbour_labels), 4),
            "macro_f1": round(macro_f1(gold, neighbour_labels, labels), 4),
        },
    ]


def prompt_baseline(config, spec, rows, style, payload=None, log=print, adapter_dir=None):
    from .resolver import ResolverModel

    model = ResolverModel.load(config, spec, adapter_dir)
    texts = [row["text"] for row in rows]
    gold = [row["intent"] for row in rows]
    examples = None
    if style == "fewshot":
        order, _ = nearest_neighbours(payload, texts, config.baselines.fewshot_examples)
        shared = order[0]
        examples = [(payload["train_texts"][index], payload["train_labels"][index]) for index in reversed(shared)]
    started = time.perf_counter()
    predictions = model.predict(texts, "full", examples)
    elapsed = (time.perf_counter() - started) * 1000 / max(len(texts), 1)
    valid = [item for item in predictions if item["valid"]]
    guessed = [item["intent"] if item["valid"] else "" for item in predictions]
    prompt_tokens = float(np.mean([item.get("prompt_tokens", 0) for item in predictions])) if predictions else 0.0
    result = {
        "name": f"base model {style}",
        "rows": len(rows),
        "accuracy": round(accuracy(gold, guessed), 4),
        "macro_f1": round(macro_f1(gold, guessed), 4),
        "schema_valid_rate": round(len(valid) / max(len(predictions), 1), 4),
        "mean_prompt_tokens": round(prompt_tokens, 1),
        "latency_ms_per_ticket": round(elapsed, 3),
    }
    log(f"{result['name']} accuracy {result['accuracy']} schema valid {result['schema_valid_rate']}")
    return result


def run(config=None, spec=None, include_prompt=False, prompt_rows=0, log=print):
    config = config or load_config()
    spec = spec or load_spec()
    config.paths.ensure()
    train_rows = load_split(config, "train")
    test_rows = load_split(config, "test")
    labels = spec.labels()
    payload = load_tfidf(config)

    results = [majority_baseline(train_rows, test_rows)]
    results.extend(tfidf_results(config, payload, test_rows, labels))

    if include_prompt:
        limit = prompt_rows or config.evaluation.resolver_eval_rows
        subset = test_rows[:limit]
        for style in ("zeroshot", "fewshot"):
            results.append(prompt_baseline(config, spec, subset, style, payload, log))

    report = {"test_rows": len(test_rows), "results": results}
    (config.paths.reports_dir / "baselines.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for entry in results:
        log(f"{entry['name']}: accuracy {entry['accuracy']} macro f1 {entry['macro_f1']}")
    return report
