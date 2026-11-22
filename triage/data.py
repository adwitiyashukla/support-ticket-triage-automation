import csv
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict

import numpy as np

from .config import load_config, load_spec

SOURCES = {
    "banking77_train.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "banking77_test.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    "clinc_full.json": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
}

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_MERSENNE = (1 << 61) - 1
_PERMUTATIONS = 64
_BANDS = 16


def download(paths, force=False):
    paths.ensure()
    fetched = []
    for name, url in SOURCES.items():
        target = paths.raw_dir / name
        if target.exists() and not force:
            continue
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
        fetched.append(name)
    missing = [name for name in SOURCES if not (paths.raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"could not obtain {missing}")
    return fetched


def normalize(text):
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def shingles(text, size):
    padded = normalize(text)
    if len(padded) < size:
        return {padded} if padded else set()
    return {padded[i:i + size] for i in range(len(padded) - size + 1)}


def _hash64(value):
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def _signatures(documents, size, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MERSENNE, size=_PERMUTATIONS, dtype=np.uint64)
    b = rng.integers(0, _MERSENNE, size=_PERMUTATIONS, dtype=np.uint64)
    signature = np.full((len(documents), _PERMUTATIONS), np.iinfo(np.uint64).max, dtype=np.uint64)
    for index, document in enumerate(documents):
        grams = shingles(document, size)
        if not grams:
            continue
        hashes = np.array([_hash64(gram) % _MERSENNE for gram in grams], dtype=np.uint64)
        mixed = (a[:, None] * hashes[None, :] + b[:, None]) % np.uint64(_MERSENNE)
        signature[index] = mixed.min(axis=1)
    return signature


def _buckets(signature):
    rows = _PERMUTATIONS // _BANDS
    table = defaultdict(list)
    for index in range(signature.shape[0]):
        for band in range(_BANDS):
            chunk = signature[index, band * rows:(band + 1) * rows].tobytes()
            table[(band, chunk)].append(index)
    return table


def _jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def near_duplicate_pairs(documents, size, threshold, seed):
    signature = _signatures(documents, size, seed)
    grams = [shingles(document, size) for document in documents]
    seen = set()
    pairs = []
    for members in _buckets(signature).values():
        if len(members) < 2:
            continue
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                key = (left, right) if left < right else (right, left)
                if key in seen:
                    continue
                seen.add(key)
                if _jaccard(grams[key[0]], grams[key[1]]) >= threshold:
                    pairs.append(key)
    return pairs


def cross_split_matches(left_documents, right_documents, size, threshold, seed):
    combined = list(left_documents) + list(right_documents)
    offset = len(left_documents)
    signature = _signatures(combined, size, seed)
    grams = [shingles(document, size) for document in combined]
    flagged = set()
    checked = set()
    for members in _buckets(signature).values():
        if len(members) < 2:
            continue
        for i, first in enumerate(members):
            for second in members[i + 1:]:
                low, high = (first, second) if first < second else (second, first)
                if low >= offset or high < offset:
                    continue
                if (low, high) in checked:
                    continue
                checked.add((low, high))
                if _jaccard(grams[low], grams[high]) >= threshold:
                    flagged.add(low)
    return flagged


def _read_banking(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [(row["text"].strip(), row["category"].strip()) for row in csv.DictReader(handle)]


def _read_clinc_oos(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: [item[0].strip() for item in payload[key]] for key in ("oos_train", "oos_val", "oos_test")}


def stratified_split(rows, fraction, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["intent"]].append(row)
    rng = random.Random(seed)
    held_out, kept = [], []
    for intent in sorted(grouped):
        items = grouped[intent][:]
        rng.shuffle(items)
        take = max(1, int(round(len(items) * fraction)))
        take = min(take, len(items) - 1)
        held_out.extend(items[:take])
        kept.extend(items[take:])
    return kept, held_out


def _row(text, intent, spec, split, source):
    queue = spec.queue_of(intent)
    return {
        "id": hashlib.blake2b(f"{split}:{source}:{text}".encode("utf-8"), digest_size=8).hexdigest(),
        "text": text,
        "intent": intent,
        "queue": queue,
        "priority": spec.queue(queue).priority,
        "split": split,
        "source": source,
    }


def build(config=None, spec=None):
    config = config or load_config()
    spec = spec or load_spec()
    config.paths.ensure()
    download(config.paths)

    train_pairs = _read_banking(config.paths.raw_dir / "banking77_train.csv")
    test_pairs = _read_banking(config.paths.raw_dir / "banking77_test.csv")
    oos = _read_clinc_oos(config.paths.raw_dir / "clinc_full.json")

    limits = config.data
    stats = Counter()

    def acceptable(text):
        return limits.min_chars <= len(text.strip()) <= limits.max_chars

    train_rows = [_row(t, i, spec, "train", "banking77") for t, i in train_pairs if acceptable(t)]
    test_rows = [_row(t, i, spec, "test", "banking77") for t, i in test_pairs if acceptable(t)]
    stats["dropped_by_length"] = (len(train_pairs) + len(test_pairs)) - (len(train_rows) + len(test_rows))

    seen = {}
    deduped = []
    for row in train_rows:
        key = normalize(row["text"])
        if key in seen:
            stats["exact_duplicates_removed"] += 1
            if seen[key] != row["intent"]:
                stats["duplicates_with_conflicting_labels"] += 1
            continue
        seen[key] = row["intent"]
        deduped.append(row)
    train_rows = deduped

    test_keys = {normalize(row["text"]) for row in test_rows}
    kept = [row for row in train_rows if normalize(row["text"]) not in test_keys]
    stats["exact_test_matches_removed"] = len(train_rows) - len(kept)
    train_rows = kept

    pairs = near_duplicate_pairs([row["text"] for row in train_rows], limits.shingle_size, limits.near_dup_jaccard, limits.seed)
    drop = set()
    for left, right in pairs:
        if train_rows[left]["intent"] != train_rows[right]["intent"]:
            continue
        if left not in drop:
            drop.add(right)
    stats["near_duplicates_removed"] = len(drop)
    train_rows = [row for index, row in enumerate(train_rows) if index not in drop]

    contaminated = cross_split_matches(
        [row["text"] for row in train_rows],
        [row["text"] for row in test_rows],
        limits.shingle_size,
        limits.near_dup_jaccard,
        limits.seed,
    )
    stats["train_rows_contaminated_by_test"] = len(contaminated)
    train_rows = [row for index, row in enumerate(train_rows) if index not in contaminated]

    train_rows, val_rows = stratified_split(train_rows, limits.val_fraction, limits.seed)
    for row in val_rows:
        row["split"] = "val"

    label = spec.out_of_scope_label
    oos_plan = [("test", oos["oos_test"], limits.oos_test), ("val", oos["oos_val"], limits.oos_val), ("train", oos["oos_train"], limits.oos_train)]
    rng = random.Random(limits.seed + 1)
    buckets = {"train": train_rows, "val": val_rows, "test": test_rows}
    used = {normalize(row["text"]) for rows in buckets.values() for row in rows}
    for split, pool, count in oos_plan:
        chosen = []
        available = [text for text in pool if acceptable(text)]
        rng.shuffle(available)
        for text in available:
            key = normalize(text)
            if key in used:
                stats["out_of_scope_duplicates_skipped"] += 1
                continue
            used.add(key)
            chosen.append(text)
            if len(chosen) >= count:
                break
        stats[f"out_of_scope_{split}"] = len(chosen)
        buckets[split].extend(_row(text, label, spec, split, "clinc_oos") for text in chosen)

    for split, rows in buckets.items():
        rng.shuffle(rows)
        target = config.paths.processed_dir / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[f"rows_{split}"] = len(rows)

    corpus = config.paths.processed_dir / "corpus.txt"
    with corpus.open("w", encoding="utf-8") as handle:
        for row in buckets["train"]:
            handle.write(row["text"].replace("\n", " ").strip() + "\n")

    queue_path = config.paths.processed_dir / "queue.jsonl"
    sample = buckets["test"][:200]
    with queue_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(sample):
            handle.write(json.dumps({"ticket_id": f"T{index:05d}", "text": row["text"], "gold_intent": row["intent"]}, ensure_ascii=False) + "\n")

    summary = {
        "counts": dict(stats),
        "labels": len(spec.labels()),
        "train_intents": len({row["intent"] for row in buckets["train"]}),
    }
    (config.paths.processed_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_split(config, split):
    path = config.paths.processed_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}, run the prepare stage first")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
