import json

import pytest

from triage.data import cross_split_matches, near_duplicate_pairs, normalize, shingles, stratified_split


def test_normalize_strips_punctuation_and_case():
    assert normalize("  My CARD, please!!  ") == "my card please"


def test_shingles_cover_short_strings():
    assert shingles("ab", 4) == {"ab"}
    assert len(shingles("abcdef", 4)) == 3


def test_near_duplicate_pairs_finds_reworded_copies():
    documents = [
        "my card has not arrived yet",
        "my card has not arrived yet!",
        "how do I change my pin",
        "the exchange rate looks wrong",
    ]
    pairs = near_duplicate_pairs(documents, 4, 0.9, 13)
    assert (0, 1) in pairs
    assert all(set(pair) != {2, 3} for pair in pairs)


def test_cross_split_matches_flags_only_left_side():
    left = ["I lost my card in Paris", "why is my top up pending"]
    right = ["I lost my card in Paris"]
    flagged = cross_split_matches(left, right, 4, 0.9, 13)
    assert flagged == {0}


def test_stratified_split_keeps_every_intent_in_both_halves():
    rows = [{"intent": f"intent_{index % 5}", "text": f"ticket {index}"} for index in range(50)]
    kept, held_out = stratified_split(rows, 0.2, 13)
    assert len({row["intent"] for row in kept}) == 5
    assert len({row["intent"] for row in held_out}) == 5
    assert len(kept) + len(held_out) == 50


def test_stratified_split_never_empties_a_class():
    rows = [{"intent": "only", "text": "a"}, {"intent": "only", "text": "b"}]
    kept, held_out = stratified_split(rows, 0.9, 13)
    assert kept and held_out


def _read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def test_prepared_splits_do_not_overlap(config, processed_available):
    if not processed_available:
        pytest.skip("run the prepare stage first")
    train = _read(config.paths.processed_dir / "train.jsonl")
    val = _read(config.paths.processed_dir / "val.jsonl")
    test = _read(config.paths.processed_dir / "test.jsonl")
    train_keys = {normalize(row["text"]) for row in train}
    val_keys = {normalize(row["text"]) for row in val}
    test_keys = {normalize(row["text"]) for row in test}
    assert not train_keys & val_keys
    assert not train_keys & test_keys


def test_prepared_splits_carry_queue_and_priority(config, spec, processed_available):
    if not processed_available:
        pytest.skip("run the prepare stage first")
    for row in _read(config.paths.processed_dir / "test.jsonl")[:200]:
        assert row["queue"] == spec.queue_of(row["intent"])
        assert row["priority"] == spec.queue(row["queue"]).priority
