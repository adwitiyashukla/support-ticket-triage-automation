import dataclasses
import json

from triage.prompts import build_messages, render_prompt, target_text
from triage.resolver import ResolverModel, _span_confidence
from triage.train_resolver import IGNORE_INDEX, build_examples, collate


def tiny_config(config, tiny_model_dir, tmp_path):
    paths = dataclasses.replace(config.paths, processed_dir=tmp_path / "processed", artifacts_dir=tmp_path / "artifacts", reports_dir=tmp_path / "reports")
    resolver = dataclasses.replace(
        config.resolver, base_model=tiny_model_dir, epochs=1, batch_size=2, grad_accum=1,
        max_len=192, max_new_tokens=12, infer_batch_size=2, train_subset=0, device="cpu", dtype="float32",
    )
    return dataclasses.replace(config, paths=paths, resolver=resolver)


def write_splits(config, spec):
    config.paths.ensure()
    rows = [
        {"text": "my card has not arrived", "intent": "card_arrival"},
        {"text": "how long does a transfer take", "intent": "transfer_timing"},
        {"text": "I lost my card", "intent": "lost_or_stolen_card"},
        {"text": "why was I charged a fee", "intent": "transfer_fee_charged"},
    ]
    for row in rows:
        row["queue"] = spec.queue_of(row["intent"])
    for split in ("train", "val"):
        path = config.paths.processed_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


def test_prompt_tokens_are_masked_out_of_the_loss(config, spec, tiny_model_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    rows = [{"text": "my card has not arrived", "intent": "card_arrival"}]
    examples = build_examples(rows, tokenizer, spec, 192)
    ids, labels = examples[0]
    assert len(ids) == len(labels)
    assert labels[0] == IGNORE_INDEX
    assert any(label != IGNORE_INDEX for label in labels)
    supervised = tokenizer.decode([label for label in labels if label != IGNORE_INDEX])
    assert '"intent"' in supervised


def test_collate_pads_inputs_and_labels_consistently():
    batch = [([1, 2, 3], [IGNORE_INDEX, 2, 3]), ([4, 5], [IGNORE_INDEX, 5])]
    input_ids, attention, labels = collate(batch, pad_id=0)
    assert input_ids.shape == (2, 3)
    assert attention[1].tolist() == [1, 1, 0]
    assert labels[1].tolist() == [IGNORE_INDEX, 5, IGNORE_INDEX]


def test_span_confidence_uses_only_the_intent_tokens():
    pieces = ['{"intent"', ": ", '"card', "_arrival", '", "action"', ": ...}"]
    logprobs = [-0.01, -0.01, -0.2, -0.2, -5.0, -5.0]
    confidence = _span_confidence(pieces, logprobs)
    assert confidence is not None
    assert 0.75 < confidence < 0.9


def test_span_confidence_returns_none_without_an_intent_field():
    assert _span_confidence(["hello ", "world"], [-0.1, -0.1]) is None


def test_target_text_round_trips_through_the_schema(spec):
    from triage.schema import parse

    text = target_text("I was charged 20 GBP twice", "transaction_charged_twice", spec)
    assert parse(text, spec)["intent"] == "transaction_charged_twice"


def test_compact_prompt_is_shorter_than_the_taxonomy_prompt(config, spec, tiny_model_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    compact = render_prompt(tokenizer, build_messages("where is my card", spec, "compact"))
    full = render_prompt(tokenizer, build_messages("where is my card", spec, "full"))
    assert len(tokenizer(compact)["input_ids"]) * 4 < len(tokenizer(full)["input_ids"])


def test_lora_training_runs_and_saves_an_adapter(config, spec, tiny_model_dir, tmp_path):
    from triage.train_resolver import train

    scoped = tiny_config(config, tiny_model_dir, tmp_path)
    write_splits(scoped, spec)
    report = train(scoped, spec, log=lambda *_: None)
    assert report["trainable_parameters"] > 0
    assert report["trainable_parameters"] < report["total_parameters"]
    assert (scoped.paths.artifacts_dir / "resolver-adapter").exists()
    assert 0.0 <= report["sample_schema_valid_rate"] <= 1.0


def test_untrained_model_output_is_reported_as_invalid_not_crashing(config, spec, tiny_model_dir, tmp_path):
    scoped = tiny_config(config, tiny_model_dir, tmp_path)
    model = ResolverModel.load(scoped, spec)
    results = model.predict(["my card has not arrived", "how long does a transfer take"])
    assert len(results) == 2
    for item in results:
        assert "valid" in item
        if not item["valid"]:
            assert item["error"]
            assert item["confidence"] == 0.0
