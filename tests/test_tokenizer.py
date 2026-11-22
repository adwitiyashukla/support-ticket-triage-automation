from triage.tokenizer import ByteBPE, compression_stats

CORPUS = [
    "my card has not arrived yet",
    "my card is not working",
    "why was I charged a fee",
    "the exchange rate is wrong",
    "I need a new pin for my card",
] * 20


def test_training_respects_the_vocabulary_budget():
    tokenizer = ByteBPE.train(CORPUS, 400, 2)
    assert tokenizer.vocab_size <= 400
    assert tokenizer.vocab_size > 256


def test_roundtrip_is_lossless_for_unseen_text():
    tokenizer = ByteBPE.train(CORPUS, 500, 2)
    for text in ["totally unseen sentence 1234", "punctuation!? yes", "unicode cafe latte", "  spaced  out  "]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_are_added_and_stripped():
    tokenizer = ByteBPE.train(CORPUS, 400, 2)
    ids = tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "hello"


def test_save_and_load_produce_identical_ids(tmp_path):
    tokenizer = ByteBPE.train(CORPUS, 500, 2)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    reloaded = ByteBPE.load(path)
    assert reloaded.vocab_size == tokenizer.vocab_size
    assert reloaded.encode("my card has not arrived") == tokenizer.encode("my card has not arrived")


def test_training_is_deterministic():
    first = ByteBPE.train(CORPUS, 500, 2)
    second = ByteBPE.train(CORPUS, 500, 2)
    assert first.merges == second.merges


def test_merges_beat_raw_bytes_on_compression():
    tokenizer = ByteBPE.train(CORPUS, 800, 2)
    stats = compression_stats(tokenizer, CORPUS[:20])
    assert stats["chars_per_token"] > 1.5
