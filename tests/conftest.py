import json
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from triage.config import load_config, load_spec

SAMPLE_TICKETS = [
    "my card has not arrived yet",
    "why was I charged a fee for the transfer",
    "I lost my card in Paris",
    "the atm gave me the wrong amount of cash",
    "how do I top up with a different card",
    "can I use my card in Japan",
]


@pytest.fixture(scope="session")
def spec():
    return load_spec()


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def processed_available(config):
    return (config.paths.processed_dir / "test.jsonl").exists()


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    target = tmp_path_factory.mktemp("tiny-base")
    corpus = list(SAMPLE_TICKETS) + [
        json.dumps({"intent": "card_arrival", "entities": {"amount": "", "currency": "", "card_last4": "", "date": "", "merchant": ""},
                    "action": "create_case", "reply": "Thanks for contacting Northgate Bank."}),
    ]
    backend = Tokenizer(models.BPE(unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=1024,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    backend.train_from_iterator(corpus * 8, trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>", pad_token="<|endoftext|>", unk_token="<|endoftext|>")
    tokenizer.chat_template = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    ))
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    return str(target)
