import pytest

from triage.config import ConfigError, load_config, load_spec

BASE_CONFIG = open("config.toml", encoding="utf-8").read()
BASE_SPEC = open("spec.toml", encoding="utf-8").read()


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_config_and_spec_load():
    config = load_config()
    spec = load_spec()
    assert config.screener.d_model % config.screener.n_head == 0
    assert len(spec.labels()) == len(spec.intents) + 1


def test_every_intent_maps_to_a_declared_queue(spec):
    for intent, queue in spec.intents.items():
        assert queue in spec.queues, intent


def test_queue_policy_marks_risky_queues_as_human(spec):
    assert spec.is_auto("cards_ops") is True
    assert spec.is_auto("security_fraud") is False
    assert spec.is_auto("identity_kyc") is False


def test_head_dimension_must_be_even(tmp_path):
    broken = BASE_CONFIG.replace("n_head = 4", "n_head = 3")
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, "config.toml", broken))


def test_fraction_bounds_are_enforced(tmp_path):
    broken = BASE_CONFIG.replace("val_fraction = 0.12", "val_fraction = 1.4")
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, "config.toml", broken))


def test_unknown_action_is_rejected(tmp_path):
    broken = BASE_SPEC.replace('actions = ["send_reset_link"]', 'actions = ["teleport_customer"]')
    with pytest.raises(ConfigError):
        load_spec(write(tmp_path, "spec.toml", broken))


def test_intent_pointing_at_missing_queue_is_rejected(tmp_path):
    broken = BASE_SPEC.replace('card_arrival = "cards_ops"', 'card_arrival = "does_not_exist"')
    with pytest.raises(ConfigError):
        load_spec(write(tmp_path, "spec.toml", broken))


def test_out_of_scope_label_cannot_collide(tmp_path):
    broken = BASE_SPEC.replace('out_of_scope_label = "out_of_scope"', 'out_of_scope_label = "card_arrival"')
    with pytest.raises(ConfigError):
        load_spec(write(tmp_path, "spec.toml", broken))
