import json

import pytest

from triage.entities import extract
from triage.prompts import build_reply, target_payload
from triage.schema import SchemaError, build_decision, extract_json, parse, policy_for

TYPES = ("amount", "currency", "card_last4", "date", "merchant")


def valid_payload(intent="card_arrival"):
    return {
        "intent": intent,
        "entities": {key: "" for key in TYPES},
        "action": "create_case",
        "reply": "Thanks for contacting Northgate Bank.",
    }


def test_parses_plain_json(spec):
    parsed = parse(json.dumps(valid_payload()), spec)
    assert parsed["intent"] == "card_arrival"


def test_parses_json_wrapped_in_prose_and_fences(spec):
    body = json.dumps(valid_payload())
    assert parse(f"Sure, here you go:\n```json\n{body}\n```", spec)["intent"] == "card_arrival"


def test_rejects_unknown_intent(spec):
    payload = valid_payload("teleportation_request")
    with pytest.raises(SchemaError):
        parse(json.dumps(payload), spec)


def test_accepts_the_out_of_scope_label(spec):
    payload = valid_payload(spec.out_of_scope_label)
    payload["action"] = "escalate_to_human"
    assert parse(json.dumps(payload), spec)["intent"] == spec.out_of_scope_label


def test_rejects_missing_keys(spec):
    payload = valid_payload()
    payload.pop("reply")
    with pytest.raises(SchemaError):
        parse(json.dumps(payload), spec)


def test_rejects_empty_reply(spec):
    payload = valid_payload()
    payload["reply"] = "   "
    with pytest.raises(SchemaError):
        parse(json.dumps(payload), spec)


def test_missing_entities_are_filled_with_empty_strings(spec):
    payload = valid_payload()
    payload["entities"] = {"amount": 45.2}
    parsed = parse(json.dumps(payload), spec)
    assert parsed["entities"]["amount"] == "45.2"
    assert parsed["entities"]["merchant"] == ""


def test_no_json_object_raises():
    with pytest.raises(SchemaError):
        extract_json("I think this is a card problem")


def test_policy_is_derived_from_the_spec_not_the_model(spec):
    policy = policy_for("lost_or_stolen_card", spec)
    assert policy["queue"] == "security_fraud"
    assert policy["auto"] is False
    assert "freeze_card" in policy["actions"]


def test_decision_flags_action_disagreement(spec):
    decision = build_decision("T1", "text", "card_arrival", {}, "reply", 0.9, "resolver", spec, model_action="freeze_card")
    assert decision["action_agrees_with_policy"] is False
    agreeing = build_decision("T1", "text", "card_arrival", {}, "reply", 0.9, "resolver", spec, model_action="create_case")
    assert agreeing["action_agrees_with_policy"] is True


def test_entity_extraction_reads_amounts_and_cards():
    found = extract("I was charged 45.20 GBP twice at Tesco on 12/03/2024", TYPES)
    assert found["amount"] == "45.20"
    assert found["currency"] == "GBP"
    assert found["date"] == "12/03/2024"
    assert found["merchant"] == "Tesco"
    assert extract("my card ending in 4821 stopped working", TYPES)["card_last4"] == "4821"


def test_entity_extraction_returns_blanks_when_nothing_matches():
    assert extract("why is my card not working", TYPES) == {key: "" for key in TYPES}


def test_reply_mentions_the_amount_when_known():
    reply = build_reply("payments_disputes", {"amount": "45.20", "currency": "GBP"}, "Northgate Bank")
    assert "45.20 GBP" in reply
    assert reply.startswith("Thanks for contacting Northgate Bank.")


def test_training_target_matches_the_schema(spec):
    payload = target_payload("I was charged twice for 20 GBP", "transaction_charged_twice", spec)
    assert parse(json.dumps(payload), spec)["intent"] == "transaction_charged_twice"
    assert payload["action"] in spec.queue(spec.queue_of("transaction_charged_twice")).actions
