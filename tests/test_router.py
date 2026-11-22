import json

from triage.router import Router, summarize
from triage.tools import ToolRunner


class StubScreener:
    def __init__(self, answers):
        self.answers = answers
        self.labels = [name for name, _ in answers]

    def predict(self, texts):
        return [{"intent": self.answers[index][0], "confidence": self.answers[index][1], "latency_ms": 0.5} for index in range(len(texts))]


class StubResolver:
    def __init__(self, results):
        self.results = results

    def predict(self, texts, style="compact", examples=None):
        return self.results[:len(texts)]


def resolver_result(intent, confidence, valid=True, action="create_case"):
    if not valid:
        return {"valid": False, "error": "no json object found in the model output", "raw": "sorry", "confidence": 0.0,
                "prompt_tokens": 10, "generated_tokens": 4}
    return {
        "valid": True, "error": "", "raw": "{}", "intent": intent,
        "entities": {"amount": "", "currency": "", "card_last4": "", "date": "", "merchant": ""},
        "action": action, "reply": "Thanks for contacting Northgate Bank.", "confidence": confidence,
        "sequence_confidence": confidence, "repaired": False, "prompt_tokens": 40, "generated_tokens": 30,
    }


def test_confident_screener_prediction_is_automated(config, spec):
    router = Router(config, spec, StubScreener([("card_arrival", 0.99)]))
    decision = router.route("T1", "where is my card")
    assert decision["status"] == "auto"
    assert decision["decided_by"] == "screener"
    assert decision["queue"] == "cards_ops"
    assert [receipt["tool"] for receipt in decision["executed"]] == ["create_case"]


def test_low_confidence_without_resolver_goes_to_a_human(config, spec):
    router = Router(config, spec, StubScreener([("card_arrival", 0.10)]))
    decision = router.route("T2", "something odd happened")
    assert decision["status"] == "human"
    assert decision["decided_by"] == "screener_low_confidence"
    assert [receipt["tool"] for receipt in decision["executed"]] == ["escalate_to_human"]


def test_risky_queue_never_automates_even_when_confident(config, spec):
    router = Router(config, spec, StubScreener([("lost_or_stolen_card", 0.999)]))
    decision = router.route("T3", "I lost my card")
    assert decision["status"] == "human"
    assert [receipt["tool"] for receipt in decision["executed"]] == ["freeze_card", "escalate_to_human"]


def test_resolver_takes_over_when_screener_is_unsure(config, spec):
    router = Router(config, spec, StubScreener([("card_arrival", 0.2)]), StubResolver([resolver_result("transfer_timing", 0.95)]))
    decision = router.route("T4", "how long does a transfer take")
    assert decision["decided_by"] == "resolver"
    assert decision["status"] == "auto"
    assert decision["queue"] == "transfers"


def test_resolver_low_confidence_still_reaches_a_human(config, spec):
    router = Router(config, spec, StubScreener([("card_arrival", 0.2)]), StubResolver([resolver_result("transfer_timing", 0.1)]))
    decision = router.route("T5", "unclear ticket")
    assert decision["decided_by"] == "resolver_low_confidence"
    assert decision["status"] == "human"


def test_invalid_resolver_json_falls_back_to_screener_and_records_the_error(config, spec):
    router = Router(config, spec, StubScreener([("card_arrival", 0.2)]), StubResolver([resolver_result("", 0.0, valid=False)]))
    decision = router.route("T6", "unparseable")
    assert decision["decided_by"] == "resolver_invalid"
    assert decision["status"] == "human"
    assert decision["schema_error"]


def test_thresholds_can_be_overridden(config, spec):
    strict = Router(config, spec, StubScreener([("card_arrival", 0.95)]), screener_threshold=0.99)
    assert strict.route("T7", "where is my card")["status"] == "human"
    relaxed = Router(config, spec, StubScreener([("card_arrival", 0.95)]), screener_threshold=0.5)
    assert relaxed.route("T7", "where is my card")["status"] == "auto"


def test_tools_are_idempotent_per_ticket(config, spec):
    router = Router(config, spec, StubScreener([("card_arrival", 0.99)]))
    first = router.route("SAME", "where is my card")
    second = router.route("SAME", "where is my card")
    assert first["executed"][0]["replayed"] is False
    assert second["executed"][0]["replayed"] is True
    assert first["executed"][0]["details"]["reference"] == second["executed"][0]["details"]["reference"]


def test_audit_log_records_every_executed_action(config, spec, tmp_path):
    log_path = tmp_path / "audit.jsonl"
    router = Router(config, spec, StubScreener([("card_arrival", 0.99)]), runner=ToolRunner(spec, log_path))
    router.route("T8", "where is my card")
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert entries[0]["receipt"]["tool"] == "create_case"
    assert entries[0]["decision"]["ticket_id"] == "T8"


def test_batch_routing_matches_single_routing(config, spec):
    answers = [("card_arrival", 0.99), ("lost_or_stolen_card", 0.99)]
    router = Router(config, spec, StubScreener(answers))
    decisions = router.route_batch([("B1", "where is my card"), ("B2", "I lost my card")])
    assert [decision["status"] for decision in decisions] == ["auto", "human"]
    report = summarize(decisions)
    assert report["tickets"] == 2
    assert report["automated"] == 1
    assert report["automation_rate"] == 0.5
