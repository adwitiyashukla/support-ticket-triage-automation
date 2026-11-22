import time

import numpy as np
import torch

from .model import ScreenerModel
from .schema import build_decision, policy_for
from .tokenizer import ByteBPE
from .tools import ToolRunner
from .train_screener import collate, resolve_device

ESCALATE = "escalate_to_human"


class ScreenerPredictor:
    def __init__(self, model, tokenizer, labels, temperature, device, max_seq_len, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.temperature = temperature
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    @classmethod
    def load(cls, config, spec):
        path = config.paths.artifacts_dir / "screener.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}, train screener first")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        model = ScreenerModel(**artifact["model_args"], pad_id=artifact["pad_id"])
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        device = resolve_device(config.screener.device)
        model.to(device)
        tokenizer = ByteBPE.load(config.paths.artifacts_dir / "tokenizer.json")
        return cls(model, tokenizer, artifact["labels"], artifact["temperature"], device, config.screener.max_seq_len, config.screener.batch_size)

    def probabilities(self, texts):
        sequences = [self.tokenizer.encode(text, add_bos=True, add_eos=True)[:self.max_seq_len] or [self.tokenizer.bos_id] for text in texts]
        chunks = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                indices = range(start, min(start + self.batch_size, len(sequences)))
                ids, mask = collate(sequences, indices, self.tokenizer.pad_id)
                logits = self.model.forward_cls(ids.to(self.device), mask.to(self.device)).cpu()
                chunks.append((logits / self.temperature).softmax(dim=1))
        return torch.cat(chunks, dim=0).numpy()

    def predict(self, texts):
        started = time.perf_counter()
        probabilities = self.probabilities(texts)
        elapsed = (time.perf_counter() - started) * 1000 / max(len(texts), 1)
        best = probabilities.argmax(axis=1)
        return [
            {"intent": self.labels[int(index)], "confidence": float(probabilities[row, index]), "latency_ms": round(elapsed, 3)}
            for row, index in enumerate(best)
        ]


class Router:
    def __init__(self, config, spec, screener, resolver=None, runner=None, screener_threshold=None, resolver_threshold=None):
        self.config = config
        self.spec = spec
        self.screener = screener
        self.resolver = resolver
        self.runner = runner or ToolRunner(spec)
        self.screener_threshold = spec.policy["screener_min_confidence"] if screener_threshold is None else screener_threshold
        self.resolver_threshold = spec.policy["resolver_min_confidence"] if resolver_threshold is None else resolver_threshold

    def _finish(self, ticket_id, text, intent, entities, reply, confidence, decided_by, model_action, auto, timings):
        decision = build_decision(ticket_id, text, intent, entities, reply, confidence, decided_by, self.spec, model_action)
        allows_auto = decision["queue_allows_auto"]
        status = "auto" if (auto and allows_auto) else "human"
        actions = list(decision["actions"]) if (status == "auto" or not allows_auto) else [ESCALATE]
        decision["status"] = status
        decision["executed"] = self.runner.run(decision, actions)
        decision["latency_ms"] = round(sum(timings.values()), 3)
        decision["stage_latency_ms"] = {key: round(value, 3) for key, value in timings.items()}
        return decision

    def route_batch(self, tickets):
        texts = [text for _, text in tickets]
        if not texts:
            return []
        screener_results = self.screener.predict(texts)
        deferred = [index for index, result in enumerate(screener_results) if result["confidence"] < self.screener_threshold]
        deferred_set = set(deferred)

        resolver_results = {}
        resolver_latency = 0.0
        if deferred and self.resolver is not None:
            started = time.perf_counter()
            predictions = self.resolver.predict([texts[index] for index in deferred])
            resolver_latency = (time.perf_counter() - started) * 1000 / len(deferred)
            resolver_results = dict(zip(deferred, predictions))

        decisions = []
        for index, (ticket_id, text) in enumerate(tickets):
            base = screener_results[index]
            timings = {"screener": base["latency_ms"]}
            if index not in deferred_set:
                entities = self.spec_entities(text)
                reply = self.default_reply(base["intent"], entities)
                decisions.append(self._finish(ticket_id, text, base["intent"], entities, reply, base["confidence"], "screener", "", True, timings))
                continue
            prediction = resolver_results.get(index)
            timings["resolver"] = resolver_latency
            if prediction is None:
                entities = self.spec_entities(text)
                reply = self.default_reply(base["intent"], entities)
                decisions.append(self._finish(ticket_id, text, base["intent"], entities, reply, base["confidence"], "screener_low_confidence", "", False, timings))
                continue
            if not prediction["valid"]:
                entities = self.spec_entities(text)
                reply = self.default_reply(base["intent"], entities)
                decision = self._finish(ticket_id, text, base["intent"], entities, reply, base["confidence"], "resolver_invalid", "", False, timings)
                decision["schema_error"] = prediction["error"]
                decisions.append(decision)
                continue
            auto = prediction["confidence"] >= self.resolver_threshold
            decision = self._finish(
                ticket_id, text, prediction["intent"], prediction["entities"], prediction["reply"],
                prediction["confidence"], "resolver" if auto else "resolver_low_confidence", prediction["action"], auto, timings,
            )
            decision["prompt_tokens"] = prediction["prompt_tokens"]
            decision["generated_tokens"] = prediction["generated_tokens"]
            decision["repaired"] = prediction["repaired"]
            decisions.append(decision)
        return decisions

    def route(self, ticket_id, text):
        return self.route_batch([(ticket_id, text)])[0]

    def spec_entities(self, text):
        from .entities import extract

        return extract(text, self.spec.entity_types)

    def default_reply(self, intent, entities):
        from .prompts import build_reply

        return build_reply(policy_for(intent, self.spec)["queue"], entities, self.spec.client["name"])


def summarize(decisions):
    if not decisions:
        return {}
    automated = [decision for decision in decisions if decision["status"] == "auto"]
    by_stage = {}
    for decision in decisions:
        by_stage[decision["decided_by"]] = by_stage.get(decision["decided_by"], 0) + 1
    latencies = np.array([decision["latency_ms"] for decision in decisions], dtype=float)
    return {
        "tickets": len(decisions),
        "automated": len(automated),
        "automation_rate": round(len(automated) / len(decisions), 4),
        "handed_to_human": len(decisions) - len(automated),
        "decided_by": dict(sorted(by_stage.items())),
        "mean_latency_ms": round(float(latencies.mean()), 3),
    }


def process_file(config, spec, router, source, destination, limit=0):
    import json

    rows = []
    with open(source, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    tickets = [(row.get("ticket_id") or f"T{index:06d}", row["text"]) for index, row in enumerate(rows)]
    decisions = router.route_batch(tickets)
    with open(destination, "w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    return decisions
