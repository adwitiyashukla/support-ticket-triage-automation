import json
import time

import numpy as np

from .config import load_config, load_spec
from .data import load_split
from .metrics import (
    accuracy,
    bootstrap_interval,
    expected_calibration_error,
    macro_f1,
    per_label_report,
    percentiles,
    reliability_table,
    roc_auc,
    top_confusions,
)
from .router import Router, ScreenerPredictor, summarize
from .tools import ToolRunner


def sweep_thresholds(confidences, correct, auto_allowed, steps, floor):
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    auto_allowed = np.asarray(auto_allowed, dtype=bool)
    grid = np.linspace(0.0, 1.0, steps)
    rows = []
    for threshold in grid:
        covered = auto_allowed & (confidences >= threshold)
        count = int(covered.sum())
        rows.append({
            "threshold": round(float(threshold), 4),
            "coverage": round(count / max(len(confidences), 1), 4),
            "accuracy_on_covered": round(float(correct[covered].mean()), 4) if count else 1.0,
            "covered": count,
        })
    eligible = [row for row in rows if row["accuracy_on_covered"] >= floor and row["covered"] > 0]
    recommended = min(eligible, key=lambda row: row["threshold"]) if eligible else None
    return rows, recommended


def screener_report(config, spec, split="test", log=print):
    predictor = ScreenerPredictor.load(config, spec)
    rows = load_split(config, split)
    texts = [row["text"] for row in rows]
    gold = np.array([row["intent"] for row in rows])
    labels = spec.labels()

    started = time.perf_counter()
    probabilities = predictor.probabilities(texts)
    batch_seconds = time.perf_counter() - started
    predicted = np.array([labels[index] for index in probabilities.argmax(axis=1)])
    confidence = probabilities.max(axis=1)
    correct = (predicted == gold).astype(float)
    auto_allowed = np.array([spec.is_auto(spec.queue_of(intent)) for intent in predicted])

    single = []
    for text in texts[:config.evaluation.latency_samples]:
        moment = time.perf_counter()
        predictor.probabilities([text])
        single.append((time.perf_counter() - moment) * 1000)
    single = single[config.evaluation.latency_warmup:]

    out_of_scope = (gold == spec.out_of_scope_label)
    out_of_scope_score = probabilities[:, labels.index(spec.out_of_scope_label)]
    predicted_out_of_scope = (predicted == spec.out_of_scope_label)
    sweep, recommended = sweep_thresholds(confidence, correct, auto_allowed, config.evaluation.threshold_steps, spec.policy["auto_accuracy_floor"])

    report = {
        "split": split,
        "rows": len(rows),
        "accuracy": round(accuracy(gold, predicted), 4),
        "accuracy_ci": bootstrap_interval(gold, predicted, accuracy),
        "macro_f1": round(macro_f1(gold, predicted, labels), 4),
        "in_scope_accuracy": round(accuracy(gold[~out_of_scope], predicted[~out_of_scope]), 4),
        "out_of_scope_recall": round(float(predicted_out_of_scope[out_of_scope].mean()), 4) if out_of_scope.any() else 0.0,
        "out_of_scope_precision": round(float(out_of_scope[predicted_out_of_scope].mean()), 4) if predicted_out_of_scope.any() else 0.0,
        "out_of_scope_auroc": round(roc_auc(out_of_scope_score, out_of_scope), 4),
        "uncertainty_auroc": round(roc_auc(1.0 - confidence, ~np.asarray(correct, dtype=bool)), 4),
        "ece": round(expected_calibration_error(confidence, correct, config.evaluation.calibration_bins), 4),
        "reliability": reliability_table(confidence, correct, config.evaluation.calibration_bins),
        "throughput_per_second_batched": round(len(texts) / max(batch_seconds, 1e-6), 1),
        "single_ticket_latency_ms": percentiles(single),
        "threshold_sweep": sweep,
        "recommended_threshold": recommended,
        "worst_labels": per_label_report(gold, predicted, labels)[:10],
        "top_confusions": top_confusions(gold, predicted),
    }
    log(f"screener {split} accuracy {report['accuracy']} macro f1 {report['macro_f1']} ece {report['ece']}")
    if recommended:
        log(f"screener threshold {recommended['threshold']} covers {recommended['coverage']} of tickets at {recommended['accuracy_on_covered']} accuracy")
    return report


def cost_model(config, spec, summary, resolver_prompt_tokens, resolver_output_tokens):
    client = spec.client
    tickets = client["tickets_per_day"]
    manual_hours = tickets * client["manual_minutes_per_ticket"] / 60.0
    manual_cost = manual_hours * client["agent_cost_per_hour_usd"]

    automated = summary.get("automation_rate", 0.0)
    saved_hours = manual_hours * automated
    per_ticket_seconds = summary.get("mean_latency_ms", 0.0) / 1000.0
    machine_hours = tickets * per_ticket_seconds / 3600.0
    machine_cost = machine_hours * config.cost.self_hosted_usd_per_hour
    machine_cost_per_thousand = machine_cost / max(tickets, 1) * 1000

    frontier_prompt = resolver_prompt_tokens or config.cost.frontier_prompt_tokens
    frontier_output = resolver_output_tokens or config.cost.frontier_output_tokens
    frontier_cost = tickets * (
        frontier_prompt / 1_000_000 * config.cost.frontier_input_usd_per_mtok
        + frontier_output / 1_000_000 * config.cost.frontier_output_usd_per_mtok
    )

    return {
        "tickets_per_day": tickets,
        "manual_agent_hours_per_day": round(manual_hours, 1),
        "manual_cost_per_day_usd": round(manual_cost, 2),
        "agent_hours_saved_per_day": round(saved_hours, 1),
        "labour_cost_avoided_per_day_usd": round(saved_hours * client["agent_cost_per_hour_usd"], 2),
        "compute_hours_per_day": round(machine_hours, 3),
        "compute_cost_per_day_usd": round(machine_cost, 4),
        "compute_cost_per_1000_tickets_usd": round(machine_cost_per_thousand, 4),
        "frontier_api_cost_per_day_usd": round(frontier_cost, 2),
        "frontier_api_cost_per_1000_tickets_usd": round(frontier_cost / max(tickets, 1) * 1000, 4),
        "frontier_assumption_usd_per_mtok": [config.cost.frontier_input_usd_per_mtok, config.cost.frontier_output_usd_per_mtok],
    }


def router_report(config, spec, use_resolver, adapter_dir=None, rows_limit=0, log=print):
    rows = load_split(config, "test")
    limit = rows_limit or config.evaluation.router_max_rows
    if limit:
        rows = rows[:limit]
    predictor = ScreenerPredictor.load(config, spec)

    resolver = None
    if use_resolver:
        from .resolver import ResolverModel

        resolver = ResolverModel.load(config, spec, adapter_dir or (config.paths.artifacts_dir / "resolver-adapter"))

    log_path = config.paths.reports_dir / "audit_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    router = Router(config, spec, predictor, resolver, ToolRunner(spec, log_path))

    tickets = [(f"T{index:06d}", row["text"]) for index, row in enumerate(rows)]
    started = time.perf_counter()
    decisions = router.route_batch(tickets)
    elapsed = time.perf_counter() - started

    single_router = Router(config, spec, predictor, resolver, ToolRunner(spec))
    single = []
    for index, row in enumerate(rows[:config.evaluation.latency_samples]):
        moment = time.perf_counter()
        single_router.route(f"L{index:06d}", row["text"])
        single.append((time.perf_counter() - moment) * 1000)
    single = single[config.evaluation.latency_warmup:]

    gold = np.array([row["intent"] for row in rows])
    predicted = np.array([decision["intent"] for decision in decisions])
    automated = np.array([decision["status"] == "auto" for decision in decisions])
    summary = summarize(decisions)
    summary["wall_seconds"] = round(elapsed, 2)
    summary["batched_tickets_per_second"] = round(len(decisions) / max(elapsed, 1e-6), 1)
    summary["single_ticket_latency_ms"] = percentiles(single)
    summary["accuracy_all"] = round(accuracy(gold, predicted), 4)
    summary["accuracy_on_automated"] = round(accuracy(gold[automated], predicted[automated]), 4) if automated.any() else 0.0
    summary["accuracy_on_handed_over"] = round(accuracy(gold[~automated], predicted[~automated]), 4) if (~automated).any() else 0.0
    summary["meets_accuracy_floor"] = bool(summary["accuracy_on_automated"] >= spec.policy["auto_accuracy_floor"])
    summary["meets_latency_budget"] = bool(summary["single_ticket_latency_ms"]["p95"] <= spec.policy["max_p95_latency_ms"])
    summary["action_policy_agreement"] = round(
        float(np.mean([decision["action_agrees_with_policy"] for decision in decisions if decision["model_action"]])), 4
    ) if any(decision["model_action"] for decision in decisions) else None
    summary["escalated_by_policy"] = int(sum(1 for decision in decisions if not decision["queue_allows_auto"]))
    summary["tool_calls"] = int(sum(len(decision["executed"]) for decision in decisions))

    resolver_decisions = [decision for decision in decisions if "prompt_tokens" in decision]
    if resolver_decisions:
        summary["resolver_calls"] = len(resolver_decisions)
        summary["mean_resolver_prompt_tokens"] = round(float(np.mean([decision["prompt_tokens"] for decision in resolver_decisions])), 1)
        summary["mean_resolver_generated_tokens"] = round(float(np.mean([decision["generated_tokens"] for decision in resolver_decisions])), 1)
        summary["resolver_repair_rate"] = round(float(np.mean([decision["repaired"] for decision in resolver_decisions])), 4)
    summary["cost"] = cost_model(config, spec, summary, 0.0, 0.0)
    log(f"router automation {summary['automation_rate']} accuracy on automated {summary['accuracy_on_automated']} p95 {summary['single_ticket_latency_ms']['p95']}ms")
    return summary, decisions


def run(config=None, spec=None, use_resolver=False, rows_limit=0, log=print):
    config = config or load_config()
    spec = spec or load_spec()
    config.paths.ensure()
    report = {"screener": screener_report(config, spec, "test", log)}
    summary, decisions = router_report(config, spec, use_resolver, rows_limit=rows_limit, log=log)
    report["router"] = summary
    sample = [
        {key: decision[key] for key in ("ticket_id", "intent", "queue", "priority", "confidence", "decided_by", "status", "reply")}
        for decision in decisions[:20]
    ]
    report["sample_decisions"] = sample
    (config.paths.reports_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
