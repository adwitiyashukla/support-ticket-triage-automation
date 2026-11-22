import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ConfigError(Exception):
    pass


def _check(condition, message):
    if not condition:
        raise ConfigError(message)


def _fraction(value, name):
    _check(isinstance(value, float) and 0.0 < value < 1.0, f"{name} must be a float in (0, 1)")
    return value


def _positive_int(value, name):
    _check(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{name} must be a positive integer")
    return value


def _non_negative_int(value, name):
    _check(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{name} must be a non negative integer")
    return value


def _positive_float(value, name):
    _check(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0, f"{name} must be positive")
    return float(value)


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    reports_dir: Path

    def ensure(self):
        for directory in (self.raw_dir, self.processed_dir, self.artifacts_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    seed: int
    val_fraction: float
    oos_train: int
    oos_val: int
    oos_test: int
    min_chars: int
    max_chars: int
    shingle_size: int
    near_dup_jaccard: float


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int
    min_pair_freq: int


@dataclass(frozen=True)
class ScreenerConfig:
    n_layer: int
    n_head: int
    d_model: int
    d_ff: int
    max_seq_len: int
    dropout: float
    rope_theta: float
    pretrain_epochs: int
    pretrain_lr: float
    finetune_epochs: int
    finetune_lr: float
    batch_size: int
    weight_decay: float
    grad_clip: float
    warmup_ratio: float
    label_smoothing: float
    skip_pretrain: bool
    device: str
    threads: int


@dataclass(frozen=True)
class ResolverConfig:
    base_model: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple
    epochs: int
    lr: float
    batch_size: int
    grad_accum: int
    max_len: int
    max_new_tokens: int
    infer_batch_size: int
    train_subset: int
    repair_attempts: int
    device: str
    dtype: str


@dataclass(frozen=True)
class BaselineConfig:
    word_ngram_max: int
    char_ngram_min: int
    char_ngram_max: int
    min_df: int
    logreg_c: float
    logreg_max_iter: int
    fewshot_examples: int
    neighbours: int


@dataclass(frozen=True)
class EvalConfig:
    calibration_bins: int
    threshold_steps: int
    latency_warmup: int
    latency_samples: int
    router_max_rows: int
    resolver_eval_rows: int


@dataclass(frozen=True)
class CostConfig:
    frontier_input_usd_per_mtok: float
    frontier_output_usd_per_mtok: float
    frontier_prompt_tokens: int
    frontier_output_tokens: int
    self_hosted_usd_per_hour: float
    self_hosted_tickets_per_hour: int


@dataclass(frozen=True)
class Config:
    paths: Paths
    data: DataConfig
    tokenizer: TokenizerConfig
    screener: ScreenerConfig
    resolver: ResolverConfig
    baselines: BaselineConfig
    evaluation: EvalConfig
    cost: CostConfig


@dataclass(frozen=True)
class Queue:
    name: str
    team: str
    priority: str
    actions: tuple
    auto: bool
    template: str


@dataclass(frozen=True)
class Spec:
    client: dict
    policy: dict
    entity_types: tuple
    queues: dict
    intents: dict

    @property
    def out_of_scope_label(self):
        return self.policy["out_of_scope_label"]

    @property
    def out_of_scope_queue(self):
        return self.policy["out_of_scope_queue"]

    def labels(self):
        return sorted(self.intents) + [self.out_of_scope_label]

    def queue_of(self, intent):
        if intent == self.out_of_scope_label:
            return self.out_of_scope_queue
        if intent not in self.intents:
            raise ConfigError(f"intent {intent} is not in the spec")
        return self.intents[intent]

    def queue(self, name):
        if name not in self.queues:
            raise ConfigError(f"queue {name} is not in the spec")
        return self.queues[name]

    def is_auto(self, queue_name):
        return self.queue(queue_name).auto


def _read_toml(path):
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"missing file {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.toml"
    raw = _read_toml(path)
    for section in ("paths", "data", "tokenizer", "screener", "resolver", "baselines", "eval", "cost"):
        _check(section in raw, f"config is missing the [{section}] section")

    paths = Paths(
        raw_dir=ROOT / raw["paths"]["raw_dir"],
        processed_dir=ROOT / raw["paths"]["processed_dir"],
        artifacts_dir=ROOT / raw["paths"]["artifacts_dir"],
        reports_dir=ROOT / raw["paths"]["reports_dir"],
    )

    d = raw["data"]
    data = DataConfig(
        seed=_non_negative_int(d["seed"], "data.seed"),
        val_fraction=_fraction(d["val_fraction"], "data.val_fraction"),
        oos_train=_non_negative_int(d["oos_train"], "data.oos_train"),
        oos_val=_non_negative_int(d["oos_val"], "data.oos_val"),
        oos_test=_non_negative_int(d["oos_test"], "data.oos_test"),
        min_chars=_positive_int(d["min_chars"], "data.min_chars"),
        max_chars=_positive_int(d["max_chars"], "data.max_chars"),
        shingle_size=_positive_int(d["shingle_size"], "data.shingle_size"),
        near_dup_jaccard=_fraction(d["near_dup_jaccard"], "data.near_dup_jaccard"),
    )
    _check(data.min_chars < data.max_chars, "data.min_chars must be below data.max_chars")

    t = raw["tokenizer"]
    tokenizer = TokenizerConfig(
        vocab_size=_positive_int(t["vocab_size"], "tokenizer.vocab_size"),
        min_pair_freq=_positive_int(t["min_pair_freq"], "tokenizer.min_pair_freq"),
    )
    _check(tokenizer.vocab_size > 260, "tokenizer.vocab_size must exceed the byte alphabet plus specials")

    z = raw["screener"]
    screener = ScreenerConfig(
        n_layer=_positive_int(z["n_layer"], "screener.n_layer"),
        n_head=_positive_int(z["n_head"], "screener.n_head"),
        d_model=_positive_int(z["d_model"], "screener.d_model"),
        d_ff=_positive_int(z["d_ff"], "screener.d_ff"),
        max_seq_len=_positive_int(z["max_seq_len"], "screener.max_seq_len"),
        dropout=float(z["dropout"]),
        rope_theta=_positive_float(z["rope_theta"], "screener.rope_theta"),
        pretrain_epochs=_non_negative_int(z["pretrain_epochs"], "screener.pretrain_epochs"),
        pretrain_lr=_positive_float(z["pretrain_lr"], "screener.pretrain_lr"),
        finetune_epochs=_positive_int(z["finetune_epochs"], "screener.finetune_epochs"),
        finetune_lr=_positive_float(z["finetune_lr"], "screener.finetune_lr"),
        batch_size=_positive_int(z["batch_size"], "screener.batch_size"),
        weight_decay=float(z["weight_decay"]),
        grad_clip=_positive_float(z["grad_clip"], "screener.grad_clip"),
        warmup_ratio=_fraction(z["warmup_ratio"], "screener.warmup_ratio"),
        label_smoothing=float(z["label_smoothing"]),
        skip_pretrain=bool(z["skip_pretrain"]),
        device=str(z["device"]),
        threads=_non_negative_int(z["threads"], "screener.threads"),
    )
    _check(0.0 <= screener.dropout < 1.0, "screener.dropout must be in [0, 1)")
    _check(0.0 <= screener.label_smoothing < 0.5, "screener.label_smoothing must be in [0, 0.5)")
    _check(screener.d_model % screener.n_head == 0, "screener.d_model must divide evenly by screener.n_head")
    _check((screener.d_model // screener.n_head) % 2 == 0, "screener head dimension must be even for rotary embeddings")

    o = raw["resolver"]
    resolver = ResolverConfig(
        base_model=str(o["base_model"]),
        lora_r=_positive_int(o["lora_r"], "resolver.lora_r"),
        lora_alpha=_positive_int(o["lora_alpha"], "resolver.lora_alpha"),
        lora_dropout=float(o["lora_dropout"]),
        lora_targets=tuple(o["lora_targets"]),
        epochs=_positive_int(o["epochs"], "resolver.epochs"),
        lr=_positive_float(o["lr"], "resolver.lr"),
        batch_size=_positive_int(o["batch_size"], "resolver.batch_size"),
        grad_accum=_positive_int(o["grad_accum"], "resolver.grad_accum"),
        max_len=_positive_int(o["max_len"], "resolver.max_len"),
        max_new_tokens=_positive_int(o["max_new_tokens"], "resolver.max_new_tokens"),
        infer_batch_size=_positive_int(o["infer_batch_size"], "resolver.infer_batch_size"),
        train_subset=_non_negative_int(o["train_subset"], "resolver.train_subset"),
        repair_attempts=_non_negative_int(o["repair_attempts"], "resolver.repair_attempts"),
        device=str(o["device"]),
        dtype=str(o["dtype"]),
    )
    _check(len(resolver.lora_targets) > 0, "resolver.lora_targets must not be empty")
    _check(0.0 <= resolver.lora_dropout < 1.0, "resolver.lora_dropout must be in [0, 1)")

    b = raw["baselines"]
    baselines = BaselineConfig(
        word_ngram_max=_positive_int(b["word_ngram_max"], "baselines.word_ngram_max"),
        char_ngram_min=_positive_int(b["char_ngram_min"], "baselines.char_ngram_min"),
        char_ngram_max=_positive_int(b["char_ngram_max"], "baselines.char_ngram_max"),
        min_df=_positive_int(b["min_df"], "baselines.min_df"),
        logreg_c=_positive_float(b["logreg_c"], "baselines.logreg_c"),
        logreg_max_iter=_positive_int(b["logreg_max_iter"], "baselines.logreg_max_iter"),
        fewshot_examples=_positive_int(b["fewshot_examples"], "baselines.fewshot_examples"),
        neighbours=_positive_int(b["neighbours"], "baselines.neighbours"),
    )
    _check(baselines.char_ngram_min <= baselines.char_ngram_max, "baselines char ngram range is inverted")

    e = raw["eval"]
    evaluation = EvalConfig(
        calibration_bins=_positive_int(e["calibration_bins"], "eval.calibration_bins"),
        threshold_steps=_positive_int(e["threshold_steps"], "eval.threshold_steps"),
        latency_warmup=_non_negative_int(e["latency_warmup"], "eval.latency_warmup"),
        latency_samples=_positive_int(e["latency_samples"], "eval.latency_samples"),
        router_max_rows=_non_negative_int(e["router_max_rows"], "eval.router_max_rows"),
        resolver_eval_rows=_non_negative_int(e["resolver_eval_rows"], "eval.resolver_eval_rows"),
    )
    _check(evaluation.threshold_steps >= 3, "eval.threshold_steps must be at least 3")

    c = raw["cost"]
    cost = CostConfig(
        frontier_input_usd_per_mtok=_positive_float(c["frontier_input_usd_per_mtok"], "cost.frontier_input_usd_per_mtok"),
        frontier_output_usd_per_mtok=_positive_float(c["frontier_output_usd_per_mtok"], "cost.frontier_output_usd_per_mtok"),
        frontier_prompt_tokens=_positive_int(c["frontier_prompt_tokens"], "cost.frontier_prompt_tokens"),
        frontier_output_tokens=_positive_int(c["frontier_output_tokens"], "cost.frontier_output_tokens"),
        self_hosted_usd_per_hour=_positive_float(c["self_hosted_usd_per_hour"], "cost.self_hosted_usd_per_hour"),
        self_hosted_tickets_per_hour=_positive_int(c["self_hosted_tickets_per_hour"], "cost.self_hosted_tickets_per_hour"),
    )

    return Config(
        paths=paths,
        data=data,
        tokenizer=tokenizer,
        screener=screener,
        resolver=resolver,
        baselines=baselines,
        evaluation=evaluation,
        cost=cost,
    )


def load_spec(path=None):
    path = Path(path) if path else ROOT / "spec.toml"
    raw = _read_toml(path)
    for section in ("client", "policy", "entities", "queues", "intents"):
        _check(section in raw, f"spec is missing the [{section}] section")

    client = raw["client"]
    for key in ("name", "process", "tickets_per_day", "manual_minutes_per_ticket", "agent_cost_per_hour_usd"):
        _check(key in client, f"spec client is missing {key}")
    _positive_int(client["tickets_per_day"], "client.tickets_per_day")
    _positive_float(client["manual_minutes_per_ticket"], "client.manual_minutes_per_ticket")
    _positive_float(client["agent_cost_per_hour_usd"], "client.agent_cost_per_hour_usd")

    policy = raw["policy"]
    for key in ("auto_accuracy_floor", "max_p95_latency_ms", "screener_min_confidence", "resolver_min_confidence", "out_of_scope_label", "out_of_scope_queue"):
        _check(key in policy, f"spec policy is missing {key}")
    _fraction(policy["auto_accuracy_floor"], "policy.auto_accuracy_floor")
    _fraction(policy["screener_min_confidence"], "policy.screener_min_confidence")
    _fraction(policy["resolver_min_confidence"], "policy.resolver_min_confidence")
    _positive_int(policy["max_p95_latency_ms"], "policy.max_p95_latency_ms")

    entity_types = tuple(raw["entities"]["types"])
    _check(len(entity_types) > 0, "spec entities.types must not be empty")

    known_actions = {"create_case", "send_template_reply", "send_reset_link", "freeze_card", "escalate_to_human", "request_more_info"}
    queues = {}
    for name, body in raw["queues"].items():
        for key in ("team", "priority", "actions", "auto"):
            _check(key in body, f"queue {name} is missing {key}")
        actions = tuple(body["actions"])
        _check(len(actions) > 0, f"queue {name} has no actions")
        unknown = sorted(set(actions) - known_actions)
        _check(not unknown, f"queue {name} uses unknown actions {unknown}")
        _check(body["priority"] in {"P1", "P2", "P3", "P4"}, f"queue {name} has an invalid priority")
        template = body.get("template", "")
        if "send_template_reply" in actions:
            _check(bool(template), f"queue {name} sends a template reply but declares no template")
        queues[name] = Queue(
            name=name,
            team=str(body["team"]),
            priority=str(body["priority"]),
            actions=actions,
            auto=bool(body["auto"]),
            template=str(template),
        )

    intents = dict(raw["intents"])
    _check(len(intents) > 0, "spec intents must not be empty")
    unmapped = sorted({q for q in intents.values() if q not in queues})
    _check(not unmapped, f"intents point at queues that do not exist: {unmapped}")
    _check(policy["out_of_scope_queue"] in queues, "policy.out_of_scope_queue is not a declared queue")
    _check(policy["out_of_scope_label"] not in intents, "policy.out_of_scope_label collides with a real intent")
    unused = sorted(set(queues) - set(intents.values()) - {policy["out_of_scope_queue"]})
    _check(not unused, f"queues declared but never routed to: {unused}")

    return Spec(client=client, policy=policy, entity_types=entity_types, queues=queues, intents=intents)
