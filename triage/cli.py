import argparse
import json

from .config import load_config, load_spec


def _print(payload):
    print(json.dumps(payload, indent=2))


def cmd_prepare(args):
    from .data import build

    _print(build())


def cmd_tokenizer(args):
    from .tokenizer import compression_stats
    from .train_screener import get_tokenizer
    from .data import load_split

    config = load_config()
    tokenizer = get_tokenizer(config, rebuild=args.rebuild)
    rows = load_split(config, "val")
    _print({"vocab_size": tokenizer.vocab_size, "merges": len(tokenizer.merges), **compression_stats(tokenizer, [row["text"] for row in rows])})


def cmd_train_screener(args):
    from .train_screener import train

    _print(train())


def cmd_train_resolver(args):
    from .train_resolver import train

    _print(train())


def cmd_baselines(args):
    from .baselines import run

    run(include_prompt=args.prompt, prompt_rows=args.rows)


def cmd_evaluate(args):
    from .evaluate import run

    report = run(use_resolver=args.resolver, rows_limit=args.rows)
    _print({"screener": {key: report["screener"][key] for key in ("accuracy", "macro_f1", "ece", "out_of_scope_auroc")}, "router": report["router"]})


def cmd_queue(args):
    from .router import process_file, summarize
    from .serve import build_router

    router, config, spec = build_router(use_resolver=args.resolver)
    source = args.input or (config.paths.processed_dir / "queue.jsonl")
    destination = config.paths.reports_dir / "queue_decisions.jsonl"
    decisions = process_file(config, spec, router, source, destination, args.limit)
    _print(summarize(decisions))


def cmd_serve(args):
    import uvicorn

    uvicorn.run("triage.serve:app", host=args.host, port=args.port, log_level="info")


def cmd_check(args):
    config = load_config()
    spec = load_spec()
    _print({
        "client": spec.client["name"],
        "intents": len(spec.intents),
        "queues": sorted(spec.queues),
        "auto_queues": sorted(name for name, queue in spec.queues.items() if queue.auto),
        "paths": {key: str(value) for key, value in vars(config.paths).items()},
    })


def build_parser():
    parser = argparse.ArgumentParser(prog="triage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check").set_defaults(handler=cmd_check)
    subparsers.add_parser("prepare").set_defaults(handler=cmd_prepare)

    tokenizer = subparsers.add_parser("tokenizer")
    tokenizer.add_argument("--rebuild", action="store_true")
    tokenizer.set_defaults(handler=cmd_tokenizer)

    subparsers.add_parser("train-screener").set_defaults(handler=cmd_train_screener)
    subparsers.add_parser("train-resolver").set_defaults(handler=cmd_train_resolver)

    baselines = subparsers.add_parser("baselines")
    baselines.add_argument("--prompt", action="store_true")
    baselines.add_argument("--rows", type=int, default=0)
    baselines.set_defaults(handler=cmd_baselines)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--resolver", action="store_true")
    evaluate.add_argument("--rows", type=int, default=0)
    evaluate.set_defaults(handler=cmd_evaluate)

    queue = subparsers.add_parser("queue")
    queue.add_argument("--input", default="")
    queue.add_argument("--limit", type=int, default=0)
    queue.add_argument("--resolver", action="store_true")
    queue.set_defaults(handler=cmd_queue)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=cmd_serve)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)

