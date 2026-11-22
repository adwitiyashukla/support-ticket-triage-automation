import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config, load_spec
from .router import Router, ScreenerPredictor
from .tools import ToolRunner

STATE = {}


class Ticket(BaseModel):
    ticket_id: str = Field(default="", max_length=64)
    text: str = Field(min_length=1, max_length=2000)


class TicketBatch(BaseModel):
    tickets: list[Ticket] = Field(min_length=1, max_length=64)


def build_router(use_resolver=None):
    config = load_config()
    spec = load_spec()
    config.paths.ensure()
    predictor = ScreenerPredictor.load(config, spec)
    resolver = None
    wanted = os.environ.get("TRIAGE_USE_RESOLVER", "0") == "1" if use_resolver is None else use_resolver
    adapter = config.paths.artifacts_dir / "resolver-adapter"
    if wanted and adapter.exists():
        from .resolver import ResolverModel

        resolver = ResolverModel.load(config, spec, adapter)
    runner = ToolRunner(spec, config.paths.reports_dir / "audit_log.jsonl")
    return Router(config, spec, predictor, resolver, runner), config, spec


@asynccontextmanager
async def lifespan(app):
    router, config, spec = build_router()
    STATE["router"] = router
    STATE["config"] = config
    STATE["spec"] = spec
    STATE["counters"] = {"requests": 0, "auto": 0, "human": 0}
    yield
    STATE.clear()


app = FastAPI(title="support ticket triage", version="1.0", lifespan=lifespan)


def _record(decisions):
    counters = STATE["counters"]
    counters["requests"] += len(decisions)
    counters["auto"] += sum(1 for decision in decisions if decision["status"] == "auto")
    counters["human"] += sum(1 for decision in decisions if decision["status"] != "auto")


@app.get("/healthz")
def healthz():
    router = STATE.get("router")
    if router is None:
        raise HTTPException(status_code=503, detail="router not loaded")
    return {"status": "ok", "resolver_loaded": router.resolver is not None, "labels": len(router.screener.labels)}


@app.get("/metrics")
def metrics():
    counters = STATE.get("counters", {})
    requests = counters.get("requests", 0)
    return {
        "requests": requests,
        "auto": counters.get("auto", 0),
        "human": counters.get("human", 0),
        "automation_rate": round(counters.get("auto", 0) / requests, 4) if requests else 0.0,
    }


@app.post("/triage")
def triage(ticket: Ticket):
    router = STATE.get("router")
    if router is None:
        raise HTTPException(status_code=503, detail="router not loaded")
    ticket_id = ticket.ticket_id or f"T{STATE['counters']['requests']:06d}"
    decision = router.route(ticket_id, ticket.text)
    _record([decision])
    return decision


@app.post("/triage/batch")
def triage_batch(batch: TicketBatch):
    router = STATE.get("router")
    if router is None:
        raise HTTPException(status_code=503, detail="router not loaded")
    offset = STATE["counters"]["requests"]
    tickets = [(item.ticket_id or f"T{offset + index:06d}", item.text) for index, item in enumerate(batch.tickets)]
    decisions = router.route_batch(tickets)
    _record(decisions)
    return {"decisions": decisions}
