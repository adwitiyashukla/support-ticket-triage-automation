import json
import re

REQUIRED_KEYS = ("intent", "entities", "action", "reply")
JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
MAX_REPLY_CHARS = 600


class SchemaError(ValueError):
    pass


def extract_json(raw):
    if raw is None:
        raise SchemaError("model returned nothing")
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    match = JSON_BLOCK.search(text)
    if not match:
        raise SchemaError("no json object found in the model output")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise SchemaError(f"invalid json: {error.msg}") from error
    if not isinstance(payload, dict):
        raise SchemaError("model output is not a json object")
    return payload


def validate(payload, spec):
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise SchemaError(f"missing keys {missing}")

    intent = payload["intent"]
    if not isinstance(intent, str):
        raise SchemaError("intent must be a string")
    intent = intent.strip()
    if intent not in spec.intents and intent != spec.out_of_scope_label:
        raise SchemaError(f"unknown intent {intent}")

    entities = payload["entities"]
    if not isinstance(entities, dict):
        raise SchemaError("entities must be an object")
    cleaned = {}
    for key in spec.entity_types:
        value = entities.get(key, "")
        if value is None:
            value = ""
        if not isinstance(value, (str, int, float)):
            raise SchemaError(f"entity {key} must be a scalar")
        cleaned[key] = str(value).strip()

    action = payload["action"]
    if not isinstance(action, str) or not action.strip():
        raise SchemaError("action must be a non empty string")

    reply = payload["reply"]
    if not isinstance(reply, str) or not reply.strip():
        raise SchemaError("reply must be a non empty string")

    return {"intent": intent, "entities": cleaned, "action": action.strip(), "reply": reply.strip()[:MAX_REPLY_CHARS]}


def parse(raw, spec):
    return validate(extract_json(raw), spec)


def policy_for(intent, spec):
    queue_name = spec.queue_of(intent)
    queue = spec.queue(queue_name)
    return {
        "queue": queue_name,
        "team": queue.team,
        "priority": queue.priority,
        "actions": list(queue.actions),
        "auto": queue.auto,
        "template": queue.template,
    }


def build_decision(ticket_id, text, intent, entities, reply, confidence, decided_by, spec, model_action=""):
    policy = policy_for(intent, spec)
    return {
        "ticket_id": ticket_id,
        "text": text,
        "intent": intent,
        "queue": policy["queue"],
        "team": policy["team"],
        "priority": policy["priority"],
        "entities": dict(entities),
        "actions": policy["actions"],
        "model_action": model_action,
        "action_agrees_with_policy": bool(model_action) and model_action in policy["actions"],
        "reply": reply,
        "confidence": round(float(confidence), 4),
        "decided_by": decided_by,
        "queue_allows_auto": policy["auto"],
    }
