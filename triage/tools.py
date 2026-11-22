import hashlib
import json
import time


class ToolError(RuntimeError):
    pass


def reference(prefix, ticket_id, tool):
    digest = hashlib.blake2b(f"{ticket_id}:{tool}".encode("utf-8"), digest_size=4).hexdigest().upper()
    return f"{prefix}-{digest}"


class ToolRunner:
    def __init__(self, spec, log_path=None):
        self.spec = spec
        self.log_path = log_path
        self.receipts = {}
        self.handlers = {
            "create_case": self._create_case,
            "send_template_reply": self._send_template_reply,
            "send_reset_link": self._send_reset_link,
            "freeze_card": self._freeze_card,
            "escalate_to_human": self._escalate_to_human,
            "request_more_info": self._request_more_info,
        }

    def _write(self, entry):
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _create_case(self, decision):
        return {
            "reference": reference("CASE", decision["ticket_id"], "create_case"),
            "queue": decision["queue"],
            "team": decision["team"],
            "priority": decision["priority"],
            "entities": decision["entities"],
        }

    def _send_template_reply(self, decision):
        queue = self.spec.queue(decision["queue"])
        if not queue.template:
            raise ToolError(f"queue {queue.name} has no template configured")
        return {
            "reference": reference("MSG", decision["ticket_id"], "send_template_reply"),
            "template": queue.template,
            "characters": len(decision["reply"]),
        }

    def _send_reset_link(self, decision):
        return {
            "reference": reference("LINK", decision["ticket_id"], "send_reset_link"),
            "channel": "registered_email",
            "expires_minutes": 30,
        }

    def _freeze_card(self, decision):
        return {
            "reference": reference("FRZ", decision["ticket_id"], "freeze_card"),
            "card_last4": decision["entities"].get("card_last4", ""),
            "reversible": True,
        }

    def _escalate_to_human(self, decision):
        return {
            "reference": reference("ESC", decision["ticket_id"], "escalate_to_human"),
            "queue": decision["queue"],
            "team": decision["team"],
            "priority": decision["priority"],
            "confidence": decision["confidence"],
        }

    def _request_more_info(self, decision):
        missing = [key for key, value in decision["entities"].items() if not value]
        return {
            "reference": reference("INFO", decision["ticket_id"], "request_more_info"),
            "missing": missing,
        }

    def run(self, decision, actions):
        results = []
        for tool in actions:
            handler = self.handlers.get(tool)
            if handler is None:
                raise ToolError(f"no handler registered for {tool}")
            key = (decision["ticket_id"], tool)
            if key in self.receipts:
                replayed = dict(self.receipts[key])
                replayed["replayed"] = True
                results.append(replayed)
                continue
            started = time.perf_counter()
            receipt = {
                "tool": tool,
                "ticket_id": decision["ticket_id"],
                "details": handler(decision),
                "replayed": False,
                "milliseconds": round((time.perf_counter() - started) * 1000, 3),
            }
            self.receipts[key] = receipt
            self._write({"decision": {k: decision[k] for k in ("ticket_id", "intent", "queue", "priority", "confidence", "decided_by")}, "receipt": receipt})
            results.append(receipt)
        return results
