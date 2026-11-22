import json

from .entities import extract

QUEUE_SENTENCES = {
    "cards_ops": "I have raised this with our Cards Operations team so they can check the card for you.",
    "atm_cash": "I have passed this to our ATM and Cash team so they can review the withdrawal.",
    "payments_disputes": "I have opened a dispute case so our Payments Disputes team can review the charge.",
    "transfers": "I have sent this to our Transfers team so they can trace the payment.",
    "topups": "Our Top Ups team will check why the top up did not go through.",
    "fx_fees": "Our exchange rate is applied at the moment the transaction is processed, and any fee is shown with it.",
    "product_info": "Here is what we currently support for that, and you can also see the full list in the app.",
    "account_admin": "I have passed your request to our Account Administration team.",
    "account_access": "I have sent a secure reset link to the contact details on your account.",
    "identity_kyc": "A member of our Identity and KYC team will contact you to finish the check.",
    "security_fraud": "I have frozen the card as a precaution and our Fraud and Security team will call you.",
    "unsupported": "I could not match this to a banking request, so I am passing it to a support agent.",
}

COMPACT_SYSTEM = "You are the triage model for a bank support desk. Reply with one JSON object holding intent, entities, action and reply."


def build_reply(queue, entities, client_name):
    parts = [f"Thanks for contacting {client_name}."]
    parts.append(QUEUE_SENTENCES.get(queue, QUEUE_SENTENCES["unsupported"]))
    amount = entities.get("amount", "")
    currency = entities.get("currency", "")
    if amount and currency:
        parts.append(f"The amount I can see is {amount} {currency}.")
    elif amount:
        parts.append(f"The amount I can see is {amount}.")
    last4 = entities.get("card_last4", "")
    if last4:
        parts.append(f"This is for the card ending {last4}.")
    return " ".join(parts)


def target_payload(text, intent, spec):
    queue_name = spec.queue_of(intent)
    queue = spec.queue(queue_name)
    entities = extract(text, spec.entity_types)
    return {
        "intent": intent,
        "entities": entities,
        "action": queue.actions[0],
        "reply": build_reply(queue_name, entities, spec.client["name"]),
    }


def target_text(text, intent, spec):
    return json.dumps(target_payload(text, intent, spec), ensure_ascii=False)


def taxonomy_block(spec):
    lines = []
    for queue_name in sorted(spec.queues):
        intents = sorted(intent for intent, target in spec.intents.items() if target == queue_name)
        if not intents:
            continue
        lines.append(f"{queue_name}: {', '.join(intents)}")
    lines.append(f"{spec.out_of_scope_queue}: {spec.out_of_scope_label}")
    return "\n".join(lines)


def full_system(spec):
    actions = sorted({action for queue in spec.queues.values() for action in queue.actions})
    return (
        f"You triage support tickets for {spec.client['name']}.\n"
        f"Pick exactly one intent from the taxonomy below.\n\n"
        f"Taxonomy by queue:\n{taxonomy_block(spec)}\n\n"
        f"Allowed actions: {', '.join(actions)}.\n"
        f"Entity keys: {', '.join(spec.entity_types)}. Use an empty string when the ticket does not mention one.\n"
        f"Answer with a single JSON object and nothing else, using the keys intent, entities, action, reply."
    )


def build_messages(text, spec, style="compact", examples=None):
    system = COMPACT_SYSTEM if style == "compact" else full_system(spec)
    messages = [{"role": "system", "content": system}]
    for example_text, example_intent in examples or []:
        messages.append({"role": "user", "content": f"Ticket: {example_text}"})
        messages.append({"role": "assistant", "content": target_text(example_text, example_intent, spec)})
    messages.append({"role": "user", "content": f"Ticket: {text}"})
    return messages


def render_prompt(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    lines = [f"{message['role']}: {message['content']}" for message in messages]
    lines.append("assistant:")
    return "\n".join(lines)
