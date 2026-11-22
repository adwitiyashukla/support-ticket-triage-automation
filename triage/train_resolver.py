import json
import math
import random
import time

import torch
import torch.nn.functional as F

from .config import load_config, load_spec
from .data import load_split
from .metrics import accuracy
from .prompts import build_messages, render_prompt, target_text
from .resolver import ResolverModel, resolve_device, resolve_dtype

IGNORE_INDEX = -100


def build_examples(rows, tokenizer, spec, max_len):
    examples = []
    for row in rows:
        prompt = render_prompt(tokenizer, build_messages(row["text"], spec, "compact"))
        target = target_text(row["text"], row["intent"], spec)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        ids = (prompt_ids + target_ids)[:max_len]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + target_ids)[:max_len]
        if all(label == IGNORE_INDEX for label in labels):
            continue
        examples.append((ids, labels))
    return examples


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for position, (ids, target) in enumerate(batch):
        input_ids[position, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[position, :len(target)] = torch.tensor(target, dtype=torch.long)
        attention[position, :len(ids)] = 1
    return input_ids, attention, labels


def batch_loss(model, input_ids, attention, labels):
    logits = model(input_ids=input_ids, attention_mask=attention).logits
    shifted_logits = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
    shifted_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shifted_logits, shifted_labels, ignore_index=IGNORE_INDEX)


def evaluate_loss(model, examples, pad_id, batch_size, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            input_ids, attention, labels = collate(chunk, pad_id)
            loss = batch_loss(model, input_ids.to(device), attention.to(device), labels.to(device))
            total += float(loss) * len(chunk)
            count += len(chunk)
    return total / max(count, 1)


def train(config=None, spec=None, log=print):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = config or load_config()
    spec = spec or load_spec()
    config.paths.ensure()
    settings = config.resolver
    device = resolve_device(settings.device)
    dtype = resolve_dtype(settings.dtype, device)
    random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_rows = load_split(config, "train")
    val_rows = load_split(config, "val")
    if settings.train_subset:
        random.Random(config.data.seed).shuffle(train_rows)
        train_rows = train_rows[:settings.train_subset]

    started = time.time()
    train_examples = build_examples(train_rows, tokenizer, spec, settings.max_len)
    val_examples = build_examples(val_rows[:400], tokenizer, spec, settings.max_len)
    log(f"resolver training examples {len(train_examples)} validation examples {len(val_examples)}")

    model = AutoModelForCausalLM.from_pretrained(settings.base_model, torch_dtype=dtype)
    model.config.use_cache = False
    lora = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=list(settings.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    log(f"resolver base {settings.base_model} trainable {trainable:,} of {total:,} on {device}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_examples) / (settings.batch_size * settings.grad_accum))
    total_steps = max(1, steps_per_epoch * settings.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=settings.lr, total_steps=total_steps, pct_start=0.06, anneal_strategy="cos")

    history = []
    for epoch in range(settings.epochs):
        model.train()
        order = list(range(len(train_examples)))
        random.Random(config.data.seed + epoch).shuffle(order)
        running, seen, micro = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(order), settings.batch_size):
            chunk = [train_examples[index] for index in order[start:start + settings.batch_size]]
            input_ids, attention, labels = collate(chunk, tokenizer.pad_token_id)
            loss = batch_loss(model, input_ids.to(device), attention.to(device), labels.to(device))
            (loss / settings.grad_accum).backward()
            running += float(loss.detach()) * len(chunk)
            seen += len(chunk)
            micro += 1
            if micro % settings.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()
            if micro % (settings.grad_accum * 25) == 0:
                log(f"resolver epoch {epoch + 1} example {seen}/{len(train_examples)} loss {running / max(seen, 1):.4f}")
        validation = evaluate_loss(model, val_examples, tokenizer.pad_token_id, settings.batch_size, device)
        entry = {"epoch": epoch + 1, "train_loss": round(running / max(seen, 1), 4), "val_loss": round(validation, 4)}
        history.append(entry)
        log(f"resolver epoch {entry['epoch']}/{settings.epochs} train loss {entry['train_loss']} val loss {entry['val_loss']}")

    adapter_dir = config.paths.artifacts_dir / "resolver-adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    model.config.use_cache = True
    tokenizer.padding_side = "left"
    checker = ResolverModel(model, tokenizer, spec, config, device)
    sample = val_rows[:min(len(val_rows), 64)]
    predictions = checker.predict([row["text"] for row in sample])
    valid = [item for item in predictions if item["valid"]]
    gold = [row["intent"] for row, item in zip(sample, predictions) if item["valid"]]
    guessed = [item["intent"] for item in valid]

    report = {
        "base_model": settings.base_model,
        "device": device,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_examples": len(train_examples),
        "epochs": settings.epochs,
        "history": history,
        "sample_size": len(sample),
        "sample_schema_valid_rate": round(len(valid) / max(len(sample), 1), 4),
        "sample_intent_accuracy": round(accuracy(gold, guessed), 4) if valid else 0.0,
        "seconds": round(time.time() - started, 1),
    }
    (config.paths.reports_dir / "resolver_train.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"resolver done in {report['seconds']}s schema valid {report['sample_schema_valid_rate']} sample accuracy {report['sample_intent_accuracy']}")
    return report
