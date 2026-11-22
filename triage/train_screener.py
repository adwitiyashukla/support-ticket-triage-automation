import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config, load_spec
from .data import load_split
from .metrics import accuracy, expected_calibration_error, macro_f1, reliability_table
from .model import ScreenerModel
from .tokenizer import ByteBPE, compression_stats, train_from_corpus


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_threads(requested):
    if requested:
        return requested
    return max(1, os.cpu_count() or 1)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_tokenizer(config, rebuild=False):
    path = config.paths.artifacts_dir / "tokenizer.json"
    if path.exists() and not rebuild:
        return ByteBPE.load(path)
    config.paths.ensure()
    tokenizer = train_from_corpus(
        config.paths.processed_dir / "corpus.txt",
        config.tokenizer.vocab_size,
        config.tokenizer.min_pair_freq,
    )
    tokenizer.save(path)
    return tokenizer


def pack_language_model_tokens(texts, tokenizer, block):
    stream = []
    for text in texts:
        stream.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    usable = (len(stream) - 1) // block * block
    if usable <= 0:
        raise ValueError("corpus too small to pack even one block")
    inputs = torch.tensor(stream[:usable], dtype=torch.long).view(-1, block)
    targets = torch.tensor(stream[1:usable + 1], dtype=torch.long).view(-1, block)
    return inputs, targets


def encode_classification(rows, tokenizer, label_to_index, max_len):
    sequences = [tokenizer.encode(row["text"], add_bos=True, add_eos=True)[:max_len] for row in rows]
    labels = torch.tensor([label_to_index[row["intent"]] for row in rows], dtype=torch.long)
    return sequences, labels


def collate(sequences, indices, pad_id):
    selected = [sequences[int(index)] for index in indices]
    width = max(len(item) for item in selected)
    ids = torch.full((len(selected), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(selected), width), dtype=torch.long)
    for position, item in enumerate(selected):
        ids[position, :len(item)] = torch.tensor(item, dtype=torch.long)
        mask[position, :len(item)] = 1
    return ids, mask


def cosine_schedule(optimizer, step, total_steps, warmup_ratio, base_lr):
    warmup = max(1, int(total_steps * warmup_ratio))
    if step < warmup:
        factor = step / warmup
    else:
        progress = (step - warmup) / max(1, total_steps - warmup)
        factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * factor
    return base_lr * factor


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() < 2 or name.endswith("norm.weight") or "norm." in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
    )


def pretrain(model, inputs, targets, settings, device, pad_id, log):
    if settings.pretrain_epochs == 0:
        return []
    model.train()
    optimizer = build_optimizer(model, settings.pretrain_lr, settings.weight_decay)
    order = torch.arange(inputs.shape[0])
    steps_per_epoch = math.ceil(inputs.shape[0] / settings.batch_size)
    total_steps = steps_per_epoch * settings.pretrain_epochs
    history = []
    step = 0
    for epoch in range(settings.pretrain_epochs):
        order = order[torch.randperm(order.shape[0])]
        running, seen = 0.0, 0
        for start in range(0, order.shape[0], settings.batch_size):
            batch = order[start:start + settings.batch_size]
            x = inputs[batch].to(device)
            y = targets[batch].to(device)
            logits = model.forward_lm(x)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1), ignore_index=pad_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            cosine_schedule(optimizer, step, total_steps, settings.warmup_ratio, settings.pretrain_lr)
            optimizer.step()
            running += loss.item() * x.shape[0]
            seen += x.shape[0]
            step += 1
        entry = {"epoch": epoch + 1, "loss": round(running / seen, 4), "perplexity": round(math.exp(min(20.0, running / seen)), 3)}
        history.append(entry)
        log(f"pretrain epoch {entry['epoch']}/{settings.pretrain_epochs} loss {entry['loss']} ppl {entry['perplexity']}")
    return history


def evaluate_logits(model, sequences, device, batch_size, pad_id):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            indices = range(start, min(start + batch_size, len(sequences)))
            ids, mask = collate(sequences, indices, pad_id)
            chunks.append(model.forward_cls(ids.to(device), mask.to(device)).cpu())
    return torch.cat(chunks, dim=0)


def finetune(model, train_data, val_data, settings, device, pad_id, log):
    train_sequences, train_labels = train_data
    val_sequences, val_labels = val_data
    optimizer = build_optimizer(model, settings.finetune_lr, settings.weight_decay)
    steps_per_epoch = math.ceil(len(train_sequences) / settings.batch_size)
    total_steps = steps_per_epoch * settings.finetune_epochs
    best_state = None
    best_accuracy = -1.0
    history = []
    step = 0
    for epoch in range(settings.finetune_epochs):
        model.train()
        order = torch.randperm(len(train_sequences))
        running, seen = 0.0, 0
        for start in range(0, order.shape[0], settings.batch_size):
            batch = order[start:start + settings.batch_size]
            ids, mask = collate(train_sequences, batch.tolist(), pad_id)
            logits = model.forward_cls(ids.to(device), mask.to(device))
            loss = F.cross_entropy(logits, train_labels[batch].to(device), label_smoothing=settings.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            cosine_schedule(optimizer, step, total_steps, settings.warmup_ratio, settings.finetune_lr)
            optimizer.step()
            running += loss.item() * batch.shape[0]
            seen += batch.shape[0]
            step += 1
        logits = evaluate_logits(model, val_sequences, device, settings.batch_size, pad_id)
        predictions = logits.argmax(dim=1).numpy()
        current = accuracy(val_labels.numpy(), predictions)
        entry = {"epoch": epoch + 1, "train_loss": round(running / seen, 4), "val_accuracy": round(current, 4)}
        history.append(entry)
        log(f"finetune epoch {entry['epoch']}/{settings.finetune_epochs} loss {entry['train_loss']} val accuracy {entry['val_accuracy']}")
        if current > best_accuracy:
            best_accuracy = current
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_accuracy


def fit_temperature(logits, labels, steps=200):
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=steps)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().item())


def train(config=None, spec=None, log=print):
    config = config or load_config()
    spec = spec or load_spec()
    config.paths.ensure()
    settings = config.screener
    device = resolve_device(settings.device)
    threads = resolve_threads(settings.threads)
    torch.set_num_threads(threads)
    set_seed(config.data.seed)

    train_rows = load_split(config, "train")
    val_rows = load_split(config, "val")
    tokenizer = get_tokenizer(config)
    labels = spec.labels()
    label_to_index = {label: index for index, label in enumerate(labels)}

    started = time.time()
    train_data = encode_classification(train_rows, tokenizer, label_to_index, settings.max_seq_len)
    val_data = encode_classification(val_rows, tokenizer, label_to_index, settings.max_seq_len)

    model = ScreenerModel(
        vocab_size=tokenizer.vocab_size,
        n_layer=settings.n_layer,
        n_head=settings.n_head,
        d_model=settings.d_model,
        d_ff=settings.d_ff,
        max_seq_len=settings.max_seq_len,
        dropout=settings.dropout,
        rope_theta=settings.rope_theta,
        n_labels=len(labels),
        pad_id=tokenizer.pad_id,
    ).to(device)
    log(f"screener parameters {model.parameter_count():,} vocab {tokenizer.vocab_size} device {device} threads {threads}")

    pretrain_history = []
    if not settings.skip_pretrain:
        corpus_texts = [row["text"] for row in train_rows]
        inputs, targets = pack_language_model_tokens(corpus_texts, tokenizer, settings.max_seq_len)
        log(f"language model blocks {inputs.shape[0]} of {settings.max_seq_len} tokens")
        pretrain_history = pretrain(model, inputs, targets, settings, device, tokenizer.pad_id, log)

    finetune_history, best_accuracy = finetune(model, train_data, val_data, settings, device, tokenizer.pad_id, log)

    val_logits = evaluate_logits(model, val_data[0], device, settings.batch_size, tokenizer.pad_id)
    temperature = fit_temperature(val_logits, val_data[1])
    raw_confidence = val_logits.softmax(dim=1).max(dim=1).values.numpy()
    calibrated_confidence = (val_logits / temperature).softmax(dim=1).max(dim=1).values.numpy()
    predictions = val_logits.argmax(dim=1).numpy()
    correct = (predictions == val_data[1].numpy()).astype(float)

    artifact = {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "labels": labels,
        "temperature": temperature,
        "pad_id": tokenizer.pad_id,
        "model_args": {
            "vocab_size": tokenizer.vocab_size,
            "n_layer": settings.n_layer,
            "n_head": settings.n_head,
            "d_model": settings.d_model,
            "d_ff": settings.d_ff,
            "max_seq_len": settings.max_seq_len,
            "dropout": 0.0,
            "rope_theta": settings.rope_theta,
            "n_labels": len(labels),
        },
    }
    torch.save(artifact, config.paths.artifacts_dir / "screener.pt")

    report = {
        "parameters": model.parameter_count(),
        "vocab_size": tokenizer.vocab_size,
        "device": str(device),
        "skip_pretrain": settings.skip_pretrain,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "pretrain": pretrain_history,
        "finetune": finetune_history,
        "best_val_accuracy": round(best_accuracy, 4),
        "val_macro_f1": round(macro_f1(val_data[1].numpy(), predictions, range(len(labels))), 4),
        "temperature": round(temperature, 4),
        "ece_raw": round(expected_calibration_error(raw_confidence, correct, config.evaluation.calibration_bins), 4),
        "ece_calibrated": round(expected_calibration_error(calibrated_confidence, correct, config.evaluation.calibration_bins), 4),
        "reliability": reliability_table(calibrated_confidence, correct, config.evaluation.calibration_bins),
        "tokenizer": compression_stats(tokenizer, [row["text"] for row in val_rows]),
        "seconds": round(time.time() - started, 1),
    }
    (config.paths.reports_dir / "screener_train.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"screener done in {report['seconds']}s val accuracy {report['best_val_accuracy']} ece {report['ece_raw']} to {report['ece_calibrated']}")
    return report
