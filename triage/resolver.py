import math
import os

import torch

from .prompts import build_messages, render_prompt
from .schema import SchemaError, parse

INTENT_MARKER = '"intent"'


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def resolve_dtype(name, device):
    if name != "auto":
        return getattr(torch, name)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(config, adapter_dir=None, trainable=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = config.resolver
    device = resolve_device(settings.device)
    dtype = resolve_dtype(settings.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(settings.base_model, torch_dtype=dtype)
    if adapter_dir is not None and os.path.isdir(adapter_dir):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)
        if not trainable:
            model = model.merge_and_unload()
    model.to(device)
    if not trainable:
        model.eval()
    return model, tokenizer, device


def _span_confidence(pieces, logprobs):
    text = "".join(pieces)
    marker = text.find(INTENT_MARKER)
    if marker < 0:
        return None
    colon = text.find(":", marker + len(INTENT_MARKER))
    if colon < 0:
        return None
    first_quote = text.find('"', colon)
    if first_quote < 0:
        return None
    second_quote = text.find('"', first_quote + 1)
    if second_quote < 0:
        return None
    start, end = first_quote + 1, second_quote
    selected = []
    cursor = 0
    for piece, logprob in zip(pieces, logprobs):
        piece_start, piece_end = cursor, cursor + len(piece)
        cursor = piece_end
        if piece_end > start and piece_start < end:
            selected.append(logprob)
    if not selected:
        return None
    return math.exp(sum(selected) / len(selected))


class ResolverModel:
    def __init__(self, model, tokenizer, spec, config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self.config = config
        self.device = device

    @classmethod
    def load(cls, config, spec, adapter_dir=None):
        model, tokenizer, device = load_model(config, adapter_dir)
        return cls(model, tokenizer, spec, config, device)

    def _prompts(self, texts, style, examples):
        return [render_prompt(self.tokenizer, build_messages(text, self.spec, style, examples)) for text in texts]

    def generate(self, texts, style="compact", examples=None, max_new_tokens=None):
        settings = self.config.resolver
        limit = max_new_tokens or settings.max_new_tokens
        prompts = self._prompts(texts, style, examples)
        outputs = []
        for start in range(0, len(prompts), settings.infer_batch_size):
            chunk = prompts[start:start + settings.infer_batch_size]
            encoded = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=settings.max_len)
            batch = {"input_ids": encoded["input_ids"].to(self.device), "attention_mask": encoded["attention_mask"].to(self.device)}
            with torch.no_grad():
                generated = self.model.generate(
                    **batch,
                    max_new_tokens=limit,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            prompt_length = batch["input_ids"].shape[1]
            sequences = generated.sequences[:, prompt_length:]
            scores = torch.stack(generated.scores, dim=1).float().log_softmax(dim=-1)
            chosen = scores.gather(2, sequences.unsqueeze(-1)).squeeze(-1)
            for row in range(sequences.shape[0]):
                token_ids = sequences[row].tolist()
                keep = []
                for position, token_id in enumerate(token_ids):
                    if token_id == self.tokenizer.pad_token_id and position > 0:
                        break
                    keep.append(position)
                    if token_id == self.tokenizer.eos_token_id:
                        break
                pieces = [self.tokenizer.decode([token_ids[position]], skip_special_tokens=True) for position in keep]
                logprobs = [float(chosen[row, position]) for position in keep]
                text = self.tokenizer.decode([token_ids[position] for position in keep], skip_special_tokens=True)
                mean_logprob = sum(logprobs) / max(len(logprobs), 1)
                outputs.append({
                    "text": text,
                    "sequence_confidence": math.exp(mean_logprob),
                    "intent_confidence": _span_confidence(pieces, logprobs),
                    "generated_tokens": len(keep),
                    "prompt_tokens": int(batch["attention_mask"][row].sum()),
                })
        return outputs

    def predict(self, texts, style="compact", examples=None):
        settings = self.config.resolver
        raw_outputs = self.generate(texts, style, examples)
        results = []
        retry_index = []
        for position, output in enumerate(raw_outputs):
            try:
                payload = parse(output["text"], self.spec)
                results.append(self._result(payload, output, repaired=False))
            except SchemaError as error:
                results.append({"valid": False, "error": str(error), "raw": output["text"], "confidence": 0.0,
                                "prompt_tokens": output["prompt_tokens"], "generated_tokens": output["generated_tokens"]})
                retry_index.append(position)
        for _ in range(settings.repair_attempts):
            if not retry_index:
                break
            repaired = self.generate([texts[position] for position in retry_index], "full", examples)
            still_failing = []
            for position, output in zip(retry_index, repaired):
                try:
                    payload = parse(output["text"], self.spec)
                    results[position] = self._result(payload, output, repaired=True)
                except SchemaError as error:
                    results[position] = {"valid": False, "error": str(error), "raw": output["text"], "confidence": 0.0,
                                         "prompt_tokens": output["prompt_tokens"], "generated_tokens": output["generated_tokens"]}
                    still_failing.append(position)
            retry_index = still_failing
        return results

    @staticmethod
    def _result(payload, output, repaired):
        confidence = output["intent_confidence"]
        if confidence is None:
            confidence = output["sequence_confidence"]
        return {
            "valid": True,
            "error": "",
            "raw": output["text"],
            "intent": payload["intent"],
            "entities": payload["entities"],
            "action": payload["action"],
            "reply": payload["reply"],
            "confidence": float(confidence),
            "sequence_confidence": float(output["sequence_confidence"]),
            "repaired": repaired,
            "prompt_tokens": output["prompt_tokens"],
            "generated_tokens": output["generated_tokens"],
        }
