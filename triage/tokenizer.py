import json
import re
from collections import Counter, defaultdict

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIALS = (PAD, BOS, EOS)
PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def bytes_to_unicode():
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {value: chr(value) for value in printable}
    spare = 0
    for value in range(256):
        if value not in mapping:
            mapping[value] = chr(256 + spare)
            spare += 1
    return mapping


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {symbol: value for value, symbol in BYTE_ENCODER.items()}


def _pairs(symbols):
    return zip(symbols, symbols[1:])


class ByteBPE:
    def __init__(self, merges):
        self.merges = [tuple(pair) for pair in merges]
        self.ranks = {pair: index for index, pair in enumerate(self.merges)}
        self.itos = list(SPECIALS) + [BYTE_ENCODER[value] for value in range(256)]
        for left, right in self.merges:
            self.itos.append(left + right)
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.itos)

    @classmethod
    def train(cls, texts, vocab_size, min_pair_freq=2, progress=None):
        budget = vocab_size - len(SPECIALS) - 256
        if budget <= 0:
            return cls([])
        counts = Counter()
        for text in texts:
            for chunk in PATTERN.findall(text):
                counts["".join(BYTE_ENCODER[value] for value in chunk.encode("utf-8"))] += 1

        words = [list(word) for word in counts]
        frequencies = list(counts.values())
        pair_counts = Counter()
        pair_index = defaultdict(set)
        for position, symbols in enumerate(words):
            weight = frequencies[position]
            for pair in _pairs(symbols):
                pair_counts[pair] += weight
                pair_index[pair].add(position)

        merges = []
        while len(merges) < budget:
            if not pair_counts:
                break
            best, count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            if count < min_pair_freq:
                break
            merges.append(best)
            joined = best[0] + best[1]
            for position in list(pair_index[best]):
                symbols = words[position]
                weight = frequencies[position]
                for pair in _pairs(symbols):
                    pair_counts[pair] -= weight
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_index[pair].discard(position)
                merged, index = [], 0
                while index < len(symbols):
                    if index < len(symbols) - 1 and symbols[index] == best[0] and symbols[index + 1] == best[1]:
                        merged.append(joined)
                        index += 2
                    else:
                        merged.append(symbols[index])
                        index += 1
                words[position] = merged
                for pair in _pairs(merged):
                    pair_counts[pair] += weight
                    pair_index[pair].add(position)
            pair_counts.pop(best, None)
            pair_index.pop(best, None)
            if progress and len(merges) % 500 == 0:
                progress(len(merges), budget)
        return cls(merges)

    def _apply(self, word):
        if word in self._cache:
            return self._cache[word]
        symbols = list(word)
        while len(symbols) > 1:
            candidates = [(self.ranks[pair], position) for position, pair in enumerate(_pairs(symbols)) if pair in self.ranks]
            if not candidates:
                break
            _, position = min(candidates)
            symbols[position:position + 2] = [symbols[position] + symbols[position + 1]]
        self._cache[word] = symbols
        return symbols

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.bos_id] if add_bos else []
        for chunk in PATTERN.findall(text):
            word = "".join(BYTE_ENCODER[value] for value in chunk.encode("utf-8"))
            ids.extend(self.stoi[symbol] for symbol in self._apply(word))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        symbols = "".join(self.itos[index] for index in ids if index >= len(SPECIALS))
        return bytes(BYTE_DECODER[symbol] for symbol in symbols).decode("utf-8", errors="replace")

    def save(self, path):
        payload = {"specials": list(SPECIALS), "merges": [" ".join(pair) for pair in self.merges]}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls([tuple(item.split(" ")) for item in payload["merges"]])


def train_from_corpus(corpus_path, vocab_size, min_pair_freq, progress=None):
    with open(corpus_path, encoding="utf-8") as handle:
        texts = [line.rstrip("\n") for line in handle if line.strip()]
    return ByteBPE.train(texts, vocab_size, min_pair_freq, progress)


def compression_stats(tokenizer, texts):
    tokens = sum(len(tokenizer.encode(text)) for text in texts)
    characters = sum(len(text) for text in texts)
    words = sum(len(text.split()) for text in texts)
    return {
        "documents": len(texts),
        "characters": characters,
        "tokens": tokens,
        "chars_per_token": round(characters / max(tokens, 1), 3),
        "tokens_per_word": round(tokens / max(words, 1), 3),
    }
