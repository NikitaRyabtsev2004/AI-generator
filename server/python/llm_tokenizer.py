from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

SPECIAL_TOKENS: List[str] = [
    "__pad__",
    "__unk__",
    "__bos__",
    "__eos__",
    "__sep__",
    "__usr__",
    "__asst__",
    "__ctx__",
]

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", ")", "]", "}", "%"}
NO_SPACE_AFTER = {"(", "[", "{"}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("\r", " ").replace("\n", " ").split()).strip()


def iter_tokens(text: str) -> Iterable[str]:
    value = normalize_text(text)
    if not value:
        return []
    return TOKEN_PATTERN.findall(value)


@dataclass
class TokenizerConfig:
    vocabulary_limit: int = 12000
    min_token_frequency: int = 2
    max_word_tokens: int = 7000
    max_subword_tokens: int = 5000


class SimpleSubwordTokenizer:
    def __init__(self, id_to_token: Sequence[str]):
        self.id_to_token = list(id_to_token)
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}
        self.vocabulary_size = len(self.id_to_token)
        self.pad_id = self.token_to_id.get("__pad__", 0)
        self.unk_id = self.token_to_id.get("__unk__", 1)
        self.bos_id = self.token_to_id.get("__bos__", 2)
        self.eos_id = self.token_to_id.get("__eos__", 3)
        self._vocab = set(self.id_to_token)

    @classmethod
    def train(cls, texts: Iterable[str], config: TokenizerConfig) -> "SimpleSubwordTokenizer":
        word_counter: Counter[str] = Counter()
        subword_counter: Counter[str] = Counter()
        char_counter: Counter[str] = Counter()

        for text in texts:
            for token in iter_tokens(text):
                if not token:
                    continue
                word_counter[token] += 1
                if len(token) == 1:
                    char_counter[token] += 1
                    continue

                for index in range(len(token)):
                    piece = token[index:]
                    marked = piece if index == 0 else f"##{piece}"
                    subword_counter[marked] += 1

                    char_piece = token[index]
                    char_marked = char_piece if index == 0 else f"##{char_piece}"
                    char_counter[char_marked] += 1

        vocab: List[str] = list(SPECIAL_TOKENS)
        reserved = len(vocab)
        max_vocab = max(config.vocabulary_limit, reserved + 64)

        def append_ranked_tokens(
            counter: Counter[str],
            *,
            limit: int,
            min_frequency: int | None,
            budget: int | None = None,
        ) -> int:
            added = 0
            for token, freq in counter.most_common():
                if len(vocab) >= limit:
                    break
                if budget is not None and added >= budget:
                    break
                if min_frequency is not None and freq < min_frequency:
                    continue
                if token in SPECIAL_TOKENS or token in vocab:
                    continue
                vocab.append(token)
                added += 1
            return added

        max_word_tokens = max(32, min(config.max_word_tokens, max_vocab - reserved))
        word_budget = min(max_word_tokens, max_vocab - len(vocab))
        append_ranked_tokens(
            word_counter,
            limit=reserved + word_budget,
            min_frequency=max(1, int(config.min_token_frequency)),
        )
        # If high-frequency words are insufficient, fill the remaining budget with rare words.
        if len(vocab) < reserved + word_budget:
            append_ranked_tokens(
                word_counter,
                limit=reserved + word_budget,
                min_frequency=1,
            )

        remaining = max_vocab - len(vocab)
        if remaining > 0:
            max_subword_tokens = max(64, min(config.max_subword_tokens, remaining))
            added_subwords = append_ranked_tokens(
                subword_counter,
                limit=max_vocab,
                min_frequency=max(1, int(config.min_token_frequency)),
                budget=max_subword_tokens,
            )
            if added_subwords < max_subword_tokens and len(vocab) < max_vocab:
                append_ranked_tokens(
                    subword_counter,
                    limit=max_vocab,
                    min_frequency=1,
                    budget=max_subword_tokens - added_subwords,
                )

        for piece, _freq in char_counter.most_common():
            if len(vocab) >= max_vocab:
                break
            if piece in SPECIAL_TOKENS or piece in vocab:
                continue
            vocab.append(piece)

        if "__unk__" not in vocab:
            vocab.insert(0, "__unk__")
        return cls(vocab[:max_vocab])

    def _encode_word(self, word: str) -> List[int]:
        if not word:
            return []

        if word in self._vocab:
            return [self.token_to_id[word]]

        ids: List[int] = []
        index = 0
        length = len(word)
        while index < length:
            match = None
            end = length
            while end > index:
                piece = word[index:end]
                candidate = piece if index == 0 else f"##{piece}"
                if candidate in self._vocab:
                    match = candidate
                    break
                end -= 1

            if match is None:
                char_piece = word[index]
                char_candidate = char_piece if index == 0 else f"##{char_piece}"
                if char_candidate in self._vocab:
                    match = char_candidate
                    end = index + 1
                else:
                    ids.append(self.unk_id)
                    index += 1
                    continue

            ids.append(self.token_to_id.get(match, self.unk_id))
            index = end

        return ids

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
        max_length: int | None = None,
        pad_to_length: int | None = None,
        truncation: bool = True,
    ) -> List[int]:
        token_ids: List[int] = []
        if add_bos:
            token_ids.append(self.bos_id)

        for token in iter_tokens(text):
            token_ids.extend(self._encode_word(token))

        if add_eos:
            token_ids.append(self.eos_id)

        if max_length is not None and max_length > 0 and len(token_ids) > max_length:
            token_ids = token_ids[:max_length] if truncation else token_ids

        if pad_to_length is not None and pad_to_length > 0 and len(token_ids) < pad_to_length:
            token_ids = token_ids + [self.pad_id] * (pad_to_length - len(token_ids))

        return token_ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        words: List[str] = []
        current = ""

        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.id_to_token):
                token = "__unk__"
            else:
                token = self.id_to_token[token_id]

            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue

            if token.startswith("##"):
                current += token[2:]
                continue

            if current:
                words.append(current)
            current = token

        if current:
            words.append(current)

        text = ""
        for token in words:
            if not text:
                text = token
                continue
            if token in NO_SPACE_BEFORE:
                text += token
            elif text[-1] in NO_SPACE_AFTER:
                text += token
            else:
                text += f" {token}"
        return text.strip()

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "kind": "simple_subword",
            "specialTokens": list(SPECIAL_TOKENS),
            "idToToken": list(self.id_to_token),
            "tokenToId": dict(self.token_to_id),
            "vocabularySize": self.vocabulary_size,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SimpleSubwordTokenizer":
        id_to_token = payload.get("idToToken")
        if not isinstance(id_to_token, list) or not id_to_token:
            raise ValueError("Tokenizer payload is missing idToToken.")
        return cls([str(token) for token in id_to_token])

    @classmethod
    def load(cls, path: str | Path) -> "SimpleSubwordTokenizer":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(payload)
