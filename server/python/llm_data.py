from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List, Sequence

import tensorflow as tf

from llm_tokenizer import SimpleSubwordTokenizer, normalize_text


@dataclass
class TokenizedDocument:
    token_ids: List[int]


@dataclass
class TokenizedCorpus:
    documents: List[TokenizedDocument]
    token_count: int
    sample_count: int


@dataclass
class DatasetBundle:
    train_dataset: tf.data.Dataset
    validation_dataset: tf.data.Dataset
    training_sample_count: int
    validation_sample_count: int
    batches_per_epoch: int


def flatten_json_value(value, prefix: str = "") -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        lines: List[str] = []
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(flatten_json_value(nested, next_prefix))
        return lines
    if isinstance(value, list):
        lines: List[str] = []
        for index, nested in enumerate(value):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            lines.extend(flatten_json_value(nested, next_prefix))
        return lines

    text = normalize_text(str(value))
    if not text:
        return []
    return [f"{prefix}: {text}" if prefix else text]


def iter_texts_from_json(path: Path) -> Iterator[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else [payload]
    for entry in entries:
        text = "\n".join(flatten_json_value(entry))
        normalized = normalize_text(text)
        if normalized:
            yield normalized


def iter_texts_from_csv(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lines = []
            for key, value in row.items():
                key_text = normalize_text(str(key))
                value_text = normalize_text(str(value or ""))
                if value_text:
                    lines.append(f"{key_text}: {value_text}" if key_text else value_text)
            text = normalize_text("\n".join(lines))
            if text:
                yield text


def iter_texts_from_txt(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = normalize_text(line)
            if text:
                yield text


def load_texts_from_dataset_files(paths: Sequence[str]) -> List[str]:
    result: List[str] = []
    for raw in paths:
        source_path = Path(raw)
        if not source_path.exists() or not source_path.is_file():
            continue

        suffix = source_path.suffix.lower()
        try:
            if suffix == ".json":
                result.extend(iter_texts_from_json(source_path))
            elif suffix == ".csv":
                result.extend(iter_texts_from_csv(source_path))
            else:
                result.extend(iter_texts_from_txt(source_path))
        except Exception:
            continue
    return result


def clean_text_stream(texts: Iterable[str], min_chars: int = 8, max_chars: int = 12000) -> Iterator[str]:
    for text in texts:
        normalized = normalize_text(text)
        if not normalized:
            continue
        if len(normalized) < min_chars:
            continue
        yield normalized[:max_chars]


def build_windows(token_ids: Sequence[int], context_length: int, stride: int) -> Iterator[List[int]]:
    if len(token_ids) < 3:
        return iter(())
    window_size = context_length + 1
    index = 0
    while index + 1 < len(token_ids):
        window = list(token_ids[index : index + window_size])
        if len(window) < 3:
            break
        if len(window) < window_size:
            window = window + [0] * (window_size - len(window))
        yield window
        if index + window_size >= len(token_ids):
            break
        index += max(1, stride)


def prepare_tokenized_corpus(
    texts: Sequence[str],
    tokenizer: SimpleSubwordTokenizer,
    context_length: int,
    max_tokens_per_text: int = 4096,
) -> TokenizedCorpus:
    documents: List[TokenizedDocument] = []
    token_count = 0
    sample_count = 0
    stride = max(1, context_length // 2)

    for text in texts:
        token_ids = tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
            max_length=max_tokens_per_text,
            truncation=True,
        )
        if len(token_ids) < 3:
            continue

        doc = TokenizedDocument(token_ids=token_ids)
        documents.append(doc)
        token_count += len(token_ids)
        for _window in build_windows(token_ids, context_length=context_length, stride=stride):
            sample_count += 1

    return TokenizedCorpus(documents=documents, token_count=token_count, sample_count=sample_count)


def create_datasets(
    corpus: TokenizedCorpus,
    context_length: int,
    batch_size: int,
    validation_split: float = 0.1,
    shuffle_buffer: int = 4096,
) -> DatasetBundle:
    validation_split = min(0.4, max(0.0, float(validation_split)))
    validation_period = max(2, int(round(1.0 / validation_split))) if validation_split > 0 else 0
    stride = max(1, context_length // 2)

    train_count = 0
    validation_count = 0
    global_index = 0
    for document in corpus.documents:
        for _window in build_windows(document.token_ids, context_length=context_length, stride=stride):
            is_validation = validation_period > 0 and global_index % validation_period == 0
            if is_validation:
                validation_count += 1
            else:
                train_count += 1
            global_index += 1

    if train_count <= 0:
        train_count = max(1, corpus.sample_count)
        validation_count = 0
        validation_period = 0

    def sample_generator(split: str) -> Generator[tuple[list[int], list[int]], None, None]:
        sample_index = 0
        for document in corpus.documents:
            for window in build_windows(document.token_ids, context_length=context_length, stride=stride):
                is_validation = validation_period > 0 and sample_index % validation_period == 0
                sample_index += 1
                if split == "validation" and not is_validation:
                    continue
                if split == "train" and is_validation:
                    continue
                x = window[:-1]
                y = window[1:]
                yield x, y

    signature = (
        tf.TensorSpec(shape=(context_length,), dtype=tf.int32),
        tf.TensorSpec(shape=(context_length,), dtype=tf.int32),
    )
    train_dataset = tf.data.Dataset.from_generator(
        lambda: sample_generator("train"),
        output_signature=signature,
    )
    train_dataset = train_dataset.shuffle(min(max(train_count, 64), shuffle_buffer))
    train_dataset = train_dataset.batch(max(1, int(batch_size)), drop_remainder=False)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_generator(
        lambda: sample_generator("validation"),
        output_signature=signature,
    )
    validation_dataset = validation_dataset.batch(max(1, int(batch_size)), drop_remainder=False)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    batches_per_epoch = max(1, (train_count + max(1, int(batch_size)) - 1) // max(1, int(batch_size)))
    return DatasetBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        training_sample_count=train_count,
        validation_sample_count=validation_count,
        batches_per_epoch=batches_per_epoch,
    )


def collect_texts(payload: Dict[str, object]) -> List[str]:
    inline = payload.get("trainingTexts")
    inline_texts = inline if isinstance(inline, list) else []
    dataset_files = payload.get("datasetFiles")
    dataset_paths = dataset_files if isinstance(dataset_files, list) else []

    all_texts = []
    all_texts.extend(str(item) for item in inline_texts if isinstance(item, str))
    all_texts.extend(load_texts_from_dataset_files([str(path) for path in dataset_paths]))
    return list(clean_text_stream(all_texts))
