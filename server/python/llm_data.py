from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List, Sequence

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - TensorFlow is optional for dataset inspection.
    tf = None

from llm_tokenizer import SimpleSubwordTokenizer, normalize_text


DEFAULT_MAX_DATASET_RECORDS_PER_FILE = 250_000
DEFAULT_MAX_CHARS_PER_RECORD = 12_000
DEFAULT_PARQUET_BATCH_SIZE = 256
TEXT_FIELD_HINTS = (
    "text",
    "content",
    "body",
    "question",
    "answer",
    "solution",
    "prompt",
    "response",
    "instruction",
    "output",
    "title",
    "summary",
    "dialog",
    "dialogue",
    "caption",
    "description",
    "comment",
)
PAIR_KEYS = (
    ("question", "answer"),
    ("problem", "solution"),
    ("prompt", "response"),
    ("instruction", "output"),
)
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


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


def _safe_int(value, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    if parsed <= 0:
        return fallback
    return parsed


def _is_text_like_key(key: str) -> bool:
    normalized = normalize_text(str(key)).lower()
    if not normalized:
        return False
    return any(hint in normalized for hint in TEXT_FIELD_HINTS)


def _split_into_chunks(text: str, max_chars: int) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    limit = max(64, int(max_chars))
    if len(normalized) <= limit:
        return [normalized]

    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        target_end = min(start + limit, len(normalized))
        if target_end >= len(normalized):
            chunk = normalize_text(normalized[start:target_end])
            if chunk:
                chunks.append(chunk)
            break

        boundary = max(
            normalized.rfind(". ", start, target_end),
            normalized.rfind("! ", start, target_end),
            normalized.rfind("? ", start, target_end),
            normalized.rfind("; ", start, target_end),
            normalized.rfind(", ", start, target_end),
            normalized.rfind(" ", start, target_end),
        )

        if boundary <= start + (limit // 3):
            boundary = target_end
        else:
            boundary += 1

        chunk = normalize_text(normalized[start:boundary])
        if chunk:
            chunks.append(chunk)
        start = boundary

    return chunks


def _label_text(label: str, text: str, max_chars: int) -> List[str]:
    prepared_chunks = _split_into_chunks(text, max_chars)
    if not prepared_chunks:
        return []

    normalized_label = normalize_text(label)
    if not normalized_label or _is_text_like_key(normalized_label):
        return prepared_chunks

    return [f"{normalized_label}: {chunk}" for chunk in prepared_chunks]


def _dedupe_records(records: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for record in records:
        normalized = normalize_text(record)
        if len(normalized) < 8:
            continue
        signature = normalized.lower()
        if signature in seen:
            continue
        seen.add(signature)
        result.append(normalized)
    return result


def _decode_binary_value(value: object) -> str:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return ""

    payload = bytes(value)
    if not payload:
        return ""

    for encoding in ("utf-8", "utf-16", "cp1251", "latin-1"):
        try:
            decoded = payload.decode(encoding)
            normalized = normalize_text(decoded)
            if normalized:
                return normalized
        except Exception:
            continue
    return ""


def extract_text_records(value: object, max_chars_per_record: int) -> List[str]:
    max_chars = max(64, int(max_chars_per_record))
    records: List[str] = []

    def walk(node: object, prefix: str = "", depth: int = 0) -> None:
        if node is None or depth > 10:
            return

        if isinstance(node, str):
            records.extend(_label_text(prefix, node, max_chars))
            return

        if isinstance(node, (bytes, bytearray, memoryview)):
            decoded = _decode_binary_value(node)
            if decoded:
                records.extend(_label_text(prefix, decoded, max_chars))
            return

        if isinstance(node, (int, float, bool)):
            records.extend(_label_text(prefix, str(node), max_chars))
            return

        if isinstance(node, list):
            for item in node:
                walk(item, prefix, depth + 1)
            return

        if isinstance(node, dict):
            normalized_keys = {
                normalize_text(str(key)).lower(): key
                for key in node.keys()
            }

            for left_key, right_key in PAIR_KEYS:
                if left_key in normalized_keys and right_key in normalized_keys:
                    left_value = normalize_text(str(node.get(normalized_keys[left_key]) or ""))
                    right_value = normalize_text(str(node.get(normalized_keys[right_key]) or ""))
                    if left_value and right_value:
                        records.extend(
                            _split_into_chunks(
                                f"Вопрос: {left_value} Ответ: {right_value}",
                                max_chars,
                            )
                        )

            metadata_lines: List[str] = []
            for raw_key, raw_value in node.items():
                key = normalize_text(str(raw_key))
                if not key:
                    continue

                if isinstance(raw_value, (str, int, float, bool)):
                    value_text = normalize_text(str(raw_value))
                    if not value_text:
                        continue
                    if _is_text_like_key(key) or len(value_text) >= max(64, max_chars // 4):
                        records.extend(_label_text(key, value_text, max_chars))
                    else:
                        metadata_lines.append(f"{key}: {value_text}")
                    continue

                walk(raw_value, key, depth + 1)

            if metadata_lines:
                records.extend(_split_into_chunks(" | ".join(metadata_lines), max_chars))
            return

        records.extend(_label_text(prefix, str(node), max_chars))

    walk(value)
    return _dedupe_records(records)


def resolve_dataset_options(options: Dict[str, object] | None = None) -> Dict[str, object]:
    payload = options if isinstance(options, dict) else {}
    parquet_columns = payload.get("parquetColumns")
    return {
        "maxRecordsPerFile": _safe_int(payload.get("maxRecordsPerFile"), DEFAULT_MAX_DATASET_RECORDS_PER_FILE),
        "maxCharsPerRecord": _safe_int(payload.get("maxCharsPerRecord"), DEFAULT_MAX_CHARS_PER_RECORD),
        "parquetBatchSize": _safe_int(payload.get("parquetBatchSize"), DEFAULT_PARQUET_BATCH_SIZE),
        "parquetColumns": [str(value) for value in parquet_columns] if isinstance(parquet_columns, list) else None,
    }


def iter_texts_from_json(path: Path, options: Dict[str, object]) -> Iterator[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else [payload]
    max_records = int(options["maxRecordsPerFile"])
    max_chars = int(options["maxCharsPerRecord"])
    yielded = 0

    for entry in entries:
        for text in extract_text_records(entry, max_chars):
            if yielded >= max_records:
                return
            yielded += 1
            yield text


def iter_texts_from_jsonl(path: Path, options: Dict[str, object]) -> Iterator[str]:
    max_records = int(options["maxRecordsPerFile"])
    max_chars = int(options["maxCharsPerRecord"])
    yielded = 0

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            if yielded >= max_records:
                return
            line = normalize_text(raw_line)
            if not line:
                continue

            try:
                payload = json.loads(line)
                records = extract_text_records(payload, max_chars)
            except Exception:
                records = _split_into_chunks(line, max_chars)

            for text in records:
                if yielded >= max_records:
                    return
                yielded += 1
                yield text


def iter_texts_from_csv(path: Path, options: Dict[str, object]) -> Iterator[str]:
    max_records = int(options["maxRecordsPerFile"])
    max_chars = int(options["maxCharsPerRecord"])
    yielded = 0

    with path.open("r", encoding="utf-8", newline="", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for text in extract_text_records(row, max_chars):
                if yielded >= max_records:
                    return
                yielded += 1
                yield text


def iter_texts_from_parquet(path: Path, options: Dict[str, object]) -> Iterator[str]:
    max_records = int(options["maxRecordsPerFile"])
    max_chars = int(options["maxCharsPerRecord"])
    batch_size = int(options["parquetBatchSize"])
    columns = options.get("parquetColumns")
    yielded = 0

    try:
        import pyarrow.parquet as pq  # pylint: disable=import-outside-toplevel
    except Exception as error:
        raise RuntimeError(
            "Parquet reader is unavailable: module 'pyarrow' is not installed."
        ) from error

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            for text in extract_text_records(row, max_chars):
                if yielded >= max_records:
                    return
                yielded += 1
                yield text


def iter_texts_from_txt(path: Path, options: Dict[str, object]) -> Iterator[str]:
    max_records = int(options["maxRecordsPerFile"])
    max_chars = int(options["maxCharsPerRecord"])
    yielded = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            for text in _split_into_chunks(line, max_chars):
                if yielded >= max_records:
                    return
                yielded += 1
                yield text


def iter_texts_from_dataset_files(paths: Sequence[str], options: Dict[str, object] | None = None) -> Iterator[str]:
    resolved_options = resolve_dataset_options(options)
    for raw in paths:
        source_path = Path(raw)
        if not source_path.exists() or not source_path.is_file():
            continue

        suffix = source_path.suffix.lower()
        try:
            if suffix == ".json":
                yield from iter_texts_from_json(source_path, resolved_options)
            elif suffix == ".jsonl" or suffix == ".ndjson":
                yield from iter_texts_from_jsonl(source_path, resolved_options)
            elif suffix == ".csv":
                yield from iter_texts_from_csv(source_path, resolved_options)
            elif suffix == ".parquet":
                yield from iter_texts_from_parquet(source_path, resolved_options)
            else:
                yield from iter_texts_from_txt(source_path, resolved_options)
        except Exception:
            if suffix == ".parquet":
                raise
            continue


def load_texts_from_dataset_files(paths: Sequence[str], options: Dict[str, object] | None = None) -> List[str]:
    return list(iter_texts_from_dataset_files(paths, options))


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
    texts: Iterable[str],
    tokenizer: SimpleSubwordTokenizer,
    context_length: int,
    max_tokens_per_text: int = 4096,
    stride: int | None = None,
) -> TokenizedCorpus:
    documents: List[TokenizedDocument] = []
    token_count = 0
    sample_count = 0
    effective_stride = max(1, int(stride) if stride else context_length)

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
        for _window in build_windows(token_ids, context_length=context_length, stride=effective_stride):
            sample_count += 1

    return TokenizedCorpus(documents=documents, token_count=token_count, sample_count=sample_count)


def create_datasets(
    corpus: TokenizedCorpus,
    context_length: int,
    batch_size: int,
    validation_split: float = 0.1,
    shuffle_buffer: int = 4096,
    stride: int | None = None,
) -> DatasetBundle:
    if tf is None:
        raise RuntimeError("TensorFlow is required to build tf.data datasets.")
    import numpy as np  # pylint: disable=import-outside-toplevel

    validation_split = min(0.4, max(0.0, float(validation_split)))
    validation_period = max(2, int(round(1.0 / validation_split))) if validation_split > 0 else 0
    effective_stride = max(1, int(stride) if stride else context_length)

    train_count = 0
    validation_count = 0
    global_index = 0
    train_windows: List[List[int]] = []
    validation_windows: List[List[int]] = []

    for document in corpus.documents:
        for window in build_windows(document.token_ids, context_length=context_length, stride=effective_stride):
            is_validation = validation_period > 0 and global_index % validation_period == 0
            if is_validation:
                validation_count += 1
                validation_windows.append(window)
            else:
                train_count += 1
                train_windows.append(window)
            global_index += 1

    if train_count <= 0:
        if validation_windows:
            train_windows = validation_windows
            validation_windows = []
            train_count = len(train_windows)
            validation_count = 0
        else:
            train_count = max(1, corpus.sample_count)
            fallback_window = [0] * (context_length + 1)
            train_windows = [fallback_window]
        validation_count = 0
        validation_period = 0

    train_array = np.asarray(train_windows, dtype=np.int32)
    validation_array = np.asarray(validation_windows, dtype=np.int32) if validation_windows else np.zeros(
        (0, context_length + 1),
        dtype=np.int32,
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_array[:, :-1], train_array[:, 1:])
    )
    train_dataset = train_dataset.shuffle(min(max(train_count, 64), shuffle_buffer))
    train_dataset = train_dataset.batch(max(1, int(batch_size)), drop_remainder=False)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (validation_array[:, :-1], validation_array[:, 1:])
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


def iter_collected_texts(payload: Dict[str, object]) -> Iterator[str]:
    inline = payload.get("trainingTexts")
    inline_texts = inline if isinstance(inline, list) else []
    dataset_files = payload.get("datasetFiles")
    dataset_paths = dataset_files if isinstance(dataset_files, list) else []
    dataset_options = payload.get("datasetOptions")
    options = dataset_options if isinstance(dataset_options, dict) else {}

    for item in inline_texts:
        if isinstance(item, str):
            yield item

    yield from iter_texts_from_dataset_files([str(path) for path in dataset_paths], options)


def collect_texts(payload: Dict[str, object]) -> List[str]:
    return list(clean_text_stream(iter_collected_texts(payload)))


def approximate_token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(normalize_text(text)))
