#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_tokenizer import SPECIAL_TOKENS, SimpleSubwordTokenizer, TokenizerConfig

EMIT_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_json_payload(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_json_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json_payload(item) for item in value]
    return value


def emit(payload: Dict[str, Any]) -> None:
    sanitized_payload = sanitize_json_payload(payload)
    with EMIT_LOCK:
        print(json.dumps(sanitized_payload, ensure_ascii=False, allow_nan=False), flush=True)


def clean_message(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def read_json(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sanitized_payload = sanitize_json_payload(payload)
    target.write_text(
        json.dumps(sanitized_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def safe_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(parsed) or math.isinf(parsed):
        return fallback
    return parsed


def safe_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed


def safe_non_negative_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    if parsed < 0:
        return fallback
    return parsed


def safe_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    return fallback


def resolve_training_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    training = settings.get("training") if isinstance(settings.get("training"), dict) else settings
    sequence_length = max(16, safe_int(training.get("sequenceLength"), 128))
    embedding_size = max(32, safe_int(training.get("embeddingSize"), 128))
    attention_heads = max(1, safe_int(training.get("attentionHeads"), 4))
    while attention_heads > 1 and embedding_size % attention_heads != 0:
        attention_heads -= 1

    stride_ratio = min(1.0, max(0.1, safe_float(training.get("windowStrideRatio"), 1.0)))
    requested_stride = safe_int(training.get("windowStride"), 0)
    if requested_stride <= 0:
        requested_stride = max(1, int(round(sequence_length * stride_ratio)))
    window_stride = min(sequence_length, max(1, requested_stride))

    transformer_layers = max(1, safe_int(training.get("transformerLayers"), 4))
    batch_size = max(1, safe_int(training.get("batchSize"), 8))
    feed_forward_requested = max(
        embedding_size,
        safe_int(training.get("feedForwardSize"), embedding_size * 4),
    )
    feed_forward_ratio_cap = max(embedding_size, embedding_size * 8)
    # Keep FFN activations under a conservative budget to reduce OOM risk.
    activation_budget_bytes = 96 * 1024 * 1024
    denominator = max(1, batch_size * sequence_length * 4)
    feed_forward_activation_cap = max(
        embedding_size,
        int((activation_budget_bytes / denominator) / max(1.0, math.sqrt(transformer_layers / 12.0))),
    )
    feed_forward_size = max(
        embedding_size,
        min(feed_forward_requested, feed_forward_ratio_cap, feed_forward_activation_cap),
    )
    optimizer_name = str(training.get("optimizer") or "adam_legacy").strip().lower()
    if optimizer_name not in ("adam", "adam_legacy"):
        optimizer_name = "adam_legacy"

    return {
        "executionMode": str(training.get("executionMode") or "native_preferred"),
        "sequenceLength": sequence_length,
        "windowStride": window_stride,
        "enableXla": safe_bool(training.get("enableXla"), True),
        "embeddingSize": embedding_size,
        "attentionHeads": attention_heads,
        "transformerLayers": transformer_layers,
        "feedForwardSize": feed_forward_size,
        "dropout": min(0.6, max(0.0, safe_float(training.get("dropout"), 0.1))),
        "learningRate": max(1e-6, safe_float(training.get("learningRate"), 0.001)),
        "optimizer": optimizer_name,
        "gradientClipNorm": min(10.0, max(0.0, safe_float(training.get("gradientClipNorm"), 0.0))),
        "batchSize": batch_size,
        "epochs": max(1, safe_int(training.get("epochs"), 1)),
        # Emit every batch by default so UI progress is real-time.
        "batchReportInterval": max(1, safe_int(training.get("batchReportInterval"), 1)),
        # Collect scalar metrics less frequently to reduce GPU↔CPU sync overhead.
        "metricsReportInterval": max(1, safe_int(training.get("metricsReportInterval"), 16)),
        # Keep chart points lighter than raw batch stream by default.
        "historyReportInterval": max(1, safe_int(training.get("historyReportInterval"), 16)),
        # Conservative tf.data prefetch lowers risk of sporadic pipeline stalls on Windows GPU.
        "datasetPrefetchBatches": min(8, max(0, safe_non_negative_int(training.get("datasetPrefetchBatches"), 1))),
        # Optional private threadpool for tf.data (0 = TensorFlow default behavior).
        "datasetPrivateThreadpoolSize": min(32, max(0, safe_non_negative_int(training.get("datasetPrivateThreadpoolSize"), 0))),
        # Background heartbeat keeps orchestration alive even if TensorFlow stalls inside one long step.
        "heartbeatIntervalSeconds": min(60, max(5, safe_int(training.get("heartbeatIntervalSeconds"), 10))),
        # Periodic recovery checkpoints reduce progress loss after crashes or forced restarts.
        "recoveryCheckpointIntervalBatches": min(
            50_000,
            max(0, safe_non_negative_int(training.get("recoveryCheckpointIntervalBatches"), 1000)),
        ),
        "recoveryCheckpointIntervalMinutes": min(
            240.0,
            max(0.0, safe_float(training.get("recoveryCheckpointIntervalMinutes"), 20.0)),
        ),
        "vocabularyLimit": max(512, safe_int(training.get("vocabularyLimit"), 12000)),
        "validationSplit": min(0.4, max(0.0, safe_float(training.get("validationSplit"), 0.1))),
    }


def resolve_generation_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    generation = settings.get("generation") if isinstance(settings.get("generation"), dict) else settings
    return {
        "responseTemperature": min(2.0, max(0.1, safe_float(generation.get("responseTemperature"), 0.7))),
        "topKSampling": max(1, safe_int(generation.get("topKSampling"), 16)),
        "repetitionPenalty": min(3.0, max(1.0, safe_float(generation.get("repetitionPenalty"), 1.15))),
        "maxGeneratedTokens": max(1, safe_int(generation.get("maxGeneratedTokens"), 96)),
    }


def get_storage_paths(storage: Dict[str, Any]) -> Dict[str, str]:
    return {
        "tokenizerPath": str(storage.get("tokenizerPath") or "").strip(),
        "weightsPath": str(storage.get("neuralWeightsPath") or "").strip(),
        "specPath": str(storage.get("neuralSpecPath") or "").strip(),
    }


def checkpoint_exists(storage: Dict[str, Any]) -> bool:
    paths = get_storage_paths(storage)
    required = [paths["tokenizerPath"], paths["weightsPath"], paths["specPath"]]
    return all(path and Path(path).exists() for path in required)


def load_tokenizer_and_spec(storage: Dict[str, Any]) -> Tuple[SimpleSubwordTokenizer, Dict[str, Any]]:
    paths = get_storage_paths(storage)
    tokenizer_payload = read_json(paths["tokenizerPath"])
    tokenizer = SimpleSubwordTokenizer.from_dict(tokenizer_payload)
    spec_payload = read_json(paths["specPath"])
    return tokenizer, spec_payload


def command_load_runtime(config: Dict[str, Any]) -> int:
    storage = config.get("storage") if isinstance(config.get("storage"), dict) else {}
    if not checkpoint_exists(storage):
        emit(
            {
                "ok": False,
                "error": "Checkpoint not found.",
                "checkpointReady": False,
            }
        )
        return 0

    tokenizer, spec_payload = load_tokenizer_and_spec(storage)
    manifest = spec_payload.get("manifest") if isinstance(spec_payload.get("manifest"), dict) else {}
    parameter_count = safe_int(manifest.get("parameterCount"), 0)
    vocabulary_size = safe_int(manifest.get("vocabularySize"), tokenizer.vocabulary_size)
    emit(
        {
            "ok": True,
            "checkpointReady": True,
            "tokenizer": tokenizer.to_dict(),
            "manifest": manifest,
            "parameterCount": parameter_count,
            "vocabularySize": vocabulary_size,
            "format": spec_payload.get("format") or "keras_llm_checkpoint_v1",
        }
    )
    return 0


def import_tensorflow():
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    return tf


def configure_tensorflow(training_settings: Dict[str, Any]) -> Dict[str, Any]:
    tf = import_tensorflow()
    execution_mode = training_settings.get("executionMode", "native_preferred")
    gpus = tf.config.list_physical_devices("GPU")
    using_gpu = bool(gpus) and execution_mode != "compatibility"
    backend_warning = ""

    tf_version = str(getattr(tf, "__version__", "0"))
    tf_major, tf_minor = 0, 0
    parts = tf_version.split(".")
    try:
        tf_major = int(parts[0])
        tf_minor = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        tf_major, tf_minor = 0, 0

    if using_gpu:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)
        except Exception:
            pass

    mixed_precision_enabled = False
    if using_gpu:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            mixed_precision_enabled = True
        except Exception:
            tf.keras.mixed_precision.set_global_policy("float32")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")

    if execution_mode == "gpu_preferred" and not using_gpu:
        if platform.system().lower() == "windows" and (tf_major > 2 or (tf_major == 2 and tf_minor >= 11)):
            backend_warning = (
                " TensorFlow не видит GPU на Windows для этой версии. "
                "Используйте setup-venv2.ps1 (TensorFlow 2.10 + CUDA 11.2 + cuDNN 8.1) или WSL2."
            )
        else:
            backend_warning = " GPU не обнаружен TensorFlow, поэтому используется CPU."

    label = "TensorFlow (GPU)" if using_gpu else "TensorFlow (CPU)"
    return {
        "tf": tf,
        "usingGpu": using_gpu,
        "mixedPrecision": mixed_precision_enabled,
        "backendName": "tensorflow",
        "backendLabel": label,
        "backendWarning": backend_warning,
    }


def load_keras_model_from_bytes(weights_path: str):
    tf = import_tensorflow()
    from llm_model import (  # pylint: disable=import-outside-toplevel
        LearnedPositionEmbedding,
        SinusoidalPositionEncoding,
        TransformerBlock,
    )

    custom_objects = {
        "LearnedPositionEmbedding": LearnedPositionEmbedding,
        "SinusoidalPositionEncoding": SinusoidalPositionEncoding,
        "TransformerBlock": TransformerBlock,
    }
    with tempfile.TemporaryDirectory(prefix="ai_generator_load_") as temp_dir:
        temp_path = Path(temp_dir) / "runtime_model.keras"
        try:
            os.link(str(weights_path), str(temp_path))
        except Exception:
            shutil.copyfile(str(weights_path), str(temp_path))
        model = tf.keras.models.load_model(str(temp_path), custom_objects=custom_objects, compile=False)
    return model


def load_keras_model_from_weights(weights_path: str, spec_payload: Dict[str, Any], tokenizer: SimpleSubwordTokenizer):
    tf = import_tensorflow()
    from llm_model import GPTConfig, build_gpt_model  # pylint: disable=import-outside-toplevel

    model_config_payload = spec_payload.get("modelConfig") if isinstance(spec_payload.get("modelConfig"), dict) else {}
    if not model_config_payload:
        raise ValueError("Checkpoint modelConfig is missing.")

    config_payload = dict(model_config_payload)
    config_payload["vocabulary_size"] = max(
        safe_int(config_payload.get("vocabulary_size"), 0),
        safe_int(spec_payload.get("manifest", {}).get("vocabularySize"), tokenizer.vocabulary_size),
        tokenizer.vocabulary_size,
    )
    config_payload["pad_token_id"] = safe_int(config_payload.get("pad_token_id"), tokenizer.pad_id)
    model_config = GPTConfig(**config_payload)
    model = build_gpt_model(model_config)
    model(tf.zeros((1, model_config.context_length), dtype=tf.int32), training=False)

    with tempfile.TemporaryDirectory(prefix="ai_generator_load_weights_") as temp_dir:
        temp_path = Path(temp_dir) / "runtime_model.weights.h5"
        try:
            os.link(str(weights_path), str(temp_path))
        except Exception:
            shutil.copyfile(str(weights_path), str(temp_path))
        model.load_weights(str(temp_path))
    return model


def save_keras_model_weights(model, weights_path: str) -> None:
    target = Path(weights_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix="ai_generator_save_weights_",
        suffix=".weights.h5",
        dir=str(target.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        model.save_weights(str(temp_path))
        os.replace(str(temp_path), str(target))
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


def save_keras_model_archive(model, weights_path: str) -> None:
    target = Path(weights_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix="ai_generator_save_archive_",
        suffix=".keras",
        dir=str(target.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        model.save(str(temp_path), include_optimizer=False)
        os.replace(str(temp_path), str(target))
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


def load_checkpoint_runtime(storage: Dict[str, Any]) -> Tuple[Any, SimpleSubwordTokenizer, Dict[str, Any]]:
    paths = get_storage_paths(storage)
    tokenizer, spec_payload = load_tokenizer_and_spec(storage)
    weights_format = str(spec_payload.get("weightsFormat") or "").strip().lower()
    if weights_format == "hdf5_weights_v1":
        model = load_keras_model_from_weights(paths["weightsPath"], spec_payload, tokenizer)
    else:
        model = load_keras_model_from_bytes(paths["weightsPath"])
    return model, tokenizer, spec_payload


def save_checkpoint_runtime(
    storage: Dict[str, Any],
    model,
    tokenizer: SimpleSubwordTokenizer,
    spec_payload: Dict[str, Any],
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> None:
    def report(stage: str, message: str) -> None:
        if callable(on_progress):
            on_progress(stage, message)

    paths = get_storage_paths(storage)
    report("tokenizer", "Сохранение токенизатора чекпоинта.")
    tokenizer.save(paths["tokenizerPath"])
    report("weights", "Сохранение весов нейросети (может занять время на больших моделях).")
    requested_weights_format = str(spec_payload.get("weightsFormat") or "hdf5_weights_v1").strip().lower()
    actual_weights_format = requested_weights_format
    if requested_weights_format == "hdf5_weights_v1":
        try:
            save_keras_model_weights(model, paths["weightsPath"])
        except Exception:
            report("weights_fallback", "Быстрое сохранение весов недоступно, используется совместимый архивный формат чекпоинта.")
            save_keras_model_archive(model, paths["weightsPath"])
            actual_weights_format = "legacy_keras_archive_v1"
    else:
        save_keras_model_archive(model, paths["weightsPath"])
        actual_weights_format = "legacy_keras_archive_v1"
    report("spec", "Сохранение спецификации чекпоинта.")
    spec_payload["weightsFormat"] = actual_weights_format
    write_json(paths["specPath"], spec_payload)
    report("done", "Чекпоинт успешно сохранен.")


def build_spec_payload(
    *,
    manifest: Dict[str, Any],
    parameter_count: int,
    completed_epochs: int,
    saved_at: str,
    tokenizer: SimpleSubwordTokenizer,
    tokenized_corpus,
    training_settings: Dict[str, Any],
    model_config,
    checkpoint_kind: str = "final",
    next_resume_epoch_offset: int = 0,
    next_resume_batch_offset: int = 0,
    processed_batches: int = 0,
) -> Dict[str, Any]:
    model_settings_manifest = {
        "sequenceLength": training_settings["sequenceLength"],
        "embeddingSize": training_settings["embeddingSize"],
        "attentionHeads": training_settings["attentionHeads"],
        "transformerLayers": training_settings["transformerLayers"],
        "feedForwardSize": training_settings["feedForwardSize"],
        "dropout": training_settings["dropout"],
        "learningRate": training_settings["learningRate"],
    }
    return {
        "version": 2,
        "format": "keras_llm_checkpoint_v1",
        "weightsFormat": "hdf5_weights_v1",
        "savedAt": saved_at,
        "modelConfig": asdict(model_config),
        "specs": [],
        "manifest": {
            **manifest,
            "parameterCount": parameter_count,
            "trainedEpochs": completed_epochs,
            "trainingSequenceCount": tokenized_corpus.sample_count,
            "corpusTokenCount": tokenized_corpus.token_count,
            "savedAt": saved_at,
            "vocabularySize": tokenizer.vocabulary_size,
            "modelSettings": model_settings_manifest,
            "checkpointKind": checkpoint_kind,
            "nextResumeEpochOffset": max(0, int(next_resume_epoch_offset)),
            "nextResumeBatchOffset": max(0, int(next_resume_batch_offset)),
            "processedBatches": max(0, int(processed_batches)),
        },
    }


def start_training_heartbeat(
    state: Dict[str, Any],
    interval_seconds: int,
    stop_event: threading.Event,
) -> threading.Thread:
    def worker() -> None:
        while not stop_event.wait(max(1, int(interval_seconds))):
            snapshot = dict(state)
            snapshot["startedAt"] = snapshot.get("startedAt") or now_iso()
            snapshot["sentAt"] = now_iso()
            emit({"type": "heartbeat", **snapshot})

    thread = threading.Thread(target=worker, name="training-heartbeat", daemon=True)
    thread.start()
    return thread


def choose_next_token(probabilities, top_k: int, repetition_penalty: float, generated: List[int], pad_id: int, unk_id: int) -> int:
    import numpy as np  # pylint: disable=import-outside-toplevel

    next_probs = probabilities.astype("float64")
    blocked = {pad_id, unk_id}
    for token_id in blocked:
        if 0 <= token_id < len(next_probs):
            next_probs[token_id] = 0.0

    if repetition_penalty > 1.0 and generated:
        for token_id in generated:
            if 0 <= token_id < len(next_probs):
                next_probs[token_id] = next_probs[token_id] / repetition_penalty

    top_k = max(1, min(int(top_k), len(next_probs)))
    top_indices = np.argpartition(next_probs, -top_k)[-top_k:]
    top_values = next_probs[top_indices]
    total = float(top_values.sum())
    if total <= 0:
        return int(top_indices[int(np.argmax(top_values))])

    normalized = top_values / total
    chosen_local = int(np.random.choice(len(top_indices), p=normalized))
    return int(top_indices[chosen_local])


def command_generate(config: Dict[str, Any]) -> int:
    tf = import_tensorflow()
    settings = config.get("settings") if isinstance(config.get("settings"), dict) else {}
    storage = config.get("storage") if isinstance(config.get("storage"), dict) else {}
    prompt_text = str(config.get("promptText") or "").strip()

    if not checkpoint_exists(storage):
        emit({"ok": False, "error": "Checkpoint is missing.", "text": "", "generatedTokenIds": []})
        return 0

    model, tokenizer, spec_payload = load_checkpoint_runtime(storage)
    manifest = spec_payload.get("manifest") if isinstance(spec_payload.get("manifest"), dict) else {}
    model_settings = manifest.get("modelSettings") if isinstance(manifest.get("modelSettings"), dict) else {}
    runtime_training = resolve_training_settings({"training": {**model_settings, **(settings.get("training") or {})}})
    runtime_generation = resolve_generation_settings(settings)

    context_length = runtime_training["sequenceLength"]
    max_generated_tokens = runtime_generation["maxGeneratedTokens"]
    temperature = runtime_generation["responseTemperature"]
    top_k = runtime_generation["topKSampling"]
    repetition_penalty = runtime_generation["repetitionPenalty"]

    prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False, max_length=context_length, truncation=True)
    generated_ids: List[int] = []
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id
    unk_id = tokenizer.unk_id

    for _step in range(max_generated_tokens):
        context_ids = (prompt_ids + generated_ids)[-context_length:]
        if len(context_ids) < context_length:
            context_ids = [pad_id] * (context_length - len(context_ids)) + context_ids

        input_tensor = tf.constant([context_ids], dtype=tf.int32)
        logits = model(input_tensor, training=False)
        next_logits = logits[:, -1, :] / max(0.1, temperature)
        probabilities = tf.nn.softmax(next_logits).numpy()[0]
        next_token_id = choose_next_token(
            probabilities=probabilities,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            generated=generated_ids,
            pad_id=pad_id,
            unk_id=unk_id,
        )

        if next_token_id == eos_id:
            break
        generated_ids.append(int(next_token_id))

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    emit(
        {
            "ok": True,
            "text": text,
            "generatedTokenIds": generated_ids,
            "manifest": manifest,
        }
    )
    return 0


def inspect_parquet_metadata(path: Path) -> Dict[str, Any]:
    try:
        import pyarrow.parquet as pq  # pylint: disable=import-outside-toplevel
    except Exception:
        return {}

    try:
        parquet_file = pq.ParquetFile(path)
        names = [str(name) for name in getattr(parquet_file.schema, "names", [])]
        row_count = safe_int(getattr(parquet_file.metadata, "num_rows", 0), 0)
        return {
            "rowCount": max(row_count, 0),
            "columnCount": len(names),
            "columns": names[:128],
        }
    except Exception:
        return {}


def command_inspect_dataset(config: Dict[str, Any]) -> int:
    from llm_data import iter_texts_from_dataset_files, resolve_dataset_options, approximate_token_count  # pylint: disable=import-outside-toplevel

    raw_dataset_files = config.get("datasetFiles")
    dataset_files = [str(item) for item in raw_dataset_files] if isinstance(raw_dataset_files, list) else []
    dataset_options = resolve_dataset_options(
        config.get("datasetOptions") if isinstance(config.get("datasetOptions"), dict) else {}
    )
    inspect_options = config.get("inspect") if isinstance(config.get("inspect"), dict) else {}
    preview_records = max(1, safe_int(inspect_options.get("previewRecords"), 256))
    estimate_sample_records = max(preview_records, safe_int(inspect_options.get("estimateSampleRecords"), 2048))
    preview_chars = max(64, safe_int(inspect_options.get("previewChars"), 16000))

    files_payload: List[Dict[str, Any]] = []
    for raw_path in dataset_files:
        source_path = Path(str(raw_path)).expanduser()
        file_payload: Dict[str, Any] = {
            "path": str(source_path),
            "name": source_path.name,
            "format": source_path.suffix.lower().lstrip(".") or "txt",
            "exists": source_path.exists() and source_path.is_file(),
            "sizeBytes": safe_int(source_path.stat().st_size, 0) if source_path.exists() else 0,
            "previewRecords": [],
            "sampleRecordCount": 0,
            "estimatedTokenCount": 0,
            "estimatedCharCount": 0,
        }

        if not file_payload["exists"]:
            file_payload["error"] = "not_found"
            files_payload.append(file_payload)
            continue

        if source_path.suffix.lower() == ".parquet":
            file_payload.update(inspect_parquet_metadata(source_path))

        sample_record_count = 0
        sample_char_count = 0
        sample_token_count = 0
        preview_items: List[str] = []
        inspection_error = ""

        try:
            for text in iter_texts_from_dataset_files([str(source_path)], dataset_options):
                normalized = clean_message(text)
                if not normalized:
                    continue

                if len(preview_items) < preview_records:
                    preview_items.append(normalized[:preview_chars])

                if sample_record_count < estimate_sample_records:
                    sample_record_count += 1
                    sample_char_count += len(normalized)
                    sample_token_count += approximate_token_count(normalized)

                if len(preview_items) >= preview_records and sample_record_count >= estimate_sample_records:
                    break
        except Exception as error:
            inspection_error = clean_message(str(error))

        row_count = safe_int(file_payload.get("rowCount"), 0)
        if row_count > sample_record_count > 0:
            scale = row_count / sample_record_count
            estimated_char_count = int(sample_char_count * scale)
            estimated_token_count = int(sample_token_count * scale)
        else:
            estimated_char_count = sample_char_count
            estimated_token_count = sample_token_count

        file_payload.update(
            {
                "previewRecords": preview_items,
                "sampleRecordCount": sample_record_count,
                "estimatedRecordCount": row_count if row_count > 0 else sample_record_count,
                "estimatedTokenCount": max(0, estimated_token_count),
                "estimatedCharCount": max(0, estimated_char_count),
            }
        )
        if inspection_error:
            file_payload["error"] = inspection_error
        files_payload.append(file_payload)

    emit(
        {
            "ok": True,
            "files": files_payload,
        }
    )
    return 0


def stop_requested(stop_signal_path: str) -> bool:
    return bool(stop_signal_path and Path(stop_signal_path).exists())


def read_training_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    payload_texts: List[str] = []
    dataset_paths: List[str] = []
    dataset_options: Dict[str, Any] = {}

    training_payload_path = str(config.get("trainingPayloadPath") or "").strip()
    if training_payload_path and Path(training_payload_path).exists():
        training_payload = read_json(training_payload_path)
        values = training_payload.get("trainingTexts")
        if isinstance(values, list):
            payload_texts.extend(str(item) for item in values if isinstance(item, str))
        payload_dataset_files = training_payload.get("datasetFiles")
        if isinstance(payload_dataset_files, list):
            dataset_paths.extend(str(value) for value in payload_dataset_files)
        payload_dataset_options = training_payload.get("datasetOptions")
        if isinstance(payload_dataset_options, dict):
            dataset_options = {**dataset_options, **payload_dataset_options}

    inline_values = config.get("trainingTexts")
    if isinstance(inline_values, list):
        payload_texts.extend(str(item) for item in inline_values if isinstance(item, str))

    dataset_files = config.get("datasetFiles")
    if isinstance(dataset_files, list):
        dataset_paths.extend(str(value) for value in dataset_files)
    config_dataset_options = config.get("datasetOptions")
    if isinstance(config_dataset_options, dict):
        dataset_options = {**dataset_options, **config_dataset_options}

    return {
        "trainingTexts": payload_texts,
        "datasetFiles": dataset_paths,
        "datasetOptions": dataset_options,
    }


def create_training_text_iterator(payload: Dict[str, Any]):
    from llm_data import clean_text_stream, iter_collected_texts  # pylint: disable=import-outside-toplevel

    return clean_text_stream(iter_collected_texts(payload))


def compute_masked_loss(tf, labels, logits, pad_id: int):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32), logits=logits)
    mask = tf.cast(tf.not_equal(labels, pad_id), losses.dtype)
    masked = losses * mask
    denominator = tf.maximum(tf.reduce_sum(mask), tf.constant(1.0, dtype=losses.dtype))
    return tf.reduce_sum(masked) / denominator


def build_train_step(
    tf,
    model,
    optimizer,
    clip_norm: float,
    pad_id: int,
    use_loss_scale: bool,
    use_xla: bool,
    return_loss: bool = True,
):
    @tf.function(reduce_retracing=True, jit_compile=use_xla)
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = compute_masked_loss(tf, y_batch, logits, pad_id=pad_id)
            scaled_loss = optimizer.get_scaled_loss(loss) if use_loss_scale and hasattr(optimizer, "get_scaled_loss") else loss

        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        if use_loss_scale and hasattr(optimizer, "get_unscaled_gradients"):
            gradients = optimizer.get_unscaled_gradients(gradients)

        grads_and_vars = [(gradient, variable) for gradient, variable in zip(gradients, model.trainable_variables) if gradient is not None]
        if grads_and_vars:
            if clip_norm > 0.0:
                grad_values = [pair[0] for pair in grads_and_vars]
                variables = [pair[1] for pair in grads_and_vars]
                clipped_gradients, _ = tf.clip_by_global_norm(grad_values, clip_norm)
                optimizer.apply_gradients(zip(clipped_gradients, variables))
            else:
                optimizer.apply_gradients(grads_and_vars)

        if return_loss:
            return tf.cast(loss, tf.float32)
        return tf.constant(0.0, dtype=tf.float32)

    return train_step


def build_validation_step(tf, model, pad_id: int, use_xla: bool):
    @tf.function(reduce_retracing=True, jit_compile=use_xla)
    def validation_step(x_batch, y_batch):
        logits = model(x_batch, training=False)
        loss = compute_masked_loss(tf, y_batch, logits, pad_id=pad_id)
        return tf.cast(loss, tf.float32)

    return validation_step


def evaluate_validation(validation_dataset, validation_step) -> Optional[float]:
    total_loss = 0.0
    batches = 0
    for x_batch, y_batch in validation_dataset:
        total_loss += float(validation_step(x_batch, y_batch).numpy())
        batches += 1
    if batches == 0:
        return None
    return float(total_loss / batches)


def command_train(config: Dict[str, Any]) -> int:
    settings = config.get("settings") if isinstance(config.get("settings"), dict) else {}
    storage = config.get("storage") if isinstance(config.get("storage"), dict) else {}
    manifest = config.get("manifest") if isinstance(config.get("manifest"), dict) else {}
    positive_feedback_count = safe_int(config.get("positiveFeedbackCount"), 0)
    resume_requested = bool(config.get("resumeFromCheckpoint"))
    resume_epoch_offset = max(0, safe_int(config.get("resumeEpochOffset"), 0))
    resume_batch_offset = max(0, safe_int(config.get("resumeBatchOffset"), 0))
    stop_signal_path = str(config.get("stopSignalPath") or "").strip()

    training_settings = resolve_training_settings(settings)
    emit(
        {
            "type": "preparing",
            "stage": "backend_initializing",
            "message": "Initializing TensorFlow backend.",
            "progress": 5,
        }
    )
    tf_info = configure_tensorflow(training_settings)
    tf = tf_info["tf"]

    training_payload = read_training_payload(config)
    parquet_dataset_files = [
        str(raw_path)
        for raw_path in (training_payload.get("datasetFiles") or [])
        if str(raw_path).strip().lower().endswith(".parquet")
    ]
    if parquet_dataset_files:
        try:
            import pyarrow.parquet as _pq  # pylint: disable=import-outside-toplevel,unused-import
        except Exception as error:
            raise RuntimeError(
                "Parquet-файлы не могут быть обработаны: отсутствует пакет pyarrow в Python-окружении. "
                "Установите зависимости из server/python/requirements*.txt и повторите запуск."
            ) from error

    probe_iterator = create_training_text_iterator(training_payload)
    first_text = next(iter(probe_iterator), None)
    if not first_text:
        if parquet_dataset_files:
            raise RuntimeError(
                "Из parquet не удалось извлечь текст после предобработки. "
                "Проверьте, что в файле есть текстовые колонки (например: text/content/prompt/response) "
                "или данные строкового типа."
            )
        raise RuntimeError("No training texts found after preprocessing.")

    def tokenizer_text_stream():
        yield first_text
        yield from probe_iterator

    def corpus_text_stream():
        return create_training_text_iterator(training_payload)

    if stop_requested(stop_signal_path):
        return 0

    emit(
        {
            "type": "preparing",
            "stage": "tokenizer_building",
            "message": "Training tokenizer on user corpus.",
            "progress": 16,
        }
    )
    tokenizer = SimpleSubwordTokenizer.train(
        tokenizer_text_stream(),
        TokenizerConfig(vocabulary_limit=training_settings["vocabularyLimit"]),
    )

    if stop_requested(stop_signal_path):
        return 0

    resumed_from_checkpoint = False
    resume_restart_reason = ""
    model = None
    if resume_requested and checkpoint_exists(storage):
        try:
            model, checkpoint_tokenizer, _loaded_spec = load_checkpoint_runtime(storage)
            tokenizer = checkpoint_tokenizer
            resumed_from_checkpoint = True
        except Exception:
            resumed_from_checkpoint = False
            resume_restart_reason = "checkpoint_unavailable"
            model = None

    emit(
        {
            "type": "preparing",
            "stage": "dataset_building",
            "message": "Building tf.data streaming pipeline.",
            "progress": 44,
        }
    )

    from llm_data import create_datasets, prepare_tokenized_corpus  # pylint: disable=import-outside-toplevel
    from llm_model import build_gpt_model, build_model_config, count_model_parameters  # pylint: disable=import-outside-toplevel

    context_length = training_settings["sequenceLength"]
    tokenized_corpus = prepare_tokenized_corpus(
        texts=corpus_text_stream(),
        tokenizer=tokenizer,
        context_length=context_length,
        max_tokens_per_text=max(1024, context_length * 24),
        stride=training_settings["windowStride"],
    )
    if tokenized_corpus.sample_count <= 0:
        raise RuntimeError("Not enough tokenized samples for training.")

    dataset_bundle = create_datasets(
        corpus=tokenized_corpus,
        context_length=context_length,
        batch_size=training_settings["batchSize"],
        validation_split=training_settings["validationSplit"],
        stride=training_settings["windowStride"],
        shuffle_seed=1337,
        reshuffle_each_iteration=False,
        prefetch_batches=training_settings["datasetPrefetchBatches"],
        deterministic=True,
        private_threadpool_size=training_settings["datasetPrivateThreadpoolSize"],
    )

    if stop_requested(stop_signal_path):
        return 0

    model_config = build_model_config(training_settings, tokenizer.vocabulary_size, tokenizer.pad_id)
    if model is None:
        emit(
            {
                "type": "preparing",
                "stage": "model_initializing",
                "message": "Creating GPT-like autoregressive Transformer.",
                "progress": 74,
            }
        )
        model = build_gpt_model(model_config)
    else:
        loaded_context = safe_int(getattr(model, "input_shape", [None, context_length])[1], context_length)
        loaded_vocab = safe_int(getattr(model, "output_shape", [None, None, tokenizer.vocabulary_size])[-1], tokenizer.vocabulary_size)
        if loaded_context != context_length or loaded_vocab != tokenizer.vocabulary_size:
            resumed_from_checkpoint = False
            resume_restart_reason = "checkpoint_incompatible"
            model = build_gpt_model(model_config)
            resume_epoch_offset = 0

    requested_epochs = training_settings["epochs"]
    effective_resume_epoch_offset = resume_epoch_offset if resumed_from_checkpoint else 0
    if not resumed_from_checkpoint:
        resume_batch_offset = 0
    effective_epochs = effective_resume_epoch_offset + requested_epochs
    parameter_count = count_model_parameters(model)
    batches_per_epoch = dataset_bundle.batches_per_epoch
    resume_batch_offset = min(resume_batch_offset, max(0, batches_per_epoch - 1))
    minimum_steps = max(1, batches_per_epoch * max(1, requested_epochs))

    emit(
        {
            "type": "prepared",
            "datasetSampleCount": tokenized_corpus.sample_count,
            "trainingSampleCount": dataset_bundle.training_sample_count,
            "validationSampleCount": dataset_bundle.validation_sample_count,
            "batchesPerEpoch": batches_per_epoch,
            "tokenizerVocabularySize": tokenizer.vocabulary_size,
            "parameterCount": parameter_count,
            "backendName": tf_info["backendName"],
            "backendLabel": tf_info["backendLabel"],
            "executionMode": training_settings["executionMode"],
            "backendWarning": tf_info["backendWarning"],
            "optimizer": training_settings["optimizer"],
            "gradientClipNorm": training_settings["gradientClipNorm"],
            "requestedBatchSize": training_settings["batchSize"],
            "effectiveBatchSize": training_settings["batchSize"],
            "windowStride": training_settings["windowStride"],
            "batchSizeAdjusted": False,
            "minimumBatchesPerEpoch": 1,
            "resumedFromCheckpoint": resumed_from_checkpoint,
            "requestedResumeEpochOffset": resume_epoch_offset,
            "effectiveResumeEpochOffset": effective_resume_epoch_offset,
            "requestedResumeBatchOffset": resume_batch_offset,
            "effectiveResumeBatchOffset": resume_batch_offset,
            "requestedEpochs": requested_epochs,
            "effectiveEpochs": effective_epochs,
            "resumeRestartReason": resume_restart_reason,
            "batchReportInterval": training_settings["batchReportInterval"],
            "metricsReportInterval": training_settings["metricsReportInterval"],
            "historyReportInterval": training_settings["historyReportInterval"],
            "datasetPrefetchBatches": training_settings["datasetPrefetchBatches"],
            "datasetPrivateThreadpoolSize": training_settings["datasetPrivateThreadpoolSize"],
        }
    )

    optimizer_name = str(training_settings.get("optimizer") or "adam_legacy").lower()
    if optimizer_name == "adam_legacy":
        legacy_module = getattr(tf.keras.optimizers, "legacy", None)
        AdamClass = getattr(legacy_module, "Adam", tf.keras.optimizers.Adam)
    else:
        AdamClass = tf.keras.optimizers.Adam
    base_optimizer = AdamClass(learning_rate=training_settings["learningRate"])
    optimizer = base_optimizer
    use_loss_scale = False
    if tf_info["mixedPrecision"]:
        try:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
            use_loss_scale = True
        except Exception:
            optimizer = base_optimizer
            use_loss_scale = False

    xla_requested = bool(training_settings.get("enableXla")) and bool(tf_info.get("usingGpu"))
    xla_enabled = xla_requested
    gradient_clip_norm = max(0.0, float(training_settings.get("gradientClipNorm") or 0.0))
    train_step_metrics = build_train_step(
        tf=tf,
        model=model,
        optimizer=optimizer,
        clip_norm=gradient_clip_norm,
        pad_id=tokenizer.pad_id,
        use_loss_scale=use_loss_scale,
        use_xla=xla_enabled,
        return_loss=True,
    )
    train_step_fast = build_train_step(
        tf=tf,
        model=model,
        optimizer=optimizer,
        clip_norm=gradient_clip_norm,
        pad_id=tokenizer.pad_id,
        use_loss_scale=use_loss_scale,
        use_xla=xla_enabled,
        return_loss=False,
    )
    validation_step = build_validation_step(
        tf=tf,
        model=model,
        pad_id=tokenizer.pad_id,
        use_xla=xla_enabled,
    )

    history: List[Dict[str, Any]] = []
    processed_batches = 0
    completed_epochs = effective_resume_epoch_offset
    total_loss = 0.0
    last_loss: Optional[float] = None
    last_validation_loss: Optional[float] = None
    best_validation_loss: Optional[float] = None
    best_weights = None
    heartbeat_interval_s = max(5, safe_int(training_settings.get("heartbeatIntervalSeconds"), 10))
    metrics_interval = max(1, safe_int(training_settings.get("metricsReportInterval"), 16))
    history_interval = max(1, safe_int(training_settings.get("historyReportInterval"), metrics_interval))
    batch_report_interval = max(1, safe_int(training_settings.get("batchReportInterval"), 1))
    recovery_checkpoint_interval_batches = max(
        0,
        safe_non_negative_int(training_settings.get("recoveryCheckpointIntervalBatches"), 1000),
    )
    recovery_checkpoint_interval_seconds = max(
        0.0,
        safe_float(training_settings.get("recoveryCheckpointIntervalMinutes"), 20.0) * 60.0,
    )
    total_planned_batches = max(1, batches_per_epoch * effective_epochs)
    run_total_planned_batches = max(
        1,
        (batches_per_epoch * max(requested_epochs, 1)) - (resume_batch_offset if requested_epochs > 0 else 0),
    )
    smoothed_batch_seconds: Optional[float] = None
    smoothed_eta_seconds: Optional[float] = None
    last_emitted_eta_seconds: Optional[int] = None
    was_stopped = False
    stopped_at_epoch = effective_resume_epoch_offset
    stopped_at_batch = 0
    last_recovery_checkpoint_batch = max(
        0,
        (max(effective_resume_epoch_offset, 0) * max(batches_per_epoch, 1)) +
        (resume_batch_offset if effective_resume_epoch_offset > 0 else 0),
    )
    last_recovery_checkpoint_at = time.time()
    last_recovery_resume_epoch_offset = effective_resume_epoch_offset
    last_recovery_resume_batch_offset = resume_batch_offset if effective_resume_epoch_offset > 0 else 0
    heartbeat_state = {
        "phase": "training_batches",
        "message": "Обучение идет в реальном времени.",
        "epoch": max(effective_resume_epoch_offset + 1, 1),
        "batch": max(resume_batch_offset, 0),
        "batchesThisEpoch": batches_per_epoch,
        "effectiveEpochs": effective_epochs,
        "completedBatches": 0,
        "totalPlannedBatches": total_planned_batches,
        "etaSeconds": None,
        "etaAt": None,
        "avgBatchDurationMs": None,
        "throughputBatchesPerSec": None,
        "startedAt": now_iso(),
    }
    heartbeat_stop_event = threading.Event()
    heartbeat_thread = start_training_heartbeat(heartbeat_state, heartbeat_interval_s, heartbeat_stop_event)

    def update_heartbeat(**updates: Any) -> None:
        heartbeat_state.update(updates)

    def emit_recovery_checkpoint(
        *,
        stage: str,
        message: str,
        epoch_offset: int,
        batch_offset: int,
        processed_batches_for_checkpoint: int,
    ) -> None:
        emit(
            {
                "type": "recovery_checkpoint",
                "stage": stage,
                "message": message,
                "epochOffset": max(0, int(epoch_offset)),
                "batchOffset": max(0, int(batch_offset)),
                "processedBatches": max(0, int(processed_batches_for_checkpoint)),
                "effectiveEpochs": effective_epochs,
                "batchesThisEpoch": batches_per_epoch,
                "savedAt": now_iso(),
            }
        )

    for epoch in range(1, requested_epochs + 1):
        batch_index = resume_batch_offset if epoch == 1 else 0
        epoch_dataset = dataset_bundle.train_dataset.skip(resume_batch_offset) if epoch == 1 and resume_batch_offset > 0 else dataset_bundle.train_dataset
        for x_batch, y_batch in epoch_dataset:
            if stop_requested(stop_signal_path):
                was_stopped = True
                stopped_at_epoch = effective_resume_epoch_offset + epoch
                stopped_at_batch = batch_index
                break

            batch_index += 1
            current_epoch = effective_resume_epoch_offset + epoch
            update_heartbeat(
                phase="training_batches",
                message="Обучение идет в реальном времени.",
                epoch=current_epoch,
                batch=batch_index,
                batchesThisEpoch=batches_per_epoch,
                effectiveEpochs=effective_epochs,
            )
            batch_started_at = time.time()
            should_collect_metrics = (
                batch_index == 1
                or batch_index == batches_per_epoch
                or (batch_index % metrics_interval == 0)
            )
            try:
                if should_collect_metrics:
                    loss_value = float(train_step_metrics(x_batch, y_batch).numpy())
                else:
                    train_step_fast(x_batch, y_batch)
                    loss_value = float(last_loss) if last_loss is not None else 0.0
            except Exception as error:
                error_message = clean_message(error)
                can_disable_xla = xla_enabled and ("xla" in error_message.lower() or "jit" in error_message.lower())
                if not can_disable_xla:
                    raise

                xla_enabled = False
                train_step_metrics = build_train_step(
                    tf=tf,
                    model=model,
                    optimizer=optimizer,
                    clip_norm=gradient_clip_norm,
                    pad_id=tokenizer.pad_id,
                    use_loss_scale=use_loss_scale,
                    use_xla=False,
                    return_loss=True,
                )
                train_step_fast = build_train_step(
                    tf=tf,
                    model=model,
                    optimizer=optimizer,
                    clip_norm=gradient_clip_norm,
                    pad_id=tokenizer.pad_id,
                    use_loss_scale=use_loss_scale,
                    use_xla=False,
                    return_loss=False,
                )
                validation_step = build_validation_step(
                    tf=tf,
                    model=model,
                    pad_id=tokenizer.pad_id,
                    use_xla=False,
                )
                emit(
                    {
                        "type": "warning",
                        "stage": "xla_disabled",
                        "message": "XLA JIT отключен из-за ошибки рантайма, продолжаем обучение без JIT.",
                        "details": error_message[:240],
                    }
                )
                loss_value = float(train_step_metrics(x_batch, y_batch).numpy())
            processed_batches += 1
            total_loss += loss_value
            last_loss = loss_value

            average_loss = total_loss / max(1, processed_batches)
            now = time.time()
            batch_duration_s = max(0.0, now - batch_started_at)
            if smoothed_batch_seconds is None:
                smoothed_batch_seconds = batch_duration_s
            else:
                smoothed_batch_seconds = (smoothed_batch_seconds * 0.9) + (batch_duration_s * 0.1)

            completed_batches_global = min(
                max(((max(current_epoch, 1) - 1) * max(batches_per_epoch, 1)) + max(batch_index, 1), 1),
                total_planned_batches,
            )
            run_completed_batches = min(run_total_planned_batches, max(0, processed_batches))
            remaining_batches = max(0, run_total_planned_batches - run_completed_batches)
            eta_seconds = None
            eta_at = None
            if smoothed_batch_seconds is not None:
                raw_eta_seconds = max(0.0, remaining_batches * smoothed_batch_seconds)
                if smoothed_eta_seconds is None:
                    smoothed_eta_seconds = raw_eta_seconds
                else:
                    smoothed_eta_seconds = (smoothed_eta_seconds * 0.96) + (raw_eta_seconds * 0.04)
                eta_seconds = int(max(0, round(smoothed_eta_seconds)))
                if last_emitted_eta_seconds is not None:
                    max_change = max(2, min(30, int(round(last_emitted_eta_seconds * 0.04))))
                    eta_seconds = max(
                        last_emitted_eta_seconds - max_change,
                        min(last_emitted_eta_seconds + max_change, eta_seconds),
                    )
                if last_emitted_eta_seconds is None or abs(eta_seconds - last_emitted_eta_seconds) >= 1:
                    last_emitted_eta_seconds = eta_seconds
                else:
                    eta_seconds = last_emitted_eta_seconds
                eta_at = datetime.fromtimestamp(now + eta_seconds, tz=timezone.utc).isoformat()
            update_heartbeat(
                phase="training_batches",
                message="Обучение идет в реальном времени.",
                epoch=current_epoch,
                batch=batch_index,
                completedBatches=run_completed_batches,
                totalPlannedBatches=run_total_planned_batches,
                etaSeconds=eta_seconds,
                etaAt=eta_at,
                avgBatchDurationMs=round((smoothed_batch_seconds or 0.0) * 1000.0, 2),
                throughputBatchesPerSec=round(1.0 / max(smoothed_batch_seconds or 1e-9, 1e-9), 4),
            )

            history_entry = None
            should_store_history = (
                should_collect_metrics
                and (
                    batch_index == 1
                    or batch_index == batches_per_epoch
                    or (batch_index % history_interval == 0)
                )
            )
            if should_store_history:
                history_entry = {
                    "step": processed_batches,
                    "epoch": current_epoch,
                    "batch": batch_index,
                    "loss": round(loss_value, 6),
                }
                history.append(history_entry)

            should_emit_batch = (
                batch_index == 1
                or batch_index == batches_per_epoch
                or (batch_index % batch_report_interval == 0)
            )
            if should_emit_batch:
                emit(
                    {
                        "type": "batch",
                        "lastLoss": round(loss_value, 6),
                        "averageLoss": round(average_loss, 6),
                        "processedBatches": processed_batches,
                        "completedBatches": completed_batches_global,
                        "totalPlannedBatches": total_planned_batches,
                        "runCompletedBatches": run_completed_batches,
                        "runTotalPlannedBatches": run_total_planned_batches,
                        "runCurrentEpoch": epoch,
                        "runTotalEpochs": requested_epochs,
                        "epoch": current_epoch,
                        "batch": batch_index,
                        "batchesThisEpoch": batches_per_epoch,
                        "effectiveEpochs": effective_epochs,
                        "requestedEpochs": requested_epochs,
                        "minimumTrainingSteps": minimum_steps,
                        "enforcedStepBudget": False,
                        "metricsCollected": should_collect_metrics,
                        "stepDurationMs": round(batch_duration_s * 1000.0, 2),
                        "avgBatchDurationMs": round((smoothed_batch_seconds or 0.0) * 1000.0, 2),
                        "throughputBatchesPerSec": round(1.0 / max(smoothed_batch_seconds or 1e-9, 1e-9), 4),
                        "etaSeconds": eta_seconds,
                        "etaAt": eta_at,
                        "historyEntry": history_entry,
                    }
                )
            should_save_recovery_checkpoint = False
            if recovery_checkpoint_interval_batches > 0:
                should_save_recovery_checkpoint = (
                    completed_batches_global - last_recovery_checkpoint_batch
                ) >= recovery_checkpoint_interval_batches
            if not should_save_recovery_checkpoint and recovery_checkpoint_interval_seconds > 0:
                should_save_recovery_checkpoint = (
                    now - last_recovery_checkpoint_at
                ) >= recovery_checkpoint_interval_seconds
            if should_save_recovery_checkpoint and completed_batches_global < total_planned_batches:
                checkpoint_saved_at = now_iso()
                update_heartbeat(
                    phase="saving_recovery_checkpoint",
                    message="Сохраняется промежуточный recovery-чекпоинт обучения.",
                )

                def emit_recovery_checkpoint_stage(stage: str, message: str) -> None:
                    emit_recovery_checkpoint(
                        stage=stage,
                        message=message,
                        epoch_offset=max(current_epoch - 1, 0),
                        batch_offset=batch_index,
                        processed_batches_for_checkpoint=processed_batches,
                    )

                emit_recovery_checkpoint_stage(
                    stage="started",
                    message="Сохраняется промежуточный recovery-чекпоинт, чтобы обучение можно было продолжить после сбоя.",
                )
                recovery_spec_payload = build_spec_payload(
                    manifest=manifest,
                    parameter_count=parameter_count,
                    completed_epochs=max(current_epoch - 1, 0),
                    saved_at=checkpoint_saved_at,
                    tokenizer=tokenizer,
                    tokenized_corpus=tokenized_corpus,
                    training_settings=training_settings,
                    model_config=model_config,
                    checkpoint_kind="recovery",
                    next_resume_epoch_offset=max(current_epoch - 1, 0),
                    next_resume_batch_offset=batch_index,
                    processed_batches=processed_batches,
                )
                save_checkpoint_runtime(
                    storage=storage,
                    model=model,
                    tokenizer=tokenizer,
                    spec_payload=recovery_spec_payload,
                    on_progress=emit_recovery_checkpoint_stage,
                )
                last_recovery_checkpoint_batch = completed_batches_global
                last_recovery_checkpoint_at = time.time()
                last_recovery_resume_epoch_offset = max(current_epoch - 1, 0)
                last_recovery_resume_batch_offset = batch_index
                emit_recovery_checkpoint(
                    stage="ready",
                    message="Промежуточный recovery-чекпоинт сохранен.",
                    epoch_offset=last_recovery_resume_epoch_offset,
                    batch_offset=last_recovery_resume_batch_offset,
                    processed_batches_for_checkpoint=processed_batches,
                )
                update_heartbeat(
                    phase="training_batches",
                    message="Обучение идет в реальном времени.",
                    epoch=current_epoch,
                    batch=batch_index,
                )
            stopped_at_epoch = current_epoch
            stopped_at_batch = batch_index

        if was_stopped:
            break

        update_heartbeat(
            phase="validation_epoch",
            message=f"Валидация эпохи {effective_resume_epoch_offset + epoch}/{effective_epochs}.",
            epoch=effective_resume_epoch_offset + epoch,
            batch=batches_per_epoch,
        )
        emit(
            {
                "type": "validation",
                "stage": "start",
                "epoch": effective_resume_epoch_offset + epoch,
                "effectiveEpochs": effective_epochs,
                "message": f"Валидация эпохи {effective_resume_epoch_offset + epoch}/{effective_epochs}.",
            }
        )
        last_validation_loss = evaluate_validation(dataset_bundle.validation_dataset, validation_step)
        emit(
            {
                "type": "validation",
                "stage": "done",
                "epoch": effective_resume_epoch_offset + epoch,
                "effectiveEpochs": effective_epochs,
                "validationLoss": round(last_validation_loss, 6) if last_validation_loss is not None else None,
                "message": f"Валидация эпохи {effective_resume_epoch_offset + epoch}/{effective_epochs} завершена.",
            }
        )
        update_heartbeat(
            phase="training_batches",
            message="Обучение идет в реальном времени.",
            epoch=effective_resume_epoch_offset + epoch,
            batch=batches_per_epoch,
        )
        if last_validation_loss is not None and (
            best_validation_loss is None or last_validation_loss < best_validation_loss
        ):
            best_validation_loss = last_validation_loss
            best_weights = model.get_weights()

        last_recovery_checkpoint_batch = max(
            last_recovery_checkpoint_batch,
            (max(effective_resume_epoch_offset + epoch, 0) * max(batches_per_epoch, 1)),
        )
        last_recovery_checkpoint_at = time.time()
        last_recovery_resume_epoch_offset = effective_resume_epoch_offset + epoch
        last_recovery_resume_batch_offset = 0
        completed_epochs = effective_resume_epoch_offset + epoch

    if best_weights is not None:
        model.set_weights(best_weights)

    average_loss = (total_loss / processed_batches) if processed_batches > 0 else None
    perplexity = math.exp(average_loss) if average_loss is not None else None

    def emit_checkpoint_stage(stage: str, message: str) -> None:
        update_heartbeat(
            phase="saving_checkpoint",
            message=message,
            epoch=completed_epochs,
            batch=stopped_at_batch,
        )
        emit(
            {
                "type": "checkpointing",
                "stage": stage,
                "message": message,
            }
        )

    emit_checkpoint_stage("started", "Обучение завершено, начинается сохранение чекпоинта.")
    parameter_count = count_model_parameters(model)
    saved_at = now_iso()
    language_model = {
        "kind": "tf-keras-llm",
        "vocabularySize": tokenizer.vocabulary_size,
        "parameterCount": parameter_count,
        "tokenizerReady": True,
        "checkpointReady": True,
        "lastSavedAt": saved_at,
        "trainingSequenceCount": tokenized_corpus.sample_count,
        "corpusTokenCount": tokenized_corpus.token_count,
        "feedbackExampleCount": positive_feedback_count,
        "validationLoss": round(last_validation_loss, 6) if last_validation_loss is not None else None,
        "bestValidationLoss": round(best_validation_loss, 6) if best_validation_loss is not None else None,
        "trainingSampleCount": dataset_bundle.training_sample_count,
        "validationSampleCount": dataset_bundle.validation_sample_count,
    }

    spec_payload = build_spec_payload(
        manifest=manifest,
        parameter_count=parameter_count,
        completed_epochs=completed_epochs,
        saved_at=saved_at,
        tokenizer=tokenizer,
        tokenized_corpus=tokenized_corpus,
        training_settings=training_settings,
        model_config=model_config,
        checkpoint_kind="final",
        next_resume_epoch_offset=completed_epochs,
        next_resume_batch_offset=min(stopped_at_batch, max(0, batches_per_epoch - 1)) if was_stopped else 0,
        processed_batches=processed_batches,
    )
    save_checkpoint_runtime(
        storage=storage,
        model=model,
        tokenizer=tokenizer,
        spec_payload=spec_payload,
        on_progress=emit_checkpoint_stage,
    )

    training_result = {
        "lastLoss": round(last_loss, 6) if last_loss is not None else None,
        "averageLoss": round(average_loss, 6) if average_loss is not None else None,
        "perplexity": round(perplexity, 6) if perplexity is not None and math.isfinite(perplexity) else None,
        "validationLoss": round(last_validation_loss, 6) if last_validation_loss is not None else None,
        "bestValidationLoss": round(best_validation_loss, 6) if best_validation_loss is not None else None,
        "bestEpoch": completed_epochs if best_validation_loss is not None else None,
        "processedBatches": processed_batches,
        "completedEpochs": completed_epochs,
        "effectiveEpochs": effective_epochs,
        "requestedEpochs": requested_epochs,
        "minimumTrainingSteps": minimum_steps,
        "enforcedStepBudget": False,
        "trainingSampleCount": dataset_bundle.training_sample_count,
        "validationSampleCount": dataset_bundle.validation_sample_count,
        "stopRequested": was_stopped,
        "stoppedAtEpoch": stopped_at_epoch,
        "stoppedAtBatch": stopped_at_batch,
        "nextResumeEpochOffset": completed_epochs,
        "nextResumeBatchOffset": min(stopped_at_batch, max(0, batches_per_epoch - 1)) if was_stopped else 0,
        "history": history[-360:],
    }

    heartbeat_stop_event.set()
    heartbeat_thread.join(timeout=1.0)
    emit(
        {
            "type": "done",
            "trainingResult": training_result,
            "parameterCount": parameter_count,
            "tokenizerVocabularySize": tokenizer.vocabulary_size,
            "resumedFromCheckpoint": resumed_from_checkpoint,
            "effectiveResumeEpochOffset": effective_resume_epoch_offset,
            "languageModel": language_model,
        }
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TensorFlow/Keras backend for AI-generator.")
    parser.add_argument("command", choices=["load_runtime", "train", "generate", "inspect_dataset"])
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = read_json(args.config)

    if args.command == "load_runtime":
        return command_load_runtime(config)
    if args.command == "train":
        return command_train(config)
    if args.command == "generate":
        return command_generate(config)
    if args.command == "inspect_dataset":
        return command_inspect_dataset(config)
    emit({"ok": False, "error": f"Unsupported command: {args.command}"})
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        emit(
            {
                "type": "error",
                "ok": False,
                "error": {
                    "message": clean_message(error, "Python backend failed."),
                },
            }
        )
        raise
