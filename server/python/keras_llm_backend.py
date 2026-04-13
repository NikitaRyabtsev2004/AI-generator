#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_tokenizer import SPECIAL_TOKENS, SimpleSubwordTokenizer, TokenizerConfig


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def clean_message(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def read_json(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def resolve_training_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    training = settings.get("training") if isinstance(settings.get("training"), dict) else settings
    embedding_size = max(32, safe_int(training.get("embeddingSize"), 128))
    attention_heads = max(1, safe_int(training.get("attentionHeads"), 4))
    while attention_heads > 1 and embedding_size % attention_heads != 0:
        attention_heads -= 1

    return {
        "executionMode": str(training.get("executionMode") or "native_preferred"),
        "sequenceLength": max(16, safe_int(training.get("sequenceLength"), 128)),
        "embeddingSize": embedding_size,
        "attentionHeads": attention_heads,
        "transformerLayers": max(1, safe_int(training.get("transformerLayers"), 4)),
        "feedForwardSize": max(embedding_size, safe_int(training.get("feedForwardSize"), embedding_size * 4)),
        "dropout": min(0.6, max(0.0, safe_float(training.get("dropout"), 0.1))),
        "learningRate": max(1e-6, safe_float(training.get("learningRate"), 0.001)),
        "batchSize": max(1, safe_int(training.get("batchSize"), 8)),
        "epochs": max(1, safe_int(training.get("epochs"), 1)),
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
        temp_path.write_bytes(Path(weights_path).read_bytes())
        model = tf.keras.models.load_model(str(temp_path), custom_objects=custom_objects, compile=False)
    return model


def save_keras_model_to_bytes(model, weights_path: str) -> None:
    target = Path(weights_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ai_generator_save_") as temp_dir:
        temp_path = Path(temp_dir) / "runtime_model.keras"
        model.save(str(temp_path), include_optimizer=False)
        target.write_bytes(temp_path.read_bytes())


def load_checkpoint_runtime(storage: Dict[str, Any]) -> Tuple[Any, SimpleSubwordTokenizer, Dict[str, Any]]:
    paths = get_storage_paths(storage)
    tokenizer, spec_payload = load_tokenizer_and_spec(storage)
    model = load_keras_model_from_bytes(paths["weightsPath"])
    return model, tokenizer, spec_payload


def save_checkpoint_runtime(
    storage: Dict[str, Any],
    model,
    tokenizer: SimpleSubwordTokenizer,
    spec_payload: Dict[str, Any],
) -> None:
    paths = get_storage_paths(storage)
    tokenizer.save(paths["tokenizerPath"])
    save_keras_model_to_bytes(model, paths["weightsPath"])
    write_json(paths["specPath"], spec_payload)


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
    losses = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    mask = tf.cast(tf.not_equal(labels, pad_id), losses.dtype)
    masked = losses * mask
    denominator = tf.maximum(tf.reduce_sum(mask), tf.constant(1.0, dtype=losses.dtype))
    return tf.reduce_sum(masked) / denominator


def train_batch(tf, model, optimizer, x_batch, y_batch, clip_norm: float, pad_id: int, use_loss_scale: bool) -> float:
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = compute_masked_loss(tf, y_batch, logits, pad_id=pad_id)
        scaled_loss = optimizer.get_scaled_loss(loss) if use_loss_scale and hasattr(optimizer, "get_scaled_loss") else loss

    gradients = tape.gradient(scaled_loss, model.trainable_variables)
    if use_loss_scale and hasattr(optimizer, "get_unscaled_gradients"):
        gradients = optimizer.get_unscaled_gradients(gradients)

    sanitized_gradients = []
    for gradient, variable in zip(gradients, model.trainable_variables):
        sanitized_gradients.append(gradient if gradient is not None else tf.zeros_like(variable))
    clipped_gradients, _ = tf.clip_by_global_norm(sanitized_gradients, clip_norm)
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
    return float(loss.numpy())


def evaluate_validation(tf, model, validation_dataset, pad_id: int) -> Optional[float]:
    total_loss = 0.0
    batches = 0
    for x_batch, y_batch in validation_dataset:
        logits = model(x_batch, training=False)
        batch_loss = compute_masked_loss(tf, y_batch, logits, pad_id=pad_id)
        total_loss += float(batch_loss.numpy())
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
    probe_iterator = create_training_text_iterator(training_payload)
    first_text = next(iter(probe_iterator), None)
    if not first_text:
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
    )
    if tokenized_corpus.sample_count <= 0:
        raise RuntimeError("Not enough tokenized samples for training.")

    dataset_bundle = create_datasets(
        corpus=tokenized_corpus,
        context_length=context_length,
        batch_size=training_settings["batchSize"],
        validation_split=training_settings["validationSplit"],
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
    effective_epochs = effective_resume_epoch_offset + requested_epochs
    parameter_count = count_model_parameters(model)
    batches_per_epoch = dataset_bundle.batches_per_epoch
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
            "requestedBatchSize": training_settings["batchSize"],
            "effectiveBatchSize": training_settings["batchSize"],
            "batchSizeAdjusted": False,
            "minimumBatchesPerEpoch": 1,
            "resumedFromCheckpoint": resumed_from_checkpoint,
            "requestedResumeEpochOffset": resume_epoch_offset,
            "effectiveResumeEpochOffset": effective_resume_epoch_offset,
            "requestedEpochs": requested_epochs,
            "effectiveEpochs": effective_epochs,
            "resumeRestartReason": resume_restart_reason,
        }
    )

    base_optimizer = tf.keras.optimizers.Adam(learning_rate=training_settings["learningRate"])
    optimizer = base_optimizer
    use_loss_scale = False
    if tf_info["mixedPrecision"]:
        try:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
            use_loss_scale = True
        except Exception:
            optimizer = base_optimizer
            use_loss_scale = False

    history: List[Dict[str, Any]] = []
    processed_batches = 0
    completed_epochs = effective_resume_epoch_offset
    total_loss = 0.0
    last_loss: Optional[float] = None
    last_validation_loss: Optional[float] = None
    best_validation_loss: Optional[float] = None
    best_weights = None
    heartbeat_interval_s = 15
    last_heartbeat_at = time.time()
    report_interval = max(1, batches_per_epoch // 60)
    was_stopped = False

    for epoch in range(1, requested_epochs + 1):
        batch_index = 0
        for x_batch, y_batch in dataset_bundle.train_dataset:
            if stop_requested(stop_signal_path):
                was_stopped = True
                break

            batch_index += 1
            loss_value = train_batch(
                tf=tf,
                model=model,
                optimizer=optimizer,
                x_batch=x_batch,
                y_batch=y_batch,
                clip_norm=1.0,
                pad_id=tokenizer.pad_id,
                use_loss_scale=use_loss_scale,
            )
            processed_batches += 1
            total_loss += loss_value
            last_loss = loss_value

            should_report = (
                batch_index == 1
                or batch_index == batches_per_epoch
                or (batch_index % report_interval == 0)
            )
            if should_report:
                average_loss = total_loss / max(1, processed_batches)
                history_entry = {
                    "step": processed_batches,
                    "epoch": effective_resume_epoch_offset + epoch,
                    "batch": batch_index,
                    "loss": round(loss_value, 6),
                }
                history.append(history_entry)
                emit(
                    {
                        "type": "batch",
                        "lastLoss": round(loss_value, 6),
                        "averageLoss": round(average_loss, 6),
                        "processedBatches": processed_batches,
                        "epoch": effective_resume_epoch_offset + epoch,
                        "batch": batch_index,
                        "batchesThisEpoch": batches_per_epoch,
                        "effectiveEpochs": effective_epochs,
                        "requestedEpochs": effective_epochs,
                        "minimumTrainingSteps": minimum_steps,
                        "enforcedStepBudget": False,
                        "historyEntry": history_entry,
                    }
                )

            now = time.time()
            if now - last_heartbeat_at >= heartbeat_interval_s:
                emit(
                    {
                        "type": "heartbeat",
                        "epoch": effective_resume_epoch_offset + epoch,
                        "batch": batch_index,
                        "batchesThisEpoch": batches_per_epoch,
                        "effectiveEpochs": effective_epochs,
                        "startedAt": now_iso(),
                    }
                )
                last_heartbeat_at = now

        if was_stopped:
            break

        last_validation_loss = evaluate_validation(
            tf=tf,
            model=model,
            validation_dataset=dataset_bundle.validation_dataset,
            pad_id=tokenizer.pad_id,
        )
        if last_validation_loss is not None and (
            best_validation_loss is None or last_validation_loss < best_validation_loss
        ):
            best_validation_loss = last_validation_loss
            best_weights = model.get_weights()

        completed_epochs = effective_resume_epoch_offset + epoch

    if best_weights is not None:
        model.set_weights(best_weights)

    average_loss = (total_loss / processed_batches) if processed_batches > 0 else None
    perplexity = math.exp(average_loss) if average_loss is not None else None

    emit({"type": "checkpointing"})
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

    model_settings_manifest = {
        "sequenceLength": training_settings["sequenceLength"],
        "embeddingSize": training_settings["embeddingSize"],
        "attentionHeads": training_settings["attentionHeads"],
        "transformerLayers": training_settings["transformerLayers"],
        "feedForwardSize": training_settings["feedForwardSize"],
        "dropout": training_settings["dropout"],
        "learningRate": training_settings["learningRate"],
    }
    spec_payload = {
        "version": 2,
        "format": "keras_llm_checkpoint_v1",
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
        },
    }
    save_checkpoint_runtime(storage=storage, model=model, tokenizer=tokenizer, spec_payload=spec_payload)

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
        "requestedEpochs": effective_epochs,
        "minimumTrainingSteps": minimum_steps,
        "enforcedStepBudget": False,
        "trainingSampleCount": dataset_bundle.training_sample_count,
        "validationSampleCount": dataset_bundle.validation_sample_count,
        "stopRequested": was_stopped,
        "history": history[-360:],
    }

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
