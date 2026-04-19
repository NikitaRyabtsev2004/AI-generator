const fs = require('fs/promises');
const { runPythonBackendJson } = require('./pythonBridge');

const TOKEN_PATTERN = /[\p{L}\p{N}_]+|[^\s]/gu;

function normalizeFilesystemPath(value) {
  return typeof value === 'string' ? value.trim() : '';
}

async function pathExists(targetPath) {
  const normalizedPath = normalizeFilesystemPath(targetPath);
  if (!normalizedPath) {
    return false;
  }

  try {
    await fs.access(normalizedPath);
    return true;
  } catch (_error) {
    return false;
  }
}

function tokenizeForModel(input = '') {
  const normalized = String(input || '').trim();
  if (!normalized) {
    return [];
  }

  return normalized.match(TOKEN_PATTERN) || [];
}

async function loadRuntime({ storage, settings }) {
  const tokenizerPath = normalizeFilesystemPath(storage?.tokenizerPath);
  const weightsPath = normalizeFilesystemPath(storage?.neuralWeightsPath);
  const specPath = normalizeFilesystemPath(storage?.neuralSpecPath);

  const checkpointReady = (
    await pathExists(tokenizerPath) &&
    await pathExists(weightsPath) &&
    await pathExists(specPath)
  );
  if (!checkpointReady) {
    return null;
  }

  try {
    const payload = await runPythonBackendJson(
      'load_runtime',
      {
        storage,
        settings,
      },
      {
        timeoutMs: 45000,
      }
    );

    if (!payload?.ok || !payload?.checkpointReady) {
      return null;
    }

    return {
      model: {
        kind: 'tf-keras-runtime',
      },
      manifest: payload.manifest || {},
      tokenizer: payload.tokenizer || null,
      parameterCount: Number(payload.parameterCount) || 0,
      vocabularySize: Number(payload.vocabularySize) || 0,
      storage: {
        ...storage,
      },
      format: payload.format || 'keras_llm_checkpoint_v1',
    };
  } catch (_error) {
    return null;
  }
}

function disposeRuntime(_runtime) {
  // The TensorFlow/Keras runtime is process-based (Python), so no local tensors to dispose.
}

async function generateText({
  runtime,
  promptText,
  settings,
  signal,
}) {
  if (!runtime?.storage) {
    return {
      text: '',
      generatedTokenIds: [],
    };
  }

  try {
    const payload = await runPythonBackendJson(
      'generate',
      {
        storage: runtime.storage,
        settings,
        promptText: String(promptText || ''),
      },
      {
        timeoutMs: 90000,
        signal,
      }
    );

    return {
      text: String(payload?.text || '').trim(),
      generatedTokenIds: Array.isArray(payload?.generatedTokenIds) ? payload.generatedTokenIds : [],
    };
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw error;
    }
    return {
      text: '',
      generatedTokenIds: [],
    };
  }
}

module.exports = {
  disposeRuntime,
  generateText,
  loadRuntime,
  tokenizeForModel,
};
