const { parentPort, workerData } = require('node:worker_threads');
const {
  buildTokenizer,
  countParameters,
  createDataset,
  createModel,
  disposeRuntime,
  ensureBackend,
  loadRuntime,
  saveRuntime,
  trainRuntime,
} = require('./neuralModel');

let stopRequested = false;

function buildEffectiveSettings(settings, backendName) {
  const effectiveSettings = structuredClone(settings);
  const requestedBatchSize = Math.max(1, Number(settings.training.batchSize) || 1);
  const isPureCpuBackend = backendName === 'cpu';
  const heavySequenceModel =
    Number(settings.training.sequenceLength) >= 80 ||
    Number(settings.training.hiddenSize) >= 192 ||
    Number(settings.training.recurrentLayers) >= 2;

  if (isPureCpuBackend) {
    effectiveSettings.training.batchSize = Math.min(
      requestedBatchSize,
      heavySequenceModel ? 1 : 2
    );
  } else {
    effectiveSettings.training.batchSize = requestedBatchSize;
  }

  return {
    effectiveSettings,
    requestedBatchSize,
    effectiveBatchSize: effectiveSettings.training.batchSize,
  };
}

function postMessage(type, payload = {}) {
  if (!parentPort) {
    return;
  }

  parentPort.postMessage({
    type,
    ...payload,
  });
}

function sameTokenizer(left, right) {
  if (!left || !right) {
    return false;
  }

  if (!Array.isArray(left.idToToken) || !Array.isArray(right.idToToken)) {
    return false;
  }

  if (left.idToToken.length !== right.idToToken.length) {
    return false;
  }

  for (let index = 0; index < left.idToToken.length; index += 1) {
    if (left.idToToken[index] !== right.idToToken[index]) {
      return false;
    }
  }

  return true;
}

if (parentPort) {
  parentPort.on('message', (message) => {
    if (message?.type === 'stop') {
      stopRequested = true;
    }
  });
}

(async () => {
  const {
    settings,
    trainingTexts,
    storage,
    resumeFromCheckpoint,
    manifest,
    tokenizerTokenCount,
    positiveFeedbackCount,
  } = workerData;

  const backendName = await ensureBackend();
  const { effectiveSettings, requestedBatchSize, effectiveBatchSize } = buildEffectiveSettings(
    settings,
    backendName
  );

  const tokenizer = buildTokenizer(
    trainingTexts,
    effectiveSettings.training.vocabularyLimit
  );
  const dataset = createDataset({
    texts: trainingTexts,
    tokenizer,
    sequenceLength: effectiveSettings.training.sequenceLength,
  });

  if (!dataset.sampleCount) {
    throw new Error(
      'После токенизации не осталось достаточного количества обучающих последовательностей.'
    );
  }

  let runtime = null;
  if (resumeFromCheckpoint) {
    runtime = await loadRuntime({ storage, settings: effectiveSettings });
    if (runtime?.tokenizer && !sameTokenizer(runtime.tokenizer, tokenizer)) {
      disposeRuntime(runtime);
      runtime = null;
    }
  }

  if (!runtime?.model) {
    runtime = {
      model: createModel({
        vocabularySize: tokenizer.vocabularySize,
        settings: effectiveSettings,
      }),
      tokenizer,
    };
  } else {
    runtime.tokenizer = tokenizer;
  }

  const parameterCount = countParameters(runtime.model);
  const batchesPerEpoch = Math.max(
    Math.ceil(dataset.sampleCount / Math.max(1, effectiveSettings.training.batchSize)),
    1
  );

  postMessage('prepared', {
    datasetSampleCount: dataset.sampleCount,
    batchesPerEpoch,
    tokenizerVocabularySize: tokenizer.vocabularySize,
    parameterCount,
    backendName,
    requestedBatchSize,
    effectiveBatchSize,
  });

  const trainingResult = await trainRuntime({
    runtime,
    dataset,
    settings: effectiveSettings,
    onBatchEnd: async (payload) => {
      postMessage('batch', payload);
    },
    shouldStop: () => stopRequested,
  });

  if (!trainingResult.processedBatches && !trainingResult.stopRequested) {
    throw new Error(
      'Обучение не обработало ни одного батча. Уменьшите размер корпуса или параметры модели и попробуйте снова.'
    );
  }

  const savedAt = new Date().toISOString();
  postMessage('checkpointing');
  await saveRuntime({
    model: runtime.model,
    tokenizer,
    storage,
    manifest: {
      ...manifest,
      parameterCount,
      trainedEpochs: trainingResult.completedEpochs,
      trainingSequenceCount: dataset.sampleCount,
      savedAt,
    },
  });

  postMessage('done', {
    trainingResult,
    parameterCount,
    tokenizerVocabularySize: tokenizer.vocabularySize,
    languageModel: {
      kind: 'tfjs-neural-lm',
      vocabularySize: tokenizer.vocabularySize,
      parameterCount,
      tokenizerReady: true,
      checkpointReady: true,
      lastSavedAt: savedAt,
      trainingSequenceCount: dataset.sampleCount,
      corpusTokenCount: tokenizerTokenCount,
      feedbackExampleCount: positiveFeedbackCount,
    },
  });

  disposeRuntime(runtime);
})().catch((error) => {
  postMessage('error', {
    error: {
      message: error.message || 'Training worker failed.',
      stack: error.stack || '',
    },
  });
  process.exitCode = 1;
});
