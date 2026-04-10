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
  splitDatasetSamples,
  trainRuntime,
} = require('./neuralModel');

let stopRequested = false;

function buildEffectiveSettings(settings) {
  const effectiveSettings = structuredClone(settings);
  const requestedBatchSize = Math.max(1, Number(settings.training.batchSize) || 1);
  effectiveSettings.training.batchSize = requestedBatchSize;

  return {
    effectiveSettings,
    requestedBatchSize,
    effectiveBatchSize: effectiveSettings.training.batchSize,
  };
}

function deriveBatchPlan(trainingSampleCount, requestedBatchSize) {
  const safeSampleCount = Math.max(Number(trainingSampleCount) || 0, 0);
  const cappedRequestedBatchSize = safeSampleCount
    ? Math.min(requestedBatchSize, safeSampleCount)
    : requestedBatchSize;

  const minimumBatchesPerEpoch = safeSampleCount >= 16
    ? 4
    : safeSampleCount >= 4
      ? 2
      : 1;

  const maxBatchSizeForMinimumBatches = minimumBatchesPerEpoch > 1
    ? Math.max(Math.floor(safeSampleCount / minimumBatchesPerEpoch), 1)
    : Math.max(cappedRequestedBatchSize, 1);

  const effectiveBatchSize = Math.max(
    1,
    Math.min(cappedRequestedBatchSize, maxBatchSizeForMinimumBatches)
  );

  return {
    minimumBatchesPerEpoch,
    effectiveBatchSize,
    adjusted: effectiveBatchSize !== requestedBatchSize,
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
    resumeEpochOffset = 0,
    manifest,
    tokenizerTokenCount,
    positiveFeedbackCount,
    stopSignalBuffer = null,
  } = workerData;
  const stopSignalView = stopSignalBuffer ? new Int32Array(stopSignalBuffer) : null;

  const backendInfo = await ensureBackend(settings);
  const { effectiveSettings, requestedBatchSize } = buildEffectiveSettings(settings);

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
  const { trainingSamples, validationSamples } = splitDatasetSamples(dataset.samples);
  const batchPlan = deriveBatchPlan(trainingSamples.length, requestedBatchSize);
  effectiveSettings.training.batchSize = batchPlan.effectiveBatchSize;
  const batchesPerEpoch = Math.max(
    Math.ceil(trainingSamples.length / Math.max(1, effectiveSettings.training.batchSize)),
    1
  );

  postMessage('prepared', {
    datasetSampleCount: dataset.sampleCount,
    trainingSampleCount: trainingSamples.length,
    validationSampleCount: validationSamples.length,
    batchesPerEpoch,
    tokenizerVocabularySize: tokenizer.vocabularySize,
    parameterCount,
    backendName: backendInfo.backendName,
    backendLabel: backendInfo.label,
    executionMode: backendInfo.executionMode,
    backendWarning: backendInfo.warning,
    requestedBatchSize,
    effectiveBatchSize: batchPlan.effectiveBatchSize,
    batchSizeAdjusted: batchPlan.adjusted,
    minimumBatchesPerEpoch: batchPlan.minimumBatchesPerEpoch,
  });

  const trainingResult = await trainRuntime({
    runtime,
    dataset,
    settings: effectiveSettings,
    epochOffset: Math.max(Number(resumeEpochOffset) || 0, 0),
    onBatchEnd: async (payload) => {
      postMessage('batch', payload);
    },
    shouldStop: () => (
      stopRequested ||
      (stopSignalView ? Atomics.load(stopSignalView, 0) === 1 : false)
    ),
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
      validationLoss: trainingResult.validationLoss,
      bestValidationLoss: trainingResult.bestValidationLoss,
      trainingSampleCount: trainingResult.trainingSampleCount,
      validationSampleCount: trainingResult.validationSampleCount,
    },
  });

  disposeRuntime(runtime);
})().catch((error) => {
  postMessage('error', {
    error: {
      message: error.message || 'Ошибка воркера обучения.',
      stack: error.stack || '',
    },
  });
  process.exitCode = 1;
});
