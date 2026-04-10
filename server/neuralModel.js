const fs = require('fs/promises');
const fsSync = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');
const { cleanText, tokenizeWords } = require('./text');

const SPECIAL_TOKENS = Object.freeze([
  '__pad__',
  '__unk__',
  '__bos__',
  '__eos__',
  '__sep__',
  '__usr__',
  '__asst__',
  '__ctx__',
]);

const SPECIAL_TOKEN_SET = new Set(SPECIAL_TOKENS);
const NO_SPACE_BEFORE = new Set(['.', ',', '!', '?', ';', ':', ')', ']', '}', '%']);
const NO_SPACE_AFTER = new Set(['(', '[', '{']);

const SUPPORTED_EXECUTION_MODES = new Set([
  'compatibility',
  'native_preferred',
  'gpu_preferred',
]);

let backendReadyState = {
  key: null,
  promise: null,
  info: null,
};
let bannerSuppressed = false;
let originalWarn = null;
let originalLog = null;
let originalInfo = null;
let nodeBackendPackages = null;

function suppressNodeBanner() {
  if (bannerSuppressed) {
    return;
  }

  if (!process.env.TF_CPP_MIN_LOG_LEVEL) {
    process.env.TF_CPP_MIN_LOG_LEVEL = '3';
  }

  originalWarn = console.warn;
  originalLog = console.log;
  originalInfo = console.info;
  const suppressedPatterns = [
    'To speed things up dramatically, install our node backend',
    "The kernel '",
    "for backend 'tensorflow' is already registered",
    'tensorflow backend was already registered',
  ];
  const shouldSuppress = (args) => args.some((value) => (
    typeof value === 'string' &&
    suppressedPatterns.some((pattern) => value.includes(pattern))
  ));

  console.warn = (...args) => {
    if (shouldSuppress(args)) {
      return;
    }

    originalWarn(...args);
  };

  console.log = (...args) => {
    if (shouldSuppress(args)) {
      return;
    }

    originalLog(...args);
  };

  console.info = (...args) => {
    if (shouldSuppress(args)) {
      return;
    }

    originalInfo(...args);
  };
  bannerSuppressed = true;
}

function normalizeExecutionMode(input) {
  const candidate = typeof input === 'string'
    ? input
    : input?.training?.executionMode;

  return SUPPORTED_EXECUTION_MODES.has(candidate)
    ? candidate
    : 'compatibility';
}

function extractBackendErrorMessage(error) {
  const rawMessage = String(error?.message || '').trim();
  if (!rawMessage) {
    return 'Не удалось загрузить native backend.';
  }

  return cleanText(rawMessage.split(/\r?\n/u)[0]);
}

function readNapiDirectories(libDir) {
  if (!fsSync.existsSync(libDir)) {
    return [];
  }

  return fsSync.readdirSync(libDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith('napi-v'))
    .map((entry) => path.join(libDir, entry.name))
    .sort();
}

function repairTensorflowDllPlacement(packageName, packageDir) {
  if (process.platform !== 'win32' || !packageDir) {
    return {
      repaired: false,
      note: '',
    };
  }

  const libDir = path.join(packageDir, 'lib');
  const napiDirs = readNapiDirectories(libDir);
  if (!napiDirs.length) {
    return {
      repaired: false,
      note: '',
    };
  }

  const donorDllPath = napiDirs
    .map((directory) => path.join(directory, 'tensorflow.dll'))
    .find((candidate) => fsSync.existsSync(candidate));

  if (!donorDllPath) {
    return {
      repaired: false,
      note: '',
    };
  }

  const repairedTargets = [];
  napiDirs.forEach((directory) => {
    const bindingPath = path.join(directory, 'tfjs_binding.node');
    const dllPath = path.join(directory, 'tensorflow.dll');

    if (!fsSync.existsSync(bindingPath) || fsSync.existsSync(dllPath)) {
      return;
    }

    fsSync.copyFileSync(donorDllPath, dllPath);
    repairedTargets.push(path.basename(directory));
  });

  return repairedTargets.length
    ? {
      repaired: true,
      note: `Сервер выровнял размещение tensorflow.dll для ${packageName} (${repairedTargets.join(', ')}).`,
    }
    : {
      repaired: false,
      note: '',
    };
}

function inspectNodeBackendPackage(packageName) {
  const result = {
    packageName,
    installed: false,
    loadable: false,
    errorCode: '',
    errorMessage: '',
    repairNote: '',
  };

  try {
    const packageJsonPath = require.resolve(`${packageName}/package.json`);
    result.installed = true;
    const packageDir = path.dirname(packageJsonPath);
    const repairInfo = repairTensorflowDllPlacement(packageName, packageDir);
    result.repairNote = repairInfo.note;
    require(packageName);
    result.loadable = true;
  } catch (error) {
    if (error?.code === 'MODULE_NOT_FOUND' && !String(error?.message || '').includes(packageName)) {
      result.installed = true;
    }

    if (!result.installed && error?.code !== 'MODULE_NOT_FOUND') {
      result.installed = true;
    }

    result.errorCode = error?.code || '';
    result.errorMessage = extractBackendErrorMessage(error);
  }

  return result;
}

function getNodeBackendPackageName(kind) {
  return kind === 'gpu'
    ? '@tensorflow/tfjs-node-gpu'
    : '@tensorflow/tfjs-node';
}

async function getNodeBackendPackageProbe(kind) {
  if (!nodeBackendPackages) {
    nodeBackendPackages = {};
  }

  if (!nodeBackendPackages[kind]) {
    nodeBackendPackages[kind] = inspectNodeBackendPackage(getNodeBackendPackageName(kind));
  }

  return nodeBackendPackages[kind];
}

function detectTensorflowDeviceMode() {
  const backend = tf.engine().backendInstance;
  const gpuFlag = typeof backend?.isUsingGpuDevice === 'function'
    ? backend.isUsingGpuDevice()
    : backend?.binding && typeof backend.binding.isUsingGpuDevice === 'function'
      ? backend.binding.isUsingGpuDevice()
      : backend?.isUsingGpuDevice;

  return {
    usingGpu: Boolean(gpuFlag),
    isGpuPackage: Boolean(backend?.isGPUPackage),
  };
}

function formatBackendLabel({ usingGpu, isGpuPackage }) {
  if (usingGpu) {
    return 'TensorFlow backend (GPU)';
  }

  return isGpuPackage
    ? 'TensorFlow backend (native CPU)'
    : 'TensorFlow native backend';
}

function buildMissingPackageWarning(packageName, fallbackLabel) {
  return `Выбран ускоренный режим, но пакет \`${packageName}\` не установлен. Используется ${fallbackLabel}.`;
}

function buildLoadFailureWarning(probe, fallbackLabel) {
  const repairSuffix = probe.repairNote ? ` ${probe.repairNote}` : '';
  return `Пакет \`${probe.packageName}\` найден, но native binding не загрузился: ${probe.errorMessage}.${repairSuffix} Используется ${fallbackLabel}.`;
}

function buildGpuUnavailableWarning(probe) {
  const repairSuffix = probe.repairNote ? ` ${probe.repairNote}` : '';
  return `GPU-режим выбран, но TensorFlow не активировал видеокарту.${repairSuffix} Для \`@tensorflow/tfjs-node-gpu\` на Windows обычно нужны CUDA 11.x и cuDNN 8.x, а DLL вроде \`cudart64_110.dll\`, \`cudnn64_8.dll\`, \`cusolver64_11.dll\` и \`cublas64_11.dll\` должны быть доступны в PATH.`;
}

function buildBackendRuntimeFailureWarning(probe, fallbackLabel, errorMessage) {
  const repairSuffix = probe.repairNote ? ` ${probe.repairNote}` : '';
  const compatibilityHint = errorMessage.includes('isNullOrUndefined')
    ? ' Похоже на несовместимость `tfjs-node(-gpu)` с текущей версией Node.js.'
    : '';
  return `TensorFlow backend загрузился, но базовая операция завершилась ошибкой: ${errorMessage}.${compatibilityHint}${repairSuffix} Используется ${fallbackLabel}.`;
}

async function verifyTensorflowBackendExecution() {
  const input = tf.tensor1d([1, 2, 3], 'float32');
  let slice = null;

  try {
    slice = tf.slice(input, [0], [1]);
    await slice.data();
    return {
      ok: true,
      errorMessage: '',
    };
  } catch (error) {
    return {
      ok: false,
      errorMessage: extractBackendErrorMessage(error),
    };
  } finally {
    slice?.dispose();
    input.dispose();
  }
}

async function selectBackend(executionMode) {
  const normalizedMode = normalizeExecutionMode(executionMode);
  suppressNodeBanner();

  if (normalizedMode === 'compatibility') {
    await tf.setBackend('cpu');
    return {
      backendName: 'cpu',
      executionMode: normalizedMode,
      label: 'CPU (совместимый режим)',
      warning: '',
    };
  }

  const prefersGpu = normalizedMode === 'gpu_preferred';
  const probeKind = prefersGpu ? 'gpu' : 'native';
  const packageProbe = await getNodeBackendPackageProbe(probeKind);
  let runtimeFailure = '';

  if (packageProbe.loadable) {
    try {
      await tf.setBackend('tensorflow');
      await tf.ready();
      const tensorflowMode = detectTensorflowDeviceMode();
      // const executionProbe = await verifyTensorflowBackendExecution();
      // if (!executionProbe.ok) {
      //   runtimeFailure = executionProbe.errorMessage;
      //   throw new Error(executionProbe.errorMessage);
      // }
      return {
        backendName: 'tensorflow',
        executionMode: normalizedMode,
        label: formatBackendLabel(tensorflowMode),
        warning: prefersGpu && !tensorflowMode.usingGpu
          ? buildGpuUnavailableWarning(packageProbe)
          : '',
      };
    } catch (error) {
      runtimeFailure ||= extractBackendErrorMessage(error);
    }
  }

  await tf.setBackend('cpu');
  return {
    backendName: 'cpu',
    executionMode: normalizedMode,
    label: 'CPU (fallback)',
    warning: runtimeFailure
      ? buildBackendRuntimeFailureWarning(packageProbe, 'CPU', runtimeFailure)
      : packageProbe.installed
        ? buildLoadFailureWarning(packageProbe, 'CPU')
        : buildMissingPackageWarning(getNodeBackendPackageName(probeKind), 'CPU'),
  };
}

async function ensureBackend(executionMode = 'compatibility') {
  const normalizedMode = normalizeExecutionMode(executionMode);
  if (!backendReadyState.promise || backendReadyState.key !== normalizedMode) {
    backendReadyState.key = normalizedMode;
    backendReadyState.promise = (async () => {
      const info = await selectBackend(normalizedMode);
      await tf.ready();
      tf.enableProdMode();
      backendReadyState.info = info;
      return info;
    })();
  }

  return backendReadyState.promise;
}

function normalizeModelText(input = '') {
  return cleanText(input)
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenizeForModel(input = '') {
  const normalized = normalizeModelText(input).toLowerCase();
  const matches = normalized.match(
    /__pad__|__unk__|__bos__|__eos__|__sep__|__usr__|__asst__|__ctx__|[\p{L}\p{N}]+|[.,!?;:%()[\]{}"'-]/gu
  );

  return matches ? matches.filter(Boolean) : [];
}

function buildTokenizer(textEntries, vocabularyLimit) {
  const frequency = {};

  textEntries.forEach((entry) => {
    tokenizeForModel(entry).forEach((token) => {
      if (SPECIAL_TOKEN_SET.has(token)) {
        return;
      }
      frequency[token] = (frequency[token] || 0) + 1;
    });
  });

  const availableVocabulary = Math.max(Number(vocabularyLimit) || 0, SPECIAL_TOKENS.length + 32);
  const learnedTokens = Object.entries(frequency)
    .sort((left, right) => right[1] - left[1])
    .slice(0, availableVocabulary - SPECIAL_TOKENS.length)
    .map(([token]) => token);

  const idToToken = [...SPECIAL_TOKENS, ...learnedTokens];
  const tokenToId = idToToken.reduce((result, token, index) => {
    result[token] = index;
    return result;
  }, {});

  return {
    version: 1,
    specialTokens: SPECIAL_TOKENS,
    idToToken,
    tokenToId,
    vocabularySize: idToToken.length,
  };
}

function encodeTokens(tokens, tokenizer) {
  const unknownId = tokenizer.tokenToId.__unk__;
  return tokens.map((token) => tokenizer.tokenToId[token] ?? unknownId);
}

function encodeText(text, tokenizer) {
  return encodeTokens(tokenizeForModel(text), tokenizer);
}

function decodeTokenIds(tokenIds, tokenizer) {
  const tokens = tokenIds
    .map((tokenId) => tokenizer.idToToken[tokenId] || '')
    .filter((token) => token && !SPECIAL_TOKEN_SET.has(token));

  if (!tokens.length) {
    return '';
  }

  let output = '';

  tokens.forEach((token, index) => {
    const previousToken = index > 0 ? tokens[index - 1] : '';
    const shouldSkipSpace =
      !output ||
      NO_SPACE_BEFORE.has(token) ||
      NO_SPACE_AFTER.has(previousToken) ||
      previousToken === '"' ||
      token === '"';

    output += shouldSkipSpace ? token : ` ${token}`;
  });

  const normalized = cleanText(
    output
      .replace(/\s+([.,!?;:%)\]}])/g, '$1')
      .replace(/([([{])\s+/g, '$1')
      .replace(/\s+"/g, ' "')
      .replace(/"\s+/g, '" ')
  );

  if (!normalized) {
    return '';
  }

  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function createSlidingWindows(tokenIds, sequenceLength) {
  const normalizedLength = Math.max(16, Number(sequenceLength) || 16);
  const stride = Math.max(8, Math.floor(normalizedLength / 2));
  const minimumSourceLength = Math.max(4, Math.floor(normalizedLength / 3));
  const padId = 0;
  const samples = [];

  if (!Array.isArray(tokenIds) || tokenIds.length < minimumSourceLength) {
    return samples;
  }

  for (let start = 0; start < tokenIds.length - 1; start += stride) {
    const window = tokenIds.slice(start, start + normalizedLength + 1);
    if (window.length < minimumSourceLength) {
      continue;
    }

    const x = window.slice(0, -1);
    const y = window.slice(1);

    while (x.length < normalizedLength) {
      x.push(padId);
      y.push(padId);
    }

    samples.push({ x, y });

    if (start + normalizedLength >= tokenIds.length - 1) {
      break;
    }
  }

  return samples;
}

function countParameters(model) {
  return model.weights.reduce((sum, weight) => {
    const weightSize = weight.shape.reduce((product, size) => product * size, 1);
    return sum + weightSize;
  }, 0);
}

function createModel({ vocabularySize, settings }) {
  const modelSettings = settings.training;
  const input = tf.input({
    shape: [modelSettings.sequenceLength],
    dtype: 'int32',
    name: 'token_input',
  });

  let current = tf.layers.embedding({
    inputDim: vocabularySize,
    outputDim: modelSettings.embeddingSize,
    name: 'token_embedding',
  }).apply(input);

  for (let layerIndex = 0; layerIndex < modelSettings.recurrentLayers; layerIndex += 1) {
    current = tf.layers.gru({
      units: modelSettings.hiddenSize,
      returnSequences: true,
      dropout: modelSettings.dropout,
      recurrentInitializer: 'glorotUniform',
      name: `gru_${layerIndex + 1}`,
    }).apply(current);
  }

  current = tf.layers.dense({
    units: modelSettings.hiddenSize,
    activation: 'relu',
    name: 'projection_dense',
  }).apply(current);

  current = tf.layers.dropout({
    rate: modelSettings.dropout,
    name: 'projection_dropout',
  }).apply(current);

  const output = tf.layers.dense({
    units: vocabularySize,
    name: 'token_logits',
  }).apply(current);

  return tf.model({
    inputs: input,
    outputs: output,
    name: 'liquid_glass_neural_lm',
  });
}

function normalizeLossValue(lossValue) {
  if (Array.isArray(lossValue)) {
    return Number(lossValue[0]);
  }
  if (typeof lossValue === 'number') {
    return Number(lossValue);
  }
  return Number(lossValue?.dataSync?.()[0] || 0);
}

function createArtifactLayout(storage) {
  return {
    neuralModelDir: storage.neuralModelDir,
    tokenizerPath: storage.tokenizerPath,
    neuralWeightsPath: storage.neuralWeightsPath,
    neuralSpecPath: storage.neuralSpecPath,
  };
}

async function saveRuntime({
  model,
  tokenizer,
  storage,
  manifest,
}) {
  const layout = createArtifactLayout(storage);
  await fs.mkdir(layout.neuralModelDir, { recursive: true });

  const weightMap = {};
  const modelWeights = model.getWeights();
  model.weights.forEach((weight, index) => {
    weightMap[weight.name] = modelWeights[index];
  });

  const encoded = await tf.io.encodeWeights(weightMap);
  const weightsBuffer = Buffer.from(encoded.data);

  await fs.writeFile(layout.tokenizerPath, JSON.stringify(tokenizer, null, 2), 'utf8');
  await fs.writeFile(layout.neuralWeightsPath, weightsBuffer);
  await fs.writeFile(
    layout.neuralSpecPath,
    JSON.stringify(
      {
        version: 1,
        specs: encoded.specs,
        manifest,
      },
      null,
      2
    ),
    'utf8'
  );
}

async function loadRuntime({ storage, settings }) {
  const layout = createArtifactLayout(storage);
  let model = null;

  try {
    await ensureBackend(settings);
    const [tokenizerRaw, specsRaw, weightsRaw] = await Promise.all([
      fs.readFile(layout.tokenizerPath, 'utf8'),
      fs.readFile(layout.neuralSpecPath, 'utf8'),
      fs.readFile(layout.neuralWeightsPath),
    ]);

    const tokenizer = JSON.parse(tokenizerRaw);
    const specBundle = JSON.parse(specsRaw);
    const manifest = specBundle.manifest || {};
    const modelSettings = manifest.modelSettings || settings.training;
    model = createModel({
      vocabularySize: tokenizer.idToToken.length,
      settings: {
        ...settings,
        training: {
          ...settings.training,
          ...modelSettings,
        },
      },
    });

    const arrayBuffer = weightsRaw.buffer.slice(
      weightsRaw.byteOffset,
      weightsRaw.byteOffset + weightsRaw.byteLength
    );
    const namedWeights = tf.io.decodeWeights(arrayBuffer, specBundle.specs);
    const normalizeWeightName = (name) => name.replace(/\/([^/]+?)_\d+$/u, '/$1');
    const orderedWeights = model.weights.map((weight) => (
      namedWeights[weight.name] || namedWeights[normalizeWeightName(weight.name)]
    ));
    if (orderedWeights.some((tensor) => !tensor)) {
      throw new Error('Не удалось сопоставить веса модели с сохраненным чекпоинтом.');
    }
    model.setWeights(orderedWeights);

    return {
      model,
      tokenizer,
      manifest,
      parameterCount: countParameters(model),
    };
  } catch (error) {
    model?.dispose?.();
    return null;
  }
}

function disposeRuntime(runtime) {
  if (!runtime?.model) {
    return;
  }

  runtime.model.dispose();
}

function createDataset({ texts, tokenizer, sequenceLength }) {
  const samples = [];
  let tokenCount = 0;

  texts.forEach((text) => {
    const encoded = encodeText(text, tokenizer);
    tokenCount += encoded.length;
    createSlidingWindows(encoded, sequenceLength).forEach((sample) => {
      samples.push(sample);
    });
  });

  return {
    samples,
    sampleCount: samples.length,
    tokenCount,
  };
}

function shuffleInPlace(items) {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [items[index], items[swapIndex]] = [items[swapIndex], items[index]];
  }
  return items;
}

function splitDatasetSamples(samples) {
  const shuffled = shuffleInPlace([...samples]);
  if (shuffled.length < 12) {
    return {
      trainingSamples: shuffled,
      validationSamples: [],
    };
  }

  const validationCount = Math.min(
    Math.max(Math.floor(shuffled.length * 0.12), 1),
    Math.max(shuffled.length - 4, 0)
  );

  return {
    trainingSamples: shuffled.slice(0, shuffled.length - validationCount),
    validationSamples: shuffled.slice(shuffled.length - validationCount),
  };
}

function deriveMinimumOptimizationSteps({ dataset, settings, batchSize }) {
  const requestedSteps = Math.max(
    Math.ceil(dataset.sampleCount / Math.max(batchSize, 1)),
    1
  ) * Math.max(Number(settings.training.epochs) || 1, 1);

  return {
    minimumSteps: requestedSteps,
    requestedSteps,
    enforced: false,
  };
}

function createBatchTensors(batchSamples, sequenceLength) {
  const xValues = [];
  const yValues = [];

  batchSamples.forEach((sample) => {
    xValues.push(...sample.x);
    yValues.push(...sample.y);
  });

  return {
    xBatch: tf.tensor2d(xValues, [batchSamples.length, sequenceLength], 'int32'),
    yBatch: tf.tensor2d(yValues, [batchSamples.length, sequenceLength], 'int32'),
  };
}

function computeMaskedLoss(logits, labels, padId = 0) {
  return tf.tidy(() => {
    const vocabularySize = logits.shape[logits.shape.length - 1];
    const flatLogits = logits.reshape([-1, vocabularySize]);
    const flatLabels = labels.reshape([-1]);
    const safeLabels = flatLabels.maximum(0).toInt();
    const mask = safeLabels.notEqual(tf.scalar(padId, 'int32')).cast('float32');
    const oneHotLabels = tf.oneHot(safeLabels, vocabularySize).toFloat();
    const logProbabilities = tf.logSoftmax(flatLogits);
    const tokenLosses = tf.neg(oneHotLabels.mul(logProbabilities).sum(-1));
    const maskedLosses = tokenLosses.mul(mask);
    const normalization = mask.sum().maximum(tf.scalar(1));
    return maskedLosses.sum().div(normalization);
  });
}

async function trainBatch({ runtime, optimizer, xBatch, yBatch, clipNorm = 1 }) {
  const trainableVariables = runtime.model.trainableWeights.map((weight) => weight.val);

  const { value: lossTensor, grads } = tf.variableGrads(() => {
    const logits = runtime.model.apply(xBatch, { training: true });
    return computeMaskedLoss(logits, yBatch);
  }, trainableVariables);

  const clippedEntries = Object.entries(grads).map(([name, gradTensor]) => {
    const clipped = clipNorm > 0
      ? tf.tidy(() => {
        const gradientNorm = gradTensor.norm();
        const scale = tf.minimum(
          tf.scalar(1),
          tf.scalar(clipNorm).div(gradientNorm.add(tf.scalar(1e-6)))
        );
        return gradTensor.mul(scale);
      })
      : gradTensor;
    if (clipped !== gradTensor) {
      gradTensor.dispose();
    }
    return [name, clipped];
  });
  const clippedGradients = Object.fromEntries(clippedEntries);

  optimizer.applyGradients(clippedGradients);
  Object.values(clippedGradients).forEach((tensor) => tensor.dispose());

  const lossValue = Number((await lossTensor.data())[0] || 0);
  lossTensor.dispose();
  return lossValue;
}

async function evaluateDataset({ runtime, samples, sequenceLength, batchSize }) {
  if (!samples.length) {
    return null;
  }

  let totalLoss = 0;
  let processedBatches = 0;

  for (let batchStart = 0; batchStart < samples.length; batchStart += batchSize) {
    const batchSamples = samples.slice(batchStart, batchStart + batchSize);
    const { xBatch, yBatch } = createBatchTensors(batchSamples, sequenceLength);
    const batchLossTensor = tf.tidy(() => {
      const logits = runtime.model.predict(xBatch);
      return computeMaskedLoss(logits, yBatch);
    });

    totalLoss += Number((await batchLossTensor.data())[0] || 0);
    processedBatches += 1;
    batchLossTensor.dispose();
    xBatch.dispose();
    yBatch.dispose();
  }

  return Number((totalLoss / Math.max(processedBatches, 1)).toFixed(4));
}

async function trainRuntime({
  runtime,
  dataset,
  settings,
  epochOffset = 0,
  onBatchEnd,
  shouldStop,
}) {
  await ensureBackend(settings);

  const optimizer = tf.train.adam(settings.training.learningRate);
  const { trainingSamples, validationSamples } = splitDatasetSamples(dataset.samples);
  const batchSize = Math.max(1, settings.training.batchSize);
  const sequenceLength = settings.training.sequenceLength;
  const baseEpochs = Math.max(Number(settings.training.epochs) || 1, 1);
  const batchesPerEpoch = Math.max(Math.ceil(trainingSamples.length / batchSize), 1);
  const stepBudget = deriveMinimumOptimizationSteps({
    dataset,
    settings,
    batchSize,
  });
  const totalEpochs = baseEpochs;
  const history = [];
  let lastLoss = null;
  let totalLoss = 0;
  let processedBatches = 0;
  let completedEpochs = epochOffset;
  let stopRequested = false;
  let lastValidationLoss = null;
  let bestValidationLoss = null;
  let bestEpoch = null;
  let bestWeights = null;

  const effectiveTrainingSamples = trainingSamples.length ? trainingSamples : [...dataset.samples];

  for (let epoch = 1; epoch <= totalEpochs; epoch += 1) {
    shuffleInPlace(effectiveTrainingSamples);

    for (let batchStart = 0; batchStart < effectiveTrainingSamples.length; batchStart += batchSize) {
      if (typeof shouldStop === 'function' && shouldStop()) {
        stopRequested = true;
        break;
      }

      const batchSamples = effectiveTrainingSamples.slice(batchStart, batchStart + batchSize);
      const { xBatch, yBatch } = createBatchTensors(batchSamples, sequenceLength);
      const batchLoss = await trainBatch({
        runtime,
        optimizer,
        xBatch,
        yBatch,
        clipNorm: 1,
      });
      xBatch.dispose();
      yBatch.dispose();

      lastLoss = Number(batchLoss.toFixed(4));
      totalLoss += lastLoss;
      processedBatches += 1;

      const historyEntry = {
        step: processedBatches,
        epoch: epochOffset + epoch,
        batch: Math.floor(batchStart / batchSize) + 1,
        loss: lastLoss,
      };
      history.push(historyEntry);

      if (typeof onBatchEnd === 'function') {
        await onBatchEnd({
          lastLoss,
          averageLoss: Number((totalLoss / processedBatches).toFixed(4)),
          processedBatches,
          epoch: epochOffset + epoch,
          batch: historyEntry.batch,
          batchesThisEpoch: Math.ceil(effectiveTrainingSamples.length / batchSize),
          effectiveEpochs: totalEpochs,
          requestedEpochs: baseEpochs,
          minimumTrainingSteps: stepBudget.minimumSteps,
          enforcedStepBudget: stepBudget.enforced,
          historyEntry,
        });
      }
    }

    if (stopRequested) {
      break;
    }

    lastValidationLoss = await evaluateDataset({
      runtime,
      samples: validationSamples,
      sequenceLength,
      batchSize,
    });

    if (lastValidationLoss !== null && (bestValidationLoss === null || lastValidationLoss < bestValidationLoss)) {
      bestValidationLoss = lastValidationLoss;
      bestEpoch = epochOffset + epoch;
      bestWeights?.forEach((tensor) => tensor.dispose());
      bestWeights = runtime.model.getWeights().map((weight) => weight.clone());
    }

    completedEpochs = epochOffset + epoch;
  }

  if (bestWeights?.length) {
    runtime.model.setWeights(bestWeights);
    bestWeights.forEach((tensor) => tensor.dispose());
  }

  optimizer.dispose?.();

  return {
    lastLoss,
    averageLoss: processedBatches ? Number((totalLoss / processedBatches).toFixed(4)) : null,
    perplexity: processedBatches
      ? Number(Math.exp(totalLoss / processedBatches).toFixed(4))
      : null,
    validationLoss: lastValidationLoss,
    bestValidationLoss,
    bestEpoch,
    processedBatches,
    completedEpochs,
    effectiveEpochs: epochOffset + totalEpochs,
    requestedEpochs: epochOffset + baseEpochs,
    minimumTrainingSteps: stepBudget.minimumSteps,
    enforcedStepBudget: stepBudget.enforced,
    trainingSampleCount: effectiveTrainingSamples.length,
    validationSampleCount: validationSamples.length,
    stopRequested,
    history,
  };
}

function applySamplingControls(probabilities, generatedTokens, bannedTokenIds, settings) {
  const tokenFrequency = generatedTokens.reduce((result, tokenId) => {
    result[tokenId] = (result[tokenId] || 0) + 1;
    return result;
  }, {});

  const nextProbabilities = probabilities.map((value, index) => {
    if (bannedTokenIds.has(index)) {
      return 0;
    }

    let nextValue = value;
    const repeats = tokenFrequency[index] || 0;
    if (repeats > 0) {
      nextValue /= Math.pow(settings.generation.repetitionPenalty, repeats);
    }

    return nextValue;
  });

  const topK = Math.max(1, settings.generation.topKSampling);
  const ranked = nextProbabilities
    .map((value, index) => ({ value, index }))
    .sort((left, right) => right.value - left.value)
    .slice(0, topK);

  const filtered = new Array(nextProbabilities.length).fill(0);
  let sum = 0;

  ranked.forEach((entry) => {
    const adjusted = Math.max(entry.value, 0);
    filtered[entry.index] = adjusted;
    sum += adjusted;
  });

  if (sum <= 0) {
    return filtered.map((_value, index) => (index === SPECIAL_TOKENS.indexOf('__eos__') ? 1 : 0));
  }

  return filtered.map((value) => value / sum);
}

function sampleFromDistribution(probabilities) {
  const randomValue = Math.random();
  let cumulative = 0;

  for (let index = 0; index < probabilities.length; index += 1) {
    cumulative += probabilities[index];
    if (randomValue <= cumulative) {
      return index;
    }
  }

  return probabilities.findIndex((value) => value > 0);
}

function isUsableGeneratedText(text) {
  const normalized = cleanText(text);
  const tokens = tokenizeWords(normalized);
  if (tokens.length < 3) {
    return false;
  }

  const uniqueTokenRatio = new Set(tokens).size / Math.max(tokens.length, 1);
  if (uniqueTokenRatio < 0.35) {
    return false;
  }

  if (/[?!.,]{4,}/u.test(normalized)) {
    return false;
  }

  return true;
}

async function generateText({
  runtime,
  promptText,
  settings,
}) {
  await ensureBackend(settings);

  if (!runtime?.model || !runtime?.tokenizer) {
    return {
      text: '',
      generatedTokenIds: [],
    };
  }

  const tokenizer = runtime.tokenizer;
  const sequenceLength = settings.training.sequenceLength;
  const temperature = Math.max(0.1, settings.generation.responseTemperature);
  const promptTokenIds = encodeText(promptText, tokenizer);
  const generatedTokenIds = [];
  const padId = tokenizer.tokenToId.__pad__;
  const eosId = tokenizer.tokenToId.__eos__;
  const bannedTokenIds = new Set([
    tokenizer.tokenToId.__pad__,
    tokenizer.tokenToId.__unk__,
    tokenizer.tokenToId.__bos__,
    tokenizer.tokenToId.__usr__,
    tokenizer.tokenToId.__asst__,
    tokenizer.tokenToId.__ctx__,
  ]);

  for (let step = 0; step < settings.generation.maxGeneratedTokens; step += 1) {
    const currentIds = [...promptTokenIds, ...generatedTokenIds].slice(-sequenceLength);
    while (currentIds.length < sequenceLength) {
      currentIds.unshift(padId);
    }

    const nextTokenProbabilities = tf.tidy(() => {
      const inputTensor = tf.tensor2d(currentIds, [1, sequenceLength], 'int32');
      const prediction = runtime.model.predict(inputTensor);
      const logits = prediction.slice([0, sequenceLength - 1, 0], [1, 1, runtime.tokenizer.idToToken.length]);
      const reshaped = logits.reshape([runtime.tokenizer.idToToken.length]);
      const scaled = reshaped.div(tf.scalar(temperature));
      const normalized = tf.softmax(scaled);
      return Array.from(normalized.dataSync());
    });

    const normalizedProbabilities = applySamplingControls(
      nextTokenProbabilities,
      generatedTokenIds,
      bannedTokenIds,
      settings
    );
    const nextTokenId = sampleFromDistribution(normalizedProbabilities);

    if (nextTokenId === -1 || nextTokenId === eosId) {
      break;
    }

    if (generatedTokenIds.length >= 4) {
      const tail = generatedTokenIds.slice(-4);
      if (tail.every((tokenId) => tokenId === nextTokenId)) {
        break;
      }
    }

    generatedTokenIds.push(nextTokenId);
  }

  const decodedText = decodeTokenIds(generatedTokenIds, tokenizer);

  return {
    text: isUsableGeneratedText(decodedText) ? decodedText : '',
    generatedTokenIds,
  };
}

module.exports = {
  SPECIAL_TOKENS,
  buildTokenizer,
  countParameters,
  createDataset,
  createModel,
  decodeTokenIds,
  disposeRuntime,
  encodeText,
  ensureBackend,
  generateText,
  loadRuntime,
  saveRuntime,
  splitDatasetSamples,
  tokenizeForModel,
  trainRuntime,
};
