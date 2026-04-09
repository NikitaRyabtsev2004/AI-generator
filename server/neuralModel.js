const fs = require('fs/promises');
const path = require('path');
const tf = require('@tensorflow/tfjs');
const { cleanText } = require('./text');

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

let backendReadyPromise = null;
let bannerSuppressed = false;
let originalWarn = null;
let originalLog = null;
let originalInfo = null;

function suppressNodeBanner() {
  if (bannerSuppressed) {
    return;
  }

  originalWarn = console.warn;
  originalLog = console.log;
  originalInfo = console.info;
  const shouldSuppress = (args) => args.some((value) => (
    typeof value === 'string' &&
    value.includes('To speed things up dramatically, install our node backend')
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

async function ensureBackend() {
  if (!backendReadyPromise) {
    backendReadyPromise = (async () => {
      suppressNodeBanner();
      await tf.setBackend('cpu');
      await tf.ready();
      tf.enableProdMode();
      return tf.getBackend();
    })();
  }

  return backendReadyPromise;
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
    activation: 'softmax',
    name: 'token_logits',
  }).apply(current);

  const model = tf.model({
    inputs: input,
    outputs: output,
    name: 'liquid_glass_neural_lm',
  });

  model.compile({
    optimizer: tf.train.adam(modelSettings.learningRate),
    loss: 'sparseCategoricalCrossentropy',
  });

  return model;
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

  try {
    const [tokenizerRaw, specsRaw, weightsRaw] = await Promise.all([
      fs.readFile(layout.tokenizerPath, 'utf8'),
      fs.readFile(layout.neuralSpecPath, 'utf8'),
      fs.readFile(layout.neuralWeightsPath),
    ]);

    const tokenizer = JSON.parse(tokenizerRaw);
    const specBundle = JSON.parse(specsRaw);
    const manifest = specBundle.manifest || {};
    const modelSettings = manifest.modelSettings || settings.training;
    const model = createModel({
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
    const orderedWeights = model.weights.map((weight) => namedWeights[weight.name]);
    model.setWeights(orderedWeights);

    return {
      model,
      tokenizer,
      manifest,
      parameterCount: countParameters(model),
    };
  } catch (error) {
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

async function trainRuntime({
  runtime,
  dataset,
  settings,
  epochOffset = 0,
  onBatchEnd,
  shouldStop,
}) {
  await ensureBackend();

  const samples = [...dataset.samples];
  const totalEpochs = settings.training.epochs;
  const batchSize = Math.max(1, settings.training.batchSize);
  const sequenceLength = settings.training.sequenceLength;
  const history = [];
  let lastLoss = null;
  let totalLoss = 0;
  let processedBatches = 0;
  let completedEpochs = epochOffset;
  let stopRequested = false;

  for (let epoch = 1; epoch <= totalEpochs; epoch += 1) {
    shuffleInPlace(samples);

    for (let batchStart = 0; batchStart < samples.length; batchStart += batchSize) {
      if (typeof shouldStop === 'function' && shouldStop()) {
        stopRequested = true;
        break;
      }

      const batchSamples = samples.slice(batchStart, batchStart + batchSize);
      const xValues = [];
      const yValues = [];

      batchSamples.forEach((sample) => {
        xValues.push(...sample.x);
        yValues.push(...sample.y);
      });

      const xBatch = tf.tensor2d(
        xValues,
        [batchSamples.length, sequenceLength],
        'int32'
      );
      const yBatch = tf.tensor2d(
        yValues,
        [batchSamples.length, sequenceLength],
        'int32'
      );

      const batchLoss = await runtime.model.trainOnBatch(xBatch, yBatch);
      xBatch.dispose();
      yBatch.dispose();

      lastLoss = Number(normalizeLossValue(batchLoss).toFixed(4));
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
          batchesThisEpoch: Math.ceil(samples.length / batchSize),
          historyEntry,
        });
      }
    }

    if (stopRequested) {
      break;
    }

    completedEpochs = epochOffset + epoch;
  }

  return {
    lastLoss,
    averageLoss: processedBatches ? Number((totalLoss / processedBatches).toFixed(4)) : null,
    perplexity: processedBatches
      ? Number(Math.exp(totalLoss / processedBatches).toFixed(4))
      : null,
    processedBatches,
    completedEpochs,
    stopRequested,
    history,
  };
}

function applySamplingControls(probabilities, generatedTokens, bannedTokenIds, settings) {
  const nextProbabilities = probabilities.map((value, index) => {
    if (bannedTokenIds.has(index)) {
      return 0;
    }

    let nextValue = value;
    if (generatedTokens.includes(index)) {
      nextValue /= settings.generation.repetitionPenalty;
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

async function generateText({
  runtime,
  promptText,
  settings,
}) {
  await ensureBackend();

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
      const adjusted = tf.pow(reshaped, tf.scalar(1 / temperature));
      const normalized = adjusted.div(adjusted.sum());
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

    generatedTokenIds.push(nextTokenId);
  }

  return {
    text: decodeTokenIds(generatedTokenIds, tokenizer),
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
  tokenizeForModel,
  trainRuntime,
};
