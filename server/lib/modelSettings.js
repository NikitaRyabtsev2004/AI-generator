function normalizeExecutionMode(input) {
  const candidate = typeof input === 'string'
    ? input
    : input?.training?.executionMode;

  return ['compatibility', 'native_preferred', 'gpu_preferred'].includes(candidate)
    ? candidate
    : 'compatibility';
}

function resolveTrainingSettings(settingsLike = {}) {
  const training = settingsLike?.training || settingsLike;
  const embeddingSize = Math.max(32, Number(training.embeddingSize) || 128);
  const requestedHeads = Math.max(1, Number(training.attentionHeads) || 4);
  let attentionHeads = requestedHeads;

  while (attentionHeads > 1 && embeddingSize % attentionHeads !== 0) {
    attentionHeads -= 1;
  }

  return {
    executionMode: normalizeExecutionMode(training.executionMode),
    sequenceLength: Math.max(32, Number(training.sequenceLength) || 128),
    embeddingSize,
    attentionHeads,
    transformerLayers: Math.max(1, Number(training.transformerLayers || training.recurrentLayers) || 4),
    feedForwardSize: Math.max(
      embeddingSize,
      Number(training.feedForwardSize || training.hiddenSize) || embeddingSize * 4
    ),
    dropout: Math.min(0.6, Math.max(0, Number(training.dropout) || 0.1)),
    learningRate: Math.max(0.00001, Number(training.learningRate) || 0.001),
    batchSize: Math.max(1, Number(training.batchSize) || 8),
    epochs: Math.max(1, Number(training.epochs) || 1),
    vocabularyLimit: Math.max(256, Number(training.vocabularyLimit) || 8000),
  };
}

module.exports = {
  normalizeExecutionMode,
  resolveTrainingSettings,
};
