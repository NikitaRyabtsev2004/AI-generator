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
  const datasetPrefetchRaw = Number(training.datasetPrefetchBatches);
  const datasetThreadpoolRaw = Number(training.datasetPrivateThreadpoolSize);
  const heartbeatIntervalRaw = Number(training.heartbeatIntervalSeconds);
  const recoveryCheckpointBatchesRaw = Number(training.recoveryCheckpointIntervalBatches);
  const recoveryCheckpointMinutesRaw = Number(training.recoveryCheckpointIntervalMinutes);
  const maxAutoRecoveryRestartsRaw = Number(training.maxAutoRecoveryRestarts);
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
    optimizer: String(training.optimizer || 'adam_legacy').toLowerCase() === 'adam' ? 'adam' : 'adam_legacy',
    gradientClipNorm: Math.min(10, Math.max(0, Number(training.gradientClipNorm) || 0)),
    batchSize: Math.max(1, Number(training.batchSize) || 8),
    epochs: Math.max(1, Number(training.epochs) || 1),
    batchReportInterval: Math.max(1, Number(training.batchReportInterval) || 1),
    metricsReportInterval: Math.max(1, Number(training.metricsReportInterval) || 16),
    historyReportInterval: Math.max(1, Number(training.historyReportInterval) || 16),
    datasetPrefetchBatches: Number.isFinite(datasetPrefetchRaw)
      ? Math.min(8, Math.max(0, Math.round(datasetPrefetchRaw)))
      : 1,
    datasetPrivateThreadpoolSize: Number.isFinite(datasetThreadpoolRaw)
      ? Math.min(32, Math.max(0, Math.round(datasetThreadpoolRaw)))
      : 0,
    heartbeatIntervalSeconds: Number.isFinite(heartbeatIntervalRaw)
      ? Math.min(60, Math.max(5, Math.round(heartbeatIntervalRaw)))
      : 10,
    recoveryCheckpointIntervalBatches: Number.isFinite(recoveryCheckpointBatchesRaw)
      ? Math.min(50000, Math.max(0, Math.round(recoveryCheckpointBatchesRaw)))
      : 1000,
    recoveryCheckpointIntervalMinutes: Number.isFinite(recoveryCheckpointMinutesRaw)
      ? Math.min(240, Math.max(0, recoveryCheckpointMinutesRaw))
      : 20,
    maxAutoRecoveryRestarts: Number.isFinite(maxAutoRecoveryRestartsRaw)
      ? Math.min(32, Math.max(0, Math.round(maxAutoRecoveryRestartsRaw)))
      : 12,
    vocabularyLimit: Math.max(256, Number(training.vocabularyLimit) || 8000),
  };
}

module.exports = {
  normalizeExecutionMode,
  resolveTrainingSettings,
};
