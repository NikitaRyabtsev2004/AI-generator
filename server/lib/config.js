const MODEL_ENGINE = 'hybrid-rag-transformer-v5';

const DEFAULT_SETTINGS = {
  training: {
    epochs: 16,
    batchSize: 8,
    executionMode: 'native_preferred',
    sequenceLength: 128,
    embeddingSize: 128,
    attentionHeads: 4,
    transformerLayers: 4,
    feedForwardSize: 384,
    hiddenSize: 384,
    recurrentLayers: 4,
    tokenizerMode: 'simple_subword',
    dropout: 0.12,
    learningRate: 0.001,
    chunkSize: 180,
    chunkOverlap: 36,
    vocabularyLimit: 12000,
    topKChunks: 6,
    minSimilarity: 0.12,
    maxHistoryPoints: 360,
    autoClearSourcesAfterTraining: true,
  },
  generation: {
    responseTemperature: 0.4,
    maxReplySentences: 4,
    maxReplyCharacters: 1400,
    maxGeneratedTokens: 96,
    topKSampling: 16,
    repetitionPenalty: 1.22,
  },
};

const MODEL_STATUS_FLOW = [
  'not_created',
  'ready_for_training',
  'training',
  'paused',
  'trained',
  'generating_reply',
  'syncing_knowledge',
  'learning_from_feedback',
  'error',
];

function createDefaultModelRegistryItem() {
  return {
    id: '',
    name: '',
    kind: 'local',
    createdAt: null,
    updatedAt: null,
    lastUsedAt: null,
    packagePath: '',
    hasCheckpoint: false,
    summary: {
      lifecycle: 'not_created',
      trainedEpochs: 0,
      parameterCount: 0,
      vocabularySize: 0,
      tokenCount: 0,
      sourceCount: 0,
      replyPairCount: 0,
      backend: 'neural',
    },
  };
}

function createDefaultModelRegistryState() {
  return {
    activeModelId: null,
    items: [],
  };
}

function createDefaultModelState() {
  return {
    exists: false,
    engine: MODEL_ENGINE,
    lifecycle: 'not_created',
    status: 'idle',
    sourceCount: 0,
    chatCount: 0,
    replyPairCount: 0,
    chunkCount: 0,
    trainingItemCount: 0,
    batchesPerEpoch: 0,
    tokenCount: 0,
    sentenceCount: 0,
    vocabularySize: 0,
    parameterCount: 0,
    computeBackend: 'cpu',
    computeBackendLabel: 'CPU (совместимый режим)',
    computeBackendWarning: '',
    trainedEpochs: 0,
    targetEpochs: 0,
    batchesProcessed: 0,
    lastLoss: null,
    averageLoss: null,
    validationLoss: null,
    bestValidationLoss: null,
    perplexity: null,
    lastTrainingAt: null,
    lastGenerationAt: null,
    corpusSignature: null,
    positiveFeedbackCount: 0,
    negativeFeedbackCount: 0,
    averageSelfScore: null,
    artifactFiles: [],
    storage: {
      databasePath: '',
      artifactDir: '',
      manifestPath: '',
      knowledgeIndexPath: '',
      languageModelPath: '',
      neuralModelDir: '',
      tokenizerPath: '',
      neuralWeightsPath: '',
      neuralSpecPath: '',
      runtimeConfigPath: '',
    },
    configSnapshot: JSON.parse(JSON.stringify(DEFAULT_SETTINGS)),
    topTerms: [],
    notes: [
      'Чаты и источники хранятся в SQLite, а чекпоинты модели, токенизатор и индекс знаний лежат отдельными артефактами на диске.',
      'Обучение использует реальную языковую модель по токенам, отдельный токенизатор, валидацию и сохранение весов.',
      'Ответы строятся из памяти чата, retrieval по корпусу, нейросетевой генерации и пользовательской обратной связи.',
    ],
  };
}

function createDefaultTrainingState() {
  return {
    status: 'idle',
    phase: 'waiting',
    message: 'Модель еще не создана.',
    startedAt: null,
    updatedAt: null,
    requestedStop: false,
    progress: {
      currentEpoch: 0,
      totalEpochs: 0,
      currentBatch: 0,
      totalBatches: 0,
      percent: 0,
    },
    history: [],
    recentStatuses: [],
  };
}

function createDefaultTrainingQueueRunnerState() {
  return {
    active: false,
    status: 'idle',
    startedAt: null,
    updatedAt: null,
    currentQueueId: null,
    currentQueueIndex: 0,
    totalQueues: 0,
    pendingQueueIds: [],
    completedQueueIds: [],
    lastCompletedQueueId: null,
    failedQueueId: null,
    lastError: '',
  };
}

function createDefaultTrainingQueuesState() {
  return {
    items: [],
    runner: createDefaultTrainingQueueRunnerState(),
  };
}

function createDefaultKnowledgeState() {
  return {
    chunks: [],
    replyMemories: [],
    vocabulary: {},
    languageModel: {
      kind: 'tf-keras-llm',
      vocabularySize: 0,
      parameterCount: 0,
      tokenizerReady: false,
      checkpointReady: false,
      lastSavedAt: null,
      trainingSequenceCount: 0,
      corpusTokenCount: 0,
      feedbackExampleCount: 0,
      validationLoss: null,
      bestValidationLoss: null,
      trainingSampleCount: 0,
      validationSampleCount: 0,
    },
    bm25: {
      knowledge: {
        documentCount: 0,
        averageDocumentLength: 0,
        documentFrequency: {},
        idf: {},
      },
      reply: {
        documentCount: 0,
        averageDocumentLength: 0,
        documentFrequency: {},
        idf: {},
      },
    },
  };
}

function createDefaultState() {
  return {
    meta: {
      version: 7,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
    settings: JSON.parse(JSON.stringify(DEFAULT_SETTINGS)),
    sources: [],
    modelRegistry: createDefaultModelRegistryState(),
    trainingQueues: createDefaultTrainingQueuesState(),
    chats: [],
    model: createDefaultModelState(),
    training: createDefaultTrainingState(),
    knowledge: createDefaultKnowledgeState(),
  };
}

module.exports = {
  DEFAULT_SETTINGS,
  MODEL_ENGINE,
  MODEL_STATUS_FLOW,
  createDefaultKnowledgeState,
  createDefaultModelState,
  createDefaultModelRegistryItem,
  createDefaultModelRegistryState,
  createDefaultState,
  createDefaultTrainingState,
  createDefaultTrainingQueueRunnerState,
  createDefaultTrainingQueuesState,
};
