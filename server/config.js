const MODEL_ENGINE = 'hybrid-rag-neural-v4';

const DEFAULT_SETTINGS = {
  training: {
    epochs: 24,
    batchSize: 8,
    executionMode: 'compatibility',
    sequenceLength: 96,
    embeddingSize: 128,
    hiddenSize: 256,
    recurrentLayers: 2,
    dropout: 0.15,
    learningRate: 0.0012,
    chunkSize: 180,
    chunkOverlap: 36,
    vocabularyLimit: 12000,
    topKChunks: 6,
    minSimilarity: 0.12,
    maxHistoryPoints: 360,
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

function createDefaultKnowledgeState() {
  return {
    chunks: [],
    replyMemories: [],
    vocabulary: {},
    languageModel: {
      kind: 'tfjs-neural-lm',
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
      version: 5,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
    settings: JSON.parse(JSON.stringify(DEFAULT_SETTINGS)),
    sources: [],
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
  createDefaultState,
  createDefaultTrainingState,
};
