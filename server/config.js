const MODEL_ENGINE = 'hybrid-rag-neural-v3';

const DEFAULT_SETTINGS = {
  training: {
    epochs: 8,
    batchSize: 16,
    sequenceLength: 80,
    embeddingSize: 96,
    hiddenSize: 192,
    recurrentLayers: 2,
    dropout: 0.12,
    learningRate: 0.0015,
    chunkSize: 180,
    chunkOverlap: 36,
    vocabularyLimit: 12000,
    topKChunks: 6,
    minSimilarity: 0.12,
    maxHistoryPoints: 360,
  },
  generation: {
    responseTemperature: 0.45,
    maxReplySentences: 4,
    maxReplyCharacters: 1400,
    maxGeneratedTokens: 72,
    topKSampling: 18,
    repetitionPenalty: 1.18,
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
    trainedEpochs: 0,
    targetEpochs: 0,
    batchesProcessed: 0,
    lastLoss: null,
    averageLoss: null,
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
      'Model data is stored separately from user data: SQLite keeps chats and sources, artifact files keep the trained index and language model.',
      'Training now fits a real recurrent neural language model over token sequences and keeps a separate tokenizer, weight checkpoint and retrieval index.',
      'Replies use neural generation plus retrieval grounding, current chat context and user feedback so the model can improve over time.',
    ],
  };
}

function createDefaultTrainingState() {
  return {
    status: 'idle',
    phase: 'waiting',
    message: 'Model has not been created yet.',
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
