const fs = require('fs/promises');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'data');
const RUNTIME_CONFIG_PATH = path.join(DATA_DIR, 'runtime-config.json');

const DEFAULT_RUNTIME_CONFIG = {
  context: {
    strategy: 'whole_chat_weighted',
    maxMessages: 1000,
    maxCharacters: 64000,
    includeAssistantMessages: false,
    assistantWeight: 0.3,
    userWeight: 1,
    decay: 0.92,
  },
  dialogueMemory: {
    useSourceDialogues: true,
    useChatDialogues: false,
    strongMatchThreshold: 1.08,
    softMatchThreshold: 0.62,
    responseOnlyIndex: true,
  },
  retrieval: {
    assistantReplyBoost: 1.3,
    knowledgeChunkBoost: 1,
    topReplyPairs: 6,
    topKnowledgeChunks: 8,
    bm25k1: 1.5,
    bm25b: 0.75,
  },
  generation: {
    backend: 'neural',
    systemPrompt:
      'Ты полезный ассистент. Отвечай на языке пользователя, не выдумывай факты и явно говори, если данных недостаточно. Используй предоставленные знания, контекст чата и найденные источники.',
    webSearchEnabled: false,
    webSearchMaxResults: 4,
    webSearchFetchPages: 2,
    webSearchTimeoutMs: 12000,
    webSearchPreferredDomains: '',
  },
  feedback: {
    autoAcceptThreshold: 0.72,
    negativeSuppressionThreshold: -0.2,
    positiveReplayWeight: 3,
  },
};

let runtimeConfig = null;

function deepMerge(baseValue, nextValue) {
  if (Array.isArray(baseValue)) {
    return Array.isArray(nextValue) ? [...nextValue] : [...baseValue];
  }

  if (
    baseValue &&
    typeof baseValue === 'object' &&
    nextValue &&
    typeof nextValue === 'object'
  ) {
    const merged = { ...baseValue };
    Object.keys(nextValue).forEach((key) => {
      merged[key] = deepMerge(baseValue[key], nextValue[key]);
    });
    return merged;
  }

  return nextValue === undefined ? baseValue : nextValue;
}

function sanitizeRuntimeConfig(nextConfig = {}) {
  const merged = deepMerge(DEFAULT_RUNTIME_CONFIG, nextConfig);

  return {
    context: deepMerge(DEFAULT_RUNTIME_CONFIG.context, merged.context),
    dialogueMemory: deepMerge(DEFAULT_RUNTIME_CONFIG.dialogueMemory, merged.dialogueMemory),
    retrieval: deepMerge(DEFAULT_RUNTIME_CONFIG.retrieval, merged.retrieval),
    generation: {
      ...deepMerge(DEFAULT_RUNTIME_CONFIG.generation, merged.generation),
      backend: 'neural',
    },
    feedback: deepMerge(DEFAULT_RUNTIME_CONFIG.feedback, merged.feedback),
  };
}

async function ensureRuntimeConfigFile() {
  await fs.mkdir(DATA_DIR, { recursive: true });

  try {
    await fs.access(RUNTIME_CONFIG_PATH);
  } catch (_error) {
    await fs.writeFile(
      RUNTIME_CONFIG_PATH,
      JSON.stringify(DEFAULT_RUNTIME_CONFIG, null, 2),
      'utf8'
    );
  }
}

async function initializeRuntimeConfig() {
  await ensureRuntimeConfigFile();
  const raw = await fs.readFile(RUNTIME_CONFIG_PATH, 'utf8');
  const parsed = JSON.parse(raw);
  runtimeConfig = sanitizeRuntimeConfig(parsed);
  return runtimeConfig;
}

function getRuntimeConfig() {
  if (!runtimeConfig) {
    throw new Error('Runtime config is not initialized yet.');
  }

  return runtimeConfig;
}

async function persistRuntimeConfig() {
  if (!runtimeConfig) {
    return;
  }

  await fs.writeFile(RUNTIME_CONFIG_PATH, JSON.stringify(runtimeConfig, null, 2), 'utf8');
}

async function updateRuntimeConfig(partialRuntimeConfig) {
  runtimeConfig = sanitizeRuntimeConfig(
    deepMerge(getRuntimeConfig(), partialRuntimeConfig || {})
  );
  await persistRuntimeConfig();
  return runtimeConfig;
}

module.exports = {
  DEFAULT_RUNTIME_CONFIG,
  RUNTIME_CONFIG_PATH,
  getRuntimeConfig,
  initializeRuntimeConfig,
  persistRuntimeConfig,
  updateRuntimeConfig,
};
