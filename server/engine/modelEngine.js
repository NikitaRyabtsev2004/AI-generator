const crypto = require('crypto');
const path = require('path');
const { Worker } = require('node:worker_threads');
const fs = require('fs/promises');
const { log, logError } = require('../lib/logger');
const {
  MODEL_STATUS_FLOW,
  createDefaultKnowledgeState,
  createDefaultModelState,
  createDefaultModelRegistryItem,
  createDefaultModelRegistryState,
  createDefaultTrainingState,
  createDefaultTrainingQueuesState,
} = require('../lib/config');
const {
  disposeRuntime,
  generateText,
  loadRuntime,
  tokenizeForModel,
} = require('./neuralModel');
const { getRuntimeConfig, updateRuntimeConfig } = require('../storage/runtimeConfig');
const {
  getModelLibraryPackagePath,
  modelLibraryPackageExists,
  readModelLibraryPackage,
  removeModelLibraryPackage,
  writeModelLibraryPackage,
} = require('../storage/modelLibraryStorage');
const {
  cleanText,
  computeStats,
  createId,
  inferChatTitle,
  previewText,
  splitIntoSentences,
  tokenizeWords,
} = require('../lib/text');
const {
  cleanupTrainingQueueStorage,
  createTrainingJobPayloadFile,
  persistTrainingQueueDatasetFile,
  readTrainingQueueSourceContent,
  removeTrainingJobPayloadFile,
  removeTrainingQueueDirectory,
  removeTrainingQueueSourceContent,
  writeTrainingQueueSourceContent,
} = require('../storage/trainingQueueStorage');
const { fetchSearchDocuments } = require('../lib/webSearch');
const { resolveTrainingSettings } = require('../lib/modelSettings');

const RETRAINING_KEYS = [
  'sequenceLength',
  'embeddingSize',
  'attentionHeads',
  'transformerLayers',
  'feedForwardSize',
  'tokenizerMode',
  'dropout',
  'chunkSize',
  'chunkOverlap',
  'vocabularyLimit',
  'topKChunks',
  'minSimilarity',
];
const CORPUS_REBUILD_KEYS = [
  'chunkSize',
  'chunkOverlap',
  'vocabularyLimit',
  'topKChunks',
  'minSimilarity',
];

const TRAINING_LOCKED_STATUSES = new Set(['training', 'syncing_knowledge']);
const ARCHIVED_REPLY_REPLAY_LIMIT = 240;
const MODEL_PACKAGE_FORMAT = 'ai-generator-model-package';
const MODEL_PACKAGE_VERSION = 1;
const TRAINING_STOP_TIMEOUT_MS = 8000;
const SHORT_SOCIAL_MESSAGE_PATTERN = /^(привет|здравствуй|здравствуйте|добрый день|доброе утро|добрый вечер|hello|hi|hey|yo|hola)\b/iu;
const MARKDOWN_LINK_PATTERN = /\[([^\]]{1,120})\]\((https?:\/\/[^\s)]+)\)/giu;
const URL_PATTERN = /https?:\/\/[^\s)]+/giu;
const NOISY_WEB_HOST_PATTERN = /(dictionary|translate|wiktionary|reverso|thesaurus)/iu;
const ARTIFACT_NOISE_PATTERN =
  /\b(commit\s*:|old_file\s*:|new_file\s*:|old_contents\s*:|new_contents\s*:|repos\s*:|license\s*:|data\.jsonl)\b/iu;
const STRUCTURED_TEXT_FIELDS = new Set([
  'text',
  'comment',
  'description',
  'explanation',
  'summary',
  'content',
]);
const STRUCTURED_CODE_FIELDS = new Set([
  'code',
  'snippet',
  'example',
  'new_contents',
  'old_contents',
]);
const STRUCTURED_LANGUAGE_FIELDS = new Set([
  'language',
  'lang',
  'syntax',
]);
const LOOSE_STRUCTURED_FIELD_PATTERN =
  /["']?(text|comment|description|explanation|summary|content|code|snippet|example|new_contents|old_contents|language|lang|syntax)["']?\s*:\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/giu;
const COMMON_QUERY_STOPWORDS = new Set([
  'и', 'а', 'но', 'или', 'если', 'то', 'же', 'ли', 'бы', 'в', 'во', 'на', 'по', 'к', 'ко',
  'с', 'со', 'у', 'о', 'об', 'от', 'до', 'за', 'из', 'под', 'над', 'при', 'для', 'про',
  'через', 'без', 'не', 'ни', 'я', 'ты', 'он', 'она', 'мы', 'вы', 'они', 'это', 'этот',
  'эта', 'эти', 'тот', 'та', 'те', 'мне', 'тебе', 'ему', 'ей', 'нам', 'вам', 'их', 'мой',
  'твой', 'его', 'ее', 'наш', 'ваш', 'их', 'как', 'что', 'кто', 'где', 'когда', 'почему',
  'зачем', 'which', 'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'a', 'an',
  'the', 'to', 'for', 'of', 'in', 'on', 'at', 'with', 'by', 'from', 'and', 'or', 'if',
  'then', 'it', 'this', 'that', 'these', 'those'
]);
const MAX_DATASET_RECORDS_PER_FILE = Math.max(1000, Number(process.env.MAX_DATASET_RECORDS_PER_FILE) || 250000);
const MAX_DATASET_CHARS_PER_RECORD = Math.max(128, Number(process.env.MAX_DATASET_CHARS_PER_RECORD) || 12000);
const DATASET_PARQUET_BATCH_SIZE = Math.max(32, Number(process.env.DATASET_PARQUET_BATCH_SIZE) || 256);

function nowIso() {
  return new Date().toISOString();
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function yieldToEventLoop() {
  return new Promise((resolve) => {
    setImmediate(resolve);
  });
}

function createTrainingPreparationStoppedError(message = 'Подготовка обучения остановлена пользователем.') {
  const error = new Error(message);
  error.code = 'TRAINING_PREPARATION_STOPPED';
  return error;
}

function throwIfTrainingPreparationStopped(shouldStop) {
  if (typeof shouldStop === 'function' && shouldStop()) {
    throw createTrainingPreparationStoppedError();
  }
}

function toPositiveNumber(value, fallback = 0) {
  const normalized = Number(value);
  if (!Number.isFinite(normalized) || normalized < 0) {
    return fallback;
  }
  return normalized;
}

function normalizeOptionalPath(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeStoragePaths(storage = {}) {
  if (!storage || typeof storage !== 'object') {
    return {};
  }

  return Object.fromEntries(
    Object.entries(storage).map(([key, value]) => [key, normalizeOptionalPath(value)])
  );
}

async function pathExists(targetPath) {
  const normalizedPath = normalizeOptionalPath(targetPath);
  if (!normalizedPath) {
    return false;
  }

  try {
    await fs.access(normalizedPath);
    return true;
  } catch (_error) {
    return false;
  }
}

async function readJsonFileIfExists(targetPath, fallbackValue = null) {
  const normalizedPath = normalizeOptionalPath(targetPath);
  if (!(await pathExists(normalizedPath))) {
    return fallbackValue;
  }

  try {
    const raw = await fs.readFile(normalizedPath, 'utf8');
    return JSON.parse(raw);
  } catch (_error) {
    return fallbackValue;
  }
}

async function writeJsonFile(targetPath, payload) {
  await fs.mkdir(path.dirname(targetPath), { recursive: true });
  await fs.writeFile(targetPath, JSON.stringify(payload, null, 2), 'utf8');
}

function computeParameterCountFromSpecs(specs = []) {
  return specs.reduce((sum, spec) => {
    const shape = Array.isArray(spec?.shape) ? spec.shape : [];
    const weightCount = shape.reduce((acc, dim) => acc * Math.max(Number(dim) || 0, 1), 1);
    return sum + weightCount;
  }, 0);
}

function normalizeModelForExport(modelState) {
  const {
    storage: _ignoredStorage,
    artifactFiles: _ignoredArtifactFiles,
    ...persistedModel
  } = modelState || {};
  return persistedModel;
}

function getGeneratorBackend() {
  return 'neural';
}

function buildTrainingSettingsSnapshot(trainingSettings = {}) {
  const resolved = resolveTrainingSettings(trainingSettings);

  return {
    sequenceLength: resolved.sequenceLength,
    embeddingSize: resolved.embeddingSize,
    attentionHeads: resolved.attentionHeads,
    transformerLayers: resolved.transformerLayers,
    feedForwardSize: resolved.feedForwardSize,
    hiddenSize: resolved.feedForwardSize,
    recurrentLayers: resolved.transformerLayers,
    dropout: resolved.dropout,
    learningRate: resolved.learningRate,
  };
}

function summarizeModelRegistryEntry(entry) {
  return {
    id: entry.id,
    name: entry.name,
    kind: entry.kind || 'local',
    createdAt: entry.createdAt || null,
    updatedAt: entry.updatedAt || null,
    lastUsedAt: entry.lastUsedAt || null,
    hasCheckpoint: Boolean(entry.hasCheckpoint),
    summary: {
      lifecycle: entry.summary?.lifecycle || 'not_created',
      trainedEpochs: Number(entry.summary?.trainedEpochs) || 0,
      parameterCount: Number(entry.summary?.parameterCount) || 0,
      vocabularySize: Number(entry.summary?.vocabularySize) || 0,
      tokenCount: Number(entry.summary?.tokenCount) || 0,
      sourceCount: Number(entry.summary?.sourceCount) || 0,
      replyPairCount: Number(entry.summary?.replyPairCount) || 0,
      backend: entry.summary?.backend || 'neural',
    },
  };
}

function buildModelRegistrySummaryFromState(state) {
  return {
    lifecycle: state.model.lifecycle,
    trainedEpochs: Number(state.model.trainedEpochs) || 0,
    parameterCount: Number(state.model.parameterCount) || 0,
    vocabularySize: Number(state.model.vocabularySize) || 0,
    tokenCount: Number(state.model.tokenCount) || 0,
    sourceCount: Number(state.model.sourceCount) || 0,
    replyPairCount: Number(state.model.replyPairCount) || 0,
    backend: getGeneratorBackend(),
  };
}

function createModelRegistryItemFromState(state, name = '') {
  const timestamp = nowIso();
  return {
    ...createDefaultModelRegistryItem(),
    id: createId('model'),
    name: cleanText(name || '') || `Модель ${Math.max((state.modelRegistry?.items || []).length, 0) + 1}`,
    kind: 'local',
    createdAt: timestamp,
    updatedAt: timestamp,
    lastUsedAt: timestamp,
    packagePath: '',
    hasCheckpoint: Boolean(state.knowledge.languageModel?.checkpointReady),
    summary: buildModelRegistrySummaryFromState(state),
  };
}

function createFreshModelRegistryItem(state, name = '') {
  const timestamp = nowIso();
  const base = createDefaultModelRegistryItem();
  return {
    ...base,
    id: createId('model'),
    name: cleanText(name || '') || `Модель ${Math.max((state.modelRegistry?.items || []).length, 0) + 1}`,
    kind: 'local',
    createdAt: timestamp,
    updatedAt: timestamp,
    lastUsedAt: timestamp,
    packagePath: '',
    hasCheckpoint: false,
    summary: {
      ...base.summary,
      lifecycle: 'not_created',
      trainedEpochs: 0,
      parameterCount: 0,
      vocabularySize: 0,
      tokenCount: 0,
      sourceCount: 0,
      replyPairCount: 0,
      backend: getGeneratorBackend(),
    },
  };
}

function ensureModelRegistryState(state) {
  if (!state.modelRegistry || !Array.isArray(state.modelRegistry.items)) {
    state.modelRegistry = createDefaultModelRegistryState();
  }

  if (!state.modelRegistry.items.length) {
    const entry = createModelRegistryItemFromState(state, state.model.exists ? 'Текущая модель' : 'Новая модель');
    entry.packagePath = getModelLibraryPackagePath(entry.id);
    state.modelRegistry.items = [entry];
    state.modelRegistry.activeModelId = entry.id;
    return entry;
  }

  if (!state.modelRegistry.activeModelId || !state.modelRegistry.items.some((item) => item.id === state.modelRegistry.activeModelId)) {
    state.modelRegistry.activeModelId = state.modelRegistry.items[0]?.id || null;
  }

  const activeItem = state.modelRegistry.items.find((item) => item.id === state.modelRegistry.activeModelId) || null;
  if (activeItem && !activeItem.packagePath) {
    activeItem.packagePath = getModelLibraryPackagePath(activeItem.id);
  }
  return activeItem;
}

function updateActiveModelRegistrySummary(state) {
  const activeItem = ensureModelRegistryState(state);
  if (!activeItem) {
    return null;
  }

  activeItem.updatedAt = nowIso();
  activeItem.lastUsedAt = nowIso();
  activeItem.hasCheckpoint = Boolean(state.knowledge.languageModel?.checkpointReady);
  activeItem.summary = buildModelRegistrySummaryFromState(state);
  return activeItem;
}

async function pruneBrokenModelRegistryItems(state) {
  ensureModelRegistryState(state);
  const activeModelId = state.modelRegistry.activeModelId;
  const items = Array.isArray(state.modelRegistry.items) ? state.modelRegistry.items : [];
  if (!items.length) {
    return;
  }

  const keptItems = [];
  for (const item of items) {
    if (!item?.id) {
      continue;
    }

    if (item.id === activeModelId) {
      keptItems.push(item);
      continue;
    }

    if (await modelLibraryPackageExists(item.id)) {
      keptItems.push(item);
    }
  }

  state.modelRegistry.items = keptItems;
  if (!state.modelRegistry.items.some((item) => item.id === state.modelRegistry.activeModelId)) {
    state.modelRegistry.activeModelId = state.modelRegistry.items[0]?.id || null;
  }
}

function structuredTrainingText(parts) {
  return cleanText(parts.filter(Boolean).join(' '));
}

function createEmptyChat() {
  const timestamp = nowIso();
  return {
    id: createId('chat'),
    title: 'Новый чат',
    createdAt: timestamp,
    updatedAt: timestamp,
    messages: [
      {
        id: createId('msg'),
        role: 'assistant',
        content:
          'Я готов работать с моделью и базой знаний. После обучения смогу отвечать с учетом корпуса, контекста чата и оценок качества.',
        createdAt: timestamp,
        metadata: {
          type: 'system',
        },
      },
    ],
  };
}

function pushStatus(state, status, phase, message, options = {}) {
  const { updateTrainingState = true } = options;
  const normalizedMessage = cleanText(message);
  const entry = {
    id: createId('status'),
    status,
    phase,
    message: normalizedMessage,
    createdAt: nowIso(),
  };

  if (updateTrainingState) {
    state.training.status = status;
    state.training.phase = phase;
    state.training.message = normalizedMessage;
    state.training.updatedAt = entry.createdAt;
  }
  state.training.recentStatuses = [entry, ...(state.training.recentStatuses || [])].slice(0, 32);
  log('info', normalizedMessage || 'Training status updated.', {
    scope: 'training_status',
    status,
    phase,
    updateTrainingState,
  });
}

function ensureChatAvailability(state) {
  if (!Array.isArray(state.chats)) {
    state.chats = [];
  }

  if (!state.chats.length) {
    state.chats.push(createEmptyChat());
  }
}

function summarizeChat(chat) {
  return {
    id: chat.id,
    title: chat.title,
    createdAt: chat.createdAt,
    updatedAt: chat.updatedAt,
    messageCount: chat.messages.length,
    lastMessagePreview: previewText(chat.messages[chat.messages.length - 1]?.content || '', 90),
  };
}

const MAX_ACTIVE_CHAT_MESSAGES = 400;
const MAX_ACTIVE_CHAT_MESSAGE_CHARS = 24000;
const MAX_ACTIVE_CHAT_TOTAL_CHARS = 2_000_000;

function sanitizeMessageMetadataForTransport(metadata = null) {
  if (!metadata || typeof metadata !== 'object') {
    return metadata || null;
  }
  const nextMetadata = { ...metadata };
  if (Array.isArray(nextMetadata.references)) {
    nextMetadata.references = nextMetadata.references.slice(0, 24).map((reference) => ({
      ...reference,
      title: previewText(reference?.title || '', 180),
      excerpt: previewText(reference?.excerpt || '', 600),
      host: previewText(reference?.host || '', 120),
      url: reference?.url || '',
    }));
  }
  return nextMetadata;
}

function summarizeActiveChat(chat) {
  if (!chat) {
    return null;
  }
  const allMessages = Array.isArray(chat.messages) ? chat.messages : [];
  const tailMessages = allMessages.slice(-MAX_ACTIVE_CHAT_MESSAGES);
  let totalChars = 0;
  const compactMessages = [];

  for (let index = tailMessages.length - 1; index >= 0; index -= 1) {
    const message = tailMessages[index];
    const content = String(message?.content || '');
    const trimmedContent = content.length > MAX_ACTIVE_CHAT_MESSAGE_CHARS
      ? `${content.slice(0, MAX_ACTIVE_CHAT_MESSAGE_CHARS)}\n...[truncated]`
      : content;
    const nextSize = totalChars + trimmedContent.length;
    if (nextSize > MAX_ACTIVE_CHAT_TOTAL_CHARS) {
      break;
    }
    totalChars = nextSize;
    compactMessages.push({
      id: message.id,
      role: message.role,
      content: trimmedContent,
      createdAt: message.createdAt,
      metadata: sanitizeMessageMetadataForTransport(message.metadata),
    });
  }

  compactMessages.reverse();
  return {
    id: chat.id,
    title: chat.title,
    createdAt: chat.createdAt,
    updatedAt: chat.updatedAt,
    messages: compactMessages,
    totalMessageCount: allMessages.length,
    returnedMessageCount: compactMessages.length,
    truncated: compactMessages.length < allMessages.length,
  };
}

function summarizeSource(source) {
  return {
    id: source.id,
    type: source.type,
    label: source.label,
    url: source.url || null,
    addedAt: source.addedAt,
    stats: source.stats,
    contentSize: Math.max(Number(source.contentSize) || 0, 0),
    preview: previewText(source.content, 220),
  };
}

function summarizeTrainingQueueSource(source) {
  return {
    id: source.id,
    type: source.type,
    label: source.label,
    url: source.url || null,
    addedAt: source.addedAt,
    stats: source.stats,
    contentSize: Math.max(Number(source.contentSize) || 0, 0),
    preview: '',
  };
}

function summarizeTrainingQueue(queue, index = 0) {
  return {
    id: queue.id,
    name: queue.name,
    status: queue.status || 'pending',
    createdAt: queue.createdAt,
    updatedAt: queue.updatedAt,
    completedAt: queue.completedAt || null,
    lastRunAt: queue.lastRunAt || null,
    lastError: cleanText(queue.lastError || ''),
    index,
    sourceCount: Array.isArray(queue.sources) ? queue.sources.length : 0,
    sources: Array.isArray(queue.sources) ? queue.sources.map(summarizeTrainingQueueSource) : [],
  };
}

function normalizeTrainingQueueName(name, fallbackIndex = 1) {
  return cleanText(name || '') || `Очередь ${fallbackIndex}`;
}

function cloneTrainingQueueSources(sources = []) {
  return (Array.isArray(sources) ? sources : []).map((source) => ({
    id: source.id,
    type: source.type,
    label: source.label,
    url: source.url || null,
    stats: source.stats || {},
    addedAt: source.addedAt || null,
    contentPath: source.contentPath || null,
    contentSize: Math.max(Number(source.contentSize) || 0, 0),
    ...(source.content ? { content: source.content } : {}),
  }));
}

function getPendingTrainingQueues(state) {
  return (state.trainingQueues?.items || []).filter(
    (queue) => Array.isArray(queue.sources) && queue.sources.length > 0 && queue.status !== 'completed'
  );
}

function hasPendingTrainingQueues(state) {
  return getPendingTrainingQueues(state).length > 0;
}

function resetTrainingQueueRunnerState(state, status = 'idle', lastError = '') {
  state.trainingQueues.runner = {
    ...createDefaultTrainingQueuesState().runner,
    status,
    active: false,
    updatedAt: nowIso(),
    lastError: cleanText(lastError || ''),
  };
}

function restoreQueuesAfterModelReset(state) {
  state.trainingQueues.items = (state.trainingQueues.items || []).map((queue) => ({
    ...queue,
    status: Array.isArray(queue.sources) && queue.sources.length ? 'pending' : 'empty',
    completedAt: null,
    lastRunAt: null,
    lastError: '',
    updatedAt: nowIso(),
  }));
  resetTrainingQueueRunnerState(state);
}

function addWeightedTokens(targetMap, tokens, weight = 1) {
  tokens.forEach((token) => {
    targetMap[token] = (targetMap[token] || 0) + weight;
  });
}

function isInformativeQueryToken(token) {
  const normalized = String(token || '').toLowerCase();
  if (!normalized) {
    return false;
  }

  if (COMMON_QUERY_STOPWORDS.has(normalized)) {
    return false;
  }

  if (normalized.length <= 1) {
    return false;
  }

  if (/^\d+$/u.test(normalized)) {
    return false;
  }

  return true;
}

function extractQueryTokens(text, { fallbackToAll = false } = {}) {
  const allTokens = tokenizeWords(text || '');
  const informativeTokens = allTokens.filter(isInformativeQueryToken);
  if (informativeTokens.length || !fallbackToAll) {
    return informativeTokens;
  }

  return allTokens;
}

function createTermMap(tokens, allowedTerms = null) {
  const map = {};
  tokens.forEach((token) => {
    if (allowedTerms && !allowedTerms.has(token)) {
      return;
    }
    map[token] = (map[token] || 0) + 1;
  });
  return map;
}

function normalizeSpeakerLabel(value) {
  const normalized = cleanText(value).toLowerCase();
  if (['user', 'human', 'client', 'пользователь', 'клиент', 'юзер', 'a', 'а'].includes(normalized)) {
    return 'user';
  }

  if (['assistant', 'bot', 'ассистент', 'бот', 'ai', 'b', 'б'].includes(normalized)) {
    return 'assistant';
  }

  return null;
}

function extractDialogueTurns(text) {
  const cleanValue = cleanText(text);
  const speakerPattern = /(Пользователь|Клиент|Юзер|User|Human|Client|Ассистент|Бот|Assistant|Bot|AI|A|B|А|Б)\s*:\s*/giu;
  const matches = Array.from(cleanValue.matchAll(speakerPattern));
  const turns = [];

  matches.forEach((match, index) => {
    const role = normalizeSpeakerLabel(match[1]);
    if (!role) {
      return;
    }

    const contentStart = (match.index || 0) + match[0].length;
    const contentEnd = index + 1 < matches.length ? matches[index + 1].index : cleanValue.length;
    const content = cleanText(cleanValue.slice(contentStart, contentEnd));
    if (!content) {
      return;
    }

    turns.push({ role, content });
  });

  return turns;
}

function extractDialoguePairsFromTurns(turns, ownerId, title, origin = 'source', score = 0.75) {
  const pairs = [];

  for (let index = 0; index < turns.length - 1; index += 1) {
    const userTurn = turns[index];
    const assistantTurn = turns[index + 1];
    if (userTurn.role !== 'user' || assistantTurn.role !== 'assistant') {
      continue;
    }

    pairs.push({
      id: createId('pair'),
      ownerId,
      title,
      origin,
      score,
      promptText: cleanText(userTurn.content),
      responseText: cleanText(assistantTurn.content),
      combinedText: cleanText(`Пользователь: ${userTurn.content}\nАссистент: ${assistantTurn.content}`),
    });
  }

  return pairs;
}

function extractDialoguePairsFromSource(source) {
  return extractDialoguePairsFromTurns(
    extractDialogueTurns(source.content),
    source.id,
    source.label,
    'source',
    0.9
  );
}

function startsWithSpeakerLabel(text) {
  return /^\s*(пользователь|user|assistant|ассистент|бот|bot|client|клиент|human|ai|a|b|а|б)\s*:/iu.test(text);
}

function sanitizeReplyText(text) {
  return cleanText(text)
    .replace(/^\s*[-\u2013\u2014]\s*/u, '')
    .replace(/^\s*(пользователь|user|assistant|ассистент|бот|bot|client|клиент|human|ai|a|b|а|б)\s*:\s*/iu, '')
    .replace(/\b(пользователь|user|assistant|ассистент|бот|bot|client|клиент|human|ai|a|b|а|б)\s*:\s*/giu, '')
    .trim();
}

function isSimpleGreetingMessage(userMessage) {
  const normalized = cleanText(userMessage || '').toLowerCase();
  if (!normalized) {
    return false;
  }
  const tokens = tokenizeWords(normalized);
  return tokens.length <= 4 && SHORT_SOCIAL_MESSAGE_PATTERN.test(normalized);
}

function buildGreetingReply(language = 'auto') {
  if (language === 'en') {
    return 'Hi! I am ready to help. Tell me what you need and I will answer briefly and clearly.';
  }
  return 'Привет! Я на связи. Напишите задачу, и я отвечу коротко и по делу.';
}

function shouldSkipWebSearchForMessage(userMessage) {
  const normalized = cleanText(userMessage || '').toLowerCase();
  const tokens = tokenizeWords(normalized);
  return tokens.length <= 4 && SHORT_SOCIAL_MESSAGE_PATTERN.test(normalized);
}

function shouldForceWebSearchForMessage(userMessage) {
  const normalized = cleanText(userMessage || '').toLowerCase();
  if (!normalized) {
    return false;
  }
  if (normalized.includes('?')) {
    return true;
  }

  const intentPattern =
    /\b(как|что|кто|почему|зачем|где|когда|какой|какая|какие|how|what|why|when|where|which|guide|tutorial)\b/iu;
  const techPattern =
    /\b(javascript|js|typescript|react|node|python|sql|api|framework|library|алгоритм|программирован|код)\b/iu;

  return intentPattern.test(normalized) || techPattern.test(normalized);
}

function isTechnicalOrMathQuery(userMessage) {
  const normalized = cleanText(userMessage || '').toLowerCase();
  if (!normalized) {
    return false;
  }

  const technicalPattern =
    /\b(javascript|js|typescript|react|node|nodejs|python|java|c\+\+|c#|golang|go|rust|php|sql|html|css|api|sdk|framework|library|regex|алгоритм|код|программир|разработк)\b/iu;
  const mathPattern =
    /\b(математ|алгебр|геометр|функци|производн|интеграл|логарифм|уравнен|вероятност|статистик|матриц|вектор|числ)\b|[=+\-*/^<>]{1,}/iu;

  return technicalPattern.test(normalized) || mathPattern.test(normalized);
}

function containsCyrillic(text) {
  return /[\u0400-\u04FF]/u.test(cleanText(text || ''));
}

function detectMessageLanguage(text) {
  const normalized = cleanText(text || '');
  if (!normalized) {
    return 'auto';
  }

  const cyrillicMatches = normalized.match(/[\u0400-\u04FF]/gu) || [];
  const latinMatches = normalized.match(/[A-Za-z]/g) || [];
  const cyrillicRatio = cyrillicMatches.length / Math.max(normalized.length, 1);
  const latinRatio = latinMatches.length / Math.max(normalized.length, 1);

  if (cyrillicRatio >= 0.12 && cyrillicMatches.length >= latinMatches.length * 0.7) {
    return 'ru';
  }
  if (latinRatio >= 0.12) {
    return 'en';
  }
  return 'auto';
}

function languageMismatchPenalty(userLanguage, text) {
  if (!text || userLanguage === 'auto') {
    return 0;
  }

  const replyLanguage = detectMessageLanguage(text);
  if (replyLanguage === 'auto' || replyLanguage === userLanguage) {
    return 0;
  }

  return 0.16;
}

function parsePreferredDomainsList(preferredDomainsRaw) {
  return String(preferredDomainsRaw || '')
    .split(',')
    .map((domain) => cleanText(domain).toLowerCase())
    .filter(Boolean)
    .map((domain) => domain.replace(/^https?:\/\//u, '').replace(/^www\./u, '').replace(/\/.*$/u, ''))
    .filter(Boolean);
}

function isPreferredDomainUrl(url, preferredDomains = []) {
  if (!url || !preferredDomains.length) {
    return false;
  }

  try {
    const host = new URL(url).host.toLowerCase().replace(/^www\./u, '');
    return preferredDomains.some((domain) => host === domain || host.endsWith(`.${domain}`));
  } catch (_error) {
    return false;
  }
}

function buildWebSearchQueries(userMessage, preferredDomainsRaw = '') {
  const normalized = cleanText(userMessage || '');
  if (!normalized) {
    return [];
  }

  const baseQueries = [normalized];
  if (containsCyrillic(normalized) && !/\b(на русском|русск)\b/iu.test(normalized)) {
    baseQueries.unshift(`${normalized} на русском`);
  }

  const preferredDomains = parsePreferredDomainsList(preferredDomainsRaw).slice(0, 3);
  const queries = [...baseQueries];
  preferredDomains.forEach((domain) => {
    baseQueries.forEach((query) => {
      queries.push(`${query} site:${domain}`);
    });
  });

  return Array.from(new Set(queries.map((query) => cleanText(query)).filter(Boolean)));
}

function collapseRepeatedWords(text) {
  const tokens = cleanText(text).split(/\s+/u).filter(Boolean);
  if (!tokens.length) {
    return '';
  }

  const result = [];
  let previous = '';
  let run = 0;

  tokens.forEach((token) => {
    const normalized = token.toLowerCase();
    if (normalized === previous) {
      run += 1;
      if (run > 2) {
        return;
      }
    } else {
      previous = normalized;
      run = 1;
    }
    result.push(token);
  });

  return cleanText(result.join(' '));
}

function collapseDuplicateSentences(text) {
  const sentences = splitIntoSentences(text);
  if (!sentences.length) {
    return '';
  }

  const seen = new Set();
  const unique = [];
  sentences.forEach((sentence) => {
    const normalized = cleanText(sentence).toLowerCase();
    if (!normalized || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    unique.push(sentence);
  });

  return cleanText(unique.join(' '));
}

function stripLinkNoise(text) {
  const value = cleanText(text)
    .replace(/([^\s])(\[[^\]]{1,120}\]\(https?:\/\/[^\s)]+\))/giu, '$1 $2')
    .replace(/(\))(?![\s.,!?;:])/gu, '$1 ');
  const linkCount = (value.match(MARKDOWN_LINK_PATTERN) || []).length;
  const urlCount = (value.match(URL_PATTERN) || []).length;
  let cleaned = value;

  if (linkCount >= 2 || urlCount >= 3) {
    cleaned = cleaned.replace(MARKDOWN_LINK_PATTERN, '$1');
    cleaned = cleaned.replace(URL_PATTERN, '');
  }

  return cleanText(cleaned);
}

function pruneLinkDenseFragments(text) {
  const normalized = cleanText(text);
  if (!normalized) {
    return '';
  }

  const sentences = splitIntoSentences(normalized);
  if (!sentences.length) {
    return normalized;
  }

  let totalLinkCount = 0;
  let totalUrlCount = 0;
  const selected = [];
  const selectedSet = new Set();

  sentences.forEach((sentence) => {
    const cleanSentence = cleanText(sentence);
    if (!cleanSentence) {
      return;
    }

    const linkCount = (cleanSentence.match(MARKDOWN_LINK_PATTERN) || []).length;
    const urlCount = (cleanSentence.match(URL_PATTERN) || []).length;
    totalLinkCount += linkCount;
    totalUrlCount += urlCount;

    if (linkCount >= 2 || urlCount >= 2) {
      return;
    }

    const normalizedSentence = cleanSentence.toLowerCase();
    if (selectedSet.has(normalizedSentence)) {
      return;
    }

    selectedSet.add(normalizedSentence);
    selected.push(cleanSentence);
  });

  if (!selected.length && (totalLinkCount > 0 || totalUrlCount > 0)) {
    return cleanText(
      normalized
        .replace(MARKDOWN_LINK_PATTERN, '$1')
        .replace(URL_PATTERN, '')
    );
  }

  return selected.length ? cleanText(selected.join(' ')) : normalized;
}

function clampReplySentenceCount(text, maxSentences = 0) {
  const normalized = cleanText(text);
  if (!normalized) {
    return '';
  }

  const limit = Math.max(Number(maxSentences) || 0, 0);
  if (!limit) {
    return normalized;
  }

  const selected = [];
  appendUniqueSentences(selected, splitIntoSentences(normalized), limit);
  return cleanText(selected.join(' ')) || normalized;
}

function normalizeFinalReplyText(text) {
  const base = sanitizeReplyText(text);
  const withoutLinkNoise = stripLinkNoise(base);
  const withoutDenseLinkFragments = pruneLinkDenseFragments(withoutLinkNoise);
  const withoutRepeatedWords = collapseRepeatedWords(withoutDenseLinkFragments);
  const withoutRepeatedSentences = collapseDuplicateSentences(withoutRepeatedWords);
  return cleanText(
    withoutRepeatedSentences ||
    withoutRepeatedWords ||
    withoutDenseLinkFragments ||
    withoutLinkNoise ||
    base
  );
}

function splitReplyIntoCodeBlocks(text = '') {
  const value = String(text || '').replace(/\r\n/g, '\n').trim();
  if (!value) {
    return [];
  }

  const blocks = [];
  const pattern = /```([\w#+.-]*)\n([\s\S]*?)```/gu;
  let lastIndex = 0;
  let match = pattern.exec(value);

  while (match) {
    if (match.index > lastIndex) {
      blocks.push({
        type: 'text',
        content: value.slice(lastIndex, match.index),
      });
    }

    blocks.push({
      type: 'code',
      language: cleanText(match[1] || '').toLowerCase(),
      content: String(match[2] || '').replace(/\n+$/u, ''),
    });

    lastIndex = pattern.lastIndex;
    match = pattern.exec(value);
  }

  if (lastIndex < value.length) {
    blocks.push({
      type: 'text',
      content: value.slice(lastIndex),
    });
  }

  return blocks.filter((block) => cleanText(block.content || '') || String(block.content || '').trim());
}

function renderMarkdownCodeBlock(language, code) {
  const normalizedCode = String(code || '').replace(/\r\n/g, '\n').trim();
  if (!normalizedCode) {
    return '';
  }

  return `\`\`\`${cleanText(language || '').toLowerCase()}\n${normalizedCode}\n\`\`\``.trim();
}

function finalizeCodeAwareReply(text, maxSentences = 0) {
  const blocks = splitReplyIntoCodeBlocks(text);
  if (!blocks.some((block) => block.type === 'code')) {
    return '';
  }

  let remainingSentences = Math.max(Number(maxSentences) || 0, 0);
  const unlimited = remainingSentences === 0;
  const resultBlocks = [];

  blocks.forEach((block) => {
    if (block.type === 'code') {
      const renderedCode = renderMarkdownCodeBlock(block.language, block.content);
      if (renderedCode) {
        resultBlocks.push(renderedCode);
      }
      return;
    }

    const normalizedText = normalizeFinalReplyText(block.content);
    if (!normalizedText) {
      return;
    }

    if (unlimited) {
      resultBlocks.push(normalizedText);
      return;
    }

    const sentences = splitIntoSentences(normalizedText);
    if (!sentences.length || remainingSentences <= 0) {
      return;
    }

    const selected = sentences.slice(0, remainingSentences);
    if (!selected.length) {
      return;
    }

    resultBlocks.push(cleanText(selected.join(' ')));
    remainingSentences -= selected.length;
  });

  return resultBlocks.join('\n\n').trim();
}

function looksLikeCodeLine(line = '') {
  const value = String(line || '').trim();
  if (!value) {
    return false;
  }
  if (/^(function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+|if\s*\(|for\s*\(|while\s*\(|return\b|class\s+\w+|import\s+|export\s+|<[/a-z!])/iu.test(value)) {
    return true;
  }
  if (/[{};<>]/u.test(value) && /(=|\(|\)|=>|<\/?[a-z])/iu.test(value)) {
    return true;
  }
  return false;
}

function looksLikeUnfencedCodeBlock(text = '') {
  const value = String(text || '').replace(/\r\n/g, '\n').trim();
  if (!value || /```/u.test(value)) {
    return false;
  }
  const lines = value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    return false;
  }
  const codeLikeCount = lines.filter(looksLikeCodeLine).length;
  const ratio = codeLikeCount / lines.length;
  return codeLikeCount >= 2 && ratio >= 0.5;
}

function enforceCodeBlocksForCodeLikeReply(text = '', userMessage = '') {
  const value = String(text || '').trim();
  if (!value || /```/u.test(value)) {
    return value;
  }
  const userAskedCode = /\b(code|function|javascript|js|typescript|python|sql|html|css|код|функц)\b/iu.test(cleanText(userMessage || ''));
  if (!userAskedCode && !looksLikeUnfencedCodeBlock(value)) {
    return value;
  }
  if (!looksLikeUnfencedCodeBlock(value)) {
    return value;
  }
  const language = detectCodeLanguage(value, '');
  return renderMarkdownCodeBlock(language, value);
}

function findClampBoundary(text, limit) {
  const candidates = [];
  const punctuationPattern = /[.!?](?=\s|$)/gu;
  const paragraphPattern = /\n{2,}/gu;
  const linePattern = /\n/gu;
  const whitespacePattern = /\s/gu;

  [paragraphPattern, punctuationPattern, linePattern, whitespacePattern].forEach((pattern) => {
    let match = pattern.exec(text);
    while (match) {
      const boundary = match.index + match[0].length;
      if (boundary <= limit) {
        candidates.push(boundary);
      }
      match = pattern.exec(text);
    }
  });

  return candidates.length ? Math.max(...candidates) : -1;
}

function clampReplyCharacters(text, maxCharacters = 0) {
  const normalized = String(text || '').trim();
  const limit = Math.max(Number(maxCharacters) || 0, 0);
  if (!normalized || !limit || normalized.length <= limit) {
    return {
      content: normalized,
      truncated: false,
    };
  }

  const codeBlocks = splitReplyIntoCodeBlocks(normalized);
  if (codeBlocks.some((block) => block.type === 'code')) {
    const buffer = Math.min(Math.max(Math.floor(limit * 0.18), 180), 2400);
    const target = limit + buffer;
    const collected = [];
    let totalLength = 0;

    for (const block of codeBlocks) {
      const rendered = block.type === 'code'
        ? renderMarkdownCodeBlock(block.language, block.content)
        : normalizeFinalReplyText(block.content);
      if (!rendered) {
        continue;
      }

      const separatorLength = collected.length ? 2 : 0;
      const projectedLength = totalLength + separatorLength + rendered.length;
      if (projectedLength <= target || !collected.length) {
        collected.push(rendered);
        totalLength = projectedLength;
        continue;
      }

      break;
    }

    const content = collected.join('\n\n').trim();
    return {
      content,
      truncated: content.length < normalized.length,
    };
  }

  const boundary = findClampBoundary(normalized, limit);
  if (boundary >= Math.floor(limit * 0.55)) {
    return {
      content: normalized.slice(0, boundary).trim(),
      truncated: true,
    };
  }

  return {
    content: normalized.slice(0, limit).trim(),
    truncated: true,
  };
}

function detectCodeLanguage(code = '', hint = '') {
  const normalizedHint = cleanText(hint).toLowerCase();
  if (normalizedHint) {
    return normalizedHint;
  }

  const value = String(code || '');
  if (!value.trim()) {
    return 'text';
  }

  if (/^\s*[\[{]/u.test(value) && /"\s*:\s*/u.test(value)) {
    return 'json';
  }
  if (/<[a-z][\s\S]*>/iu.test(value)) {
    return 'html';
  }
  if (/\b(function|const|let|var|=>|document\.|console\.)\b/u.test(value)) {
    return 'javascript';
  }
  if (/\b(def |import |print\(|self\b|None\b|True\b|False\b)\b/u.test(value)) {
    return 'python';
  }
  if (/\b(SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|JOIN)\b/u.test(value)) {
    return 'sql';
  }
  if (/\b(display\s*:|position\s*:|color\s*:|background\s*:)\b/u.test(value)) {
    return 'css';
  }
  if (/^\s*(npm|yarn|pnpm|git|cd|ls|mkdir|curl|wget)\b/mu.test(value)) {
    return 'bash';
  }

  return 'text';
}

function decodeStructuredFieldValue(value = '') {
  const normalized = String(value || '').trim();
  if (!normalized) {
    return '';
  }

  if (normalized.startsWith('"') && normalized.endsWith('"')) {
    try {
      return JSON.parse(normalized);
    } catch (_error) {
      // Fallback handled below.
    }
  }

  let unwrapped = normalized;
  if (
    (normalized.startsWith('"') && normalized.endsWith('"')) ||
    (normalized.startsWith('\'') && normalized.endsWith('\''))
  ) {
    unwrapped = normalized.slice(1, -1);
  }

  return unwrapped
    .replace(/\\\\/gu, '\\')
    .replace(/\\n/gu, '\n')
    .replace(/\\r/gu, '')
    .replace(/\\t/gu, '  ')
    .replace(/\\"/gu, '"')
    .replace(/\\'/gu, '\'')
    .trim();
}

function extractTopLevelJsonFragments(text = '') {
  const value = String(text || '').trim();
  if (!value) {
    return [];
  }

  const fragments = [];
  let startIndex = -1;
  let depth = 0;
  let inString = false;
  let isEscaped = false;

  for (let index = 0; index < value.length; index += 1) {
    const symbol = value[index];
    if (inString) {
      if (isEscaped) {
        isEscaped = false;
        continue;
      }
      if (symbol === '\\') {
        isEscaped = true;
        continue;
      }
      if (symbol === '"') {
        inString = false;
      }
      continue;
    }

    if (symbol === '"') {
      inString = true;
      continue;
    }

    if (symbol === '{' || symbol === '[') {
      if (depth === 0) {
        startIndex = index;
      }
      depth += 1;
      continue;
    }

    if (symbol === '}' || symbol === ']') {
      depth = Math.max(0, depth - 1);
      if (depth === 0 && startIndex >= 0) {
        fragments.push(value.slice(startIndex, index + 1));
        startIndex = -1;
      }
    }
  }

  return fragments;
}

function collectStructuredReplyParts(payload, target = []) {
  if (Array.isArray(payload)) {
    payload.forEach((item) => collectStructuredReplyParts(item, target));
    return target;
  }

  if (!payload || typeof payload !== 'object') {
    return target;
  }

  const title = cleanText(payload.title || payload.heading || payload.name || '');
  if (title) {
    target.push({
      type: 'text',
      content: title,
    });
  }

  const textContent = cleanText(
    payload.text ||
    payload.comment ||
    payload.description ||
    payload.explanation ||
    payload.summary ||
    payload.content ||
    ''
  );
  if (textContent) {
    target.push({
      type: 'text',
      content: textContent,
    });
  }

  const codeContent = String(payload.code || payload.snippet || payload.example || '').replace(/\r\n/g, '\n').trim();
  if (codeContent) {
    target.push({
      type: 'code',
      language: detectCodeLanguage(codeContent, payload.language || payload.lang || payload.syntax || ''),
      content: codeContent,
    });
  }

  return target;
}

function collectLooseStructuredReplyParts(rawText, target = []) {
  const raw = String(rawText || '');
  if (!raw) {
    return target;
  }

  let pendingLanguage = '';
  let match = LOOSE_STRUCTURED_FIELD_PATTERN.exec(raw);
  while (match) {
    const fieldName = cleanText(match[1] || '').toLowerCase();
    const fieldValue = decodeStructuredFieldValue(match[2] || '');

    if (STRUCTURED_LANGUAGE_FIELDS.has(fieldName)) {
      pendingLanguage = cleanText(fieldValue).toLowerCase();
    } else if (STRUCTURED_CODE_FIELDS.has(fieldName)) {
      const codeValue = String(fieldValue || '').replace(/\r\n/g, '\n').trim();
      if (codeValue) {
        target.push({
          type: 'code',
          language: detectCodeLanguage(codeValue, pendingLanguage),
          content: codeValue,
        });
      }
      pendingLanguage = '';
    } else if (STRUCTURED_TEXT_FIELDS.has(fieldName)) {
      const textValue = cleanText(fieldValue || '');
      if (textValue) {
        target.push({
          type: 'text',
          content: textValue,
        });
      }
    }

    match = LOOSE_STRUCTURED_FIELD_PATTERN.exec(raw);
  }
  LOOSE_STRUCTURED_FIELD_PATTERN.lastIndex = 0;
  return target;
}

function convertStructuredReplyToMarkdown(text) {
  const raw = String(text || '').trim();
  if (!raw) {
    return '';
  }

  if (/```/u.test(raw)) {
    return raw;
  }

  if (!/"(?:text|code|snippet|example|language|lang)"/u.test(raw)) {
    return cleanText(raw);
  }

  const candidates = [];
  if (/^[\[{]/u.test(raw)) {
    candidates.push(raw);
  }
  candidates.push(...extractTopLevelJsonFragments(raw));

  const parts = [];
  candidates.forEach((candidate) => {
    try {
      const parsed = JSON.parse(candidate);
      collectStructuredReplyParts(parsed, parts);
    } catch (_error) {
      // Ignore malformed JSON fragments and continue with the next candidate.
    }
  });

  if (!parts.length) {
    collectLooseStructuredReplyParts(raw, parts);
  }

  if (!parts.length) {
    return cleanText(raw);
  }

  const renderedParts = [];
  const seen = new Set();
  parts.forEach((part) => {
    const rendered = part.type === 'code'
      ? renderMarkdownCodeBlock(part.language, part.content)
      : cleanText(part.content);
    if (!rendered) {
      return;
    }

    const signature = `${part.type}:${rendered}`;
    if (seen.has(signature)) {
      return;
    }

    seen.add(signature);
    renderedParts.push(rendered);
  });

  return renderedParts.join('\n\n').trim();
}

function finalizeReplyForOutput(text, options = {}) {
  const maxSentences = Math.max(Number(options.maxSentences) || 0, 0);
  const normalizedMarkup = convertStructuredReplyToMarkdown(text);
  const codeEnforcedMarkup = enforceCodeBlocksForCodeLikeReply(
    normalizedMarkup,
    options.userMessage || ''
  );
  const codeAwareReply = finalizeCodeAwareReply(codeEnforcedMarkup, maxSentences);
  if (codeAwareReply) {
    return codeAwareReply;
  }

  const normalized = normalizeFinalReplyText(codeEnforcedMarkup);
  const sentenceClamped = clampReplySentenceCount(normalized, maxSentences);
  if (sentenceClamped) {
    return sentenceClamped;
  }
  if (normalized) {
    return normalized;
  }

  return 'Не удалось сформировать корректный ответ. Уточните запрос и попробуйте снова.';
}

function isNoisyWebChunk(chunk = {}) {
  const url = cleanText(chunk.url || chunk.ownerId || '').toLowerCase();
  if (NOISY_WEB_HOST_PATTERN.test(url)) {
    return true;
  }

  const text = cleanText(chunk.text || chunk.content || '');
  if (!text) {
    return true;
  }

  const markdownLinkCount = (text.match(MARKDOWN_LINK_PATTERN) || []).length;
  const urlCount = (text.match(URL_PATTERN) || []).length;
  const tokens = tokenizeWords(text);
  if (!tokens.length) {
    return true;
  }

  const uniqueRatio = new Set(tokens.map((token) => token.toLowerCase())).size / tokens.length;
  if ((markdownLinkCount || urlCount) && tokens.length < 18) {
    return true;
  }
  if ((markdownLinkCount > 0 || urlCount > 0) && /\b(перевод|словарь|синоним|translation|dictionary)\b/iu.test(text)) {
    return true;
  }
  if (markdownLinkCount >= 3 || urlCount >= 6) {
    return true;
  }

  return tokens.length > 24 && uniqueRatio < 0.22;
}

function isNoisyKnowledgeArtifactText(text = '') {
  const normalized = cleanText(text || '');
  if (!normalized) {
    return true;
  }

  if (ARTIFACT_NOISE_PATTERN.test(normalized)) {
    return true;
  }

  const tokenCount = tokenizeWords(normalized).length;
  const linkCount = (normalized.match(MARKDOWN_LINK_PATTERN) || []).length;
  const urlCount = (normalized.match(URL_PATTERN) || []).length;
  if (tokenCount < 6) {
    return false;
  }

  if (linkCount >= 2 || urlCount >= 3) {
    return true;
  }

  if (
    tokenCount > 72 &&
    /(?:old_contents|new_contents|subject|message|lang|repos)\s*:/iu.test(normalized)
  ) {
    return true;
  }

  return false;
}

function isEchoReply(userText, replyText) {
  const userTokens = new Set(tokenizeWords(userText));
  const replyTokens = tokenizeWords(replyText);
  if (!userTokens.size || !replyTokens.length) {
    return false;
  }

  let shared = 0;
  replyTokens.forEach((token) => {
    if (userTokens.has(token)) {
      shared += 1;
    }
  });

  return shared / Math.max(replyTokens.length, userTokens.size) >= 0.82;
}

function computePromptReplyRelevance(promptText, replyText) {
  const promptTokens = extractQueryTokens(promptText || '', { fallbackToAll: true });
  const replyTokens = extractQueryTokens(replyText || '', { fallbackToAll: true });
  if (!promptTokens.length || !replyTokens.length) {
    return 0;
  }

  const promptSet = new Set(promptTokens);
  const replySet = new Set(replyTokens);
  let shared = 0;
  promptSet.forEach((token) => {
    if (replySet.has(token)) {
      shared += 1;
    }
  });

  if (!shared) {
    return 0;
  }

  const precision = shared / Math.max(replySet.size, 1);
  const recall = shared / Math.max(promptSet.size, 1);
  const harmonic = (2 * precision * recall) / Math.max(precision + recall, 1e-9);
  const coverage = shared / Math.max(1, Math.min(promptSet.size, 4));
  return Math.min(1, harmonic * 0.78 + coverage * 0.22);
}

function createChunksForDocument(documentEntry, chunkSize, chunkOverlap, allowedTerms) {
  const sentences = splitIntoSentences(documentEntry.text);
  const sentenceWords = sentences.map((sentence) => tokenizeWords(sentence));
  const chunks = [];

  if (!sentences.length) {
    return chunks;
  }

  let start = 0;
  while (start < sentences.length) {
    let end = start;
    let wordCount = 0;

    while (end < sentences.length && wordCount < chunkSize) {
      wordCount += sentenceWords[end].length;
      end += 1;
    }

    const selectedSentences = sentences.slice(start, end);
    const chunkText = cleanText(selectedSentences.join(' '));
    if (!chunkText || startsWithSpeakerLabel(chunkText)) {
      if (end >= sentences.length) {
        break;
      }
      start = end;
      continue;
    }

    const chunkTokens = tokenizeWords(chunkText);
    const termMap = createTermMap(chunkTokens, allowedTerms);

    chunks.push({
      id: createId('chunk'),
      sourceId: documentEntry.sourceId,
      sourceType: documentEntry.sourceType,
      sourceKind: documentEntry.sourceKind,
      label: documentEntry.label,
      text: chunkText,
      sentences: selectedSentences,
      tokenCount: chunkTokens.length,
      documentLength: chunkTokens.length,
      termMap,
    });

    if (end >= sentences.length) {
      break;
    }

    let overlapWords = 0;
    let overlapSentenceCount = 0;
    for (let index = end - 1; index > start && overlapWords < chunkOverlap; index -= 1) {
      overlapWords += sentenceWords[index].length;
      overlapSentenceCount += 1;
    }

    start = Math.max(start + 1, end - overlapSentenceCount);
  }

  return chunks;
}

function buildSourceDocuments(source) {
  return [
    {
      sourceId: source.id,
      sourceType: source.type,
      sourceKind: 'knowledge',
      label: source.label,
      text: source.content,
    },
  ];
}

function normalizeStoredChunks(chunks = []) {
  return chunks
    .map((chunk, index) => ({
      id: chunk.id || `knowledge_chunk_${index}`,
      sourceId: chunk.sourceId || `knowledge_chunk_${index}`,
      sourceType: chunk.sourceType || 'knowledge',
      sourceKind: chunk.sourceKind || 'knowledge',
      label: cleanText(chunk.label || '') || 'Архив знаний',
      text: cleanText(chunk.text || ''),
    }))
    .filter((entry) => Boolean(entry.text));
}

function dedupeEntries(entries, getKey) {
  const seen = new Set();
  const result = [];

  entries.forEach((entry) => {
    const key = getKey(entry);
    if (!key || seen.has(key)) {
      return;
    }

    seen.add(key);
    result.push(entry);
  });

  return result;
}

function normalizeStoredReplyMemories(entries = []) {
  return entries
    .map((entry, index) => {
      const promptText = cleanText(entry.promptText || '');
      const responseText = sanitizeReplyText(entry.responseText || '');
      if (!promptText || !responseText) {
        return null;
      }

      return {
        id: entry.id || createId(`archive_pair_${index}`),
        ownerId: entry.ownerId || 'knowledge',
        title: cleanText(entry.title || '') || 'Архив знаний',
        origin: entry.origin || 'knowledge',
        score: Number(entry.score || 0.78),
        promptText,
        responseText,
        combinedText: cleanText(
          entry.combinedText || `Пользователь: ${promptText}\nАссистент: ${responseText}`
        ),
      };
    })
    .filter(Boolean);
}

function appendHash(hash, value) {
  hash.update(String(value || ''));
  hash.update('\u001f');
}

function createCorpusSignature(state, artifacts) {
  const hash = crypto.createHash('sha1');
  appendHash(hash, artifacts.sourceDocumentCount || artifacts.sourceDocuments?.length || 0);
  appendHash(hash, artifacts.replyPairCount || artifacts.replyMemories?.length || 0);
  appendHash(hash, artifacts.tokenCount || 0);
  appendHash(hash, artifacts.positiveFeedbackCount || 0);
  appendHash(hash, artifacts.negativeFeedbackCount || 0);

  (artifacts.sourceDocuments || []).forEach((documentEntry) => {
    appendHash(hash, documentEntry.label);
    appendHash(hash, documentEntry.text);
  });

  (artifacts.replyMemories || []).forEach((pair) => {
    appendHash(hash, pair.promptText);
    appendHash(hash, pair.responseText);
    appendHash(hash, pair.score);
  });

  const settingsSnapshot = {
    ...buildTrainingSettingsSnapshot(state.settings.training),
    chunkSize: state.settings.training.chunkSize,
    chunkOverlap: state.settings.training.chunkOverlap,
    vocabularyLimit: state.settings.training.vocabularyLimit,
    topKChunks: state.settings.training.topKChunks,
    minSimilarity: state.settings.training.minSimilarity,
  };

  appendHash(hash, JSON.stringify(settingsSnapshot));
  appendHash(hash, JSON.stringify(getRuntimeConfig()));
  return hash.digest('hex');
}

function buildBm25Index(entries) {
  const documentFrequency = {};
  let totalLength = 0;

  entries.forEach((entry) => {
    totalLength += entry.documentLength || 0;
    Object.keys(entry.termMap).forEach((term) => {
      documentFrequency[term] = (documentFrequency[term] || 0) + 1;
    });
  });

  const documentCount = entries.length;
  const averageDocumentLength = documentCount ? totalLength / documentCount : 0;
  const idf = {};

  Object.entries(documentFrequency).forEach(([term, frequency]) => {
    idf[term] = Math.log(1 + (documentCount - frequency + 0.5) / (frequency + 0.5));
  });

  return {
    documentCount,
    averageDocumentLength,
    documentFrequency,
    idf,
  };
}

function computeBm25Score(queryMap, entry, bm25, runtimeConfig) {
  const k1 = runtimeConfig.retrieval.bm25k1;
  const b = runtimeConfig.retrieval.bm25b;
  const avgLength = Math.max(bm25.averageDocumentLength || 0, 1);
  let score = 0;

  Object.entries(queryMap).forEach(([term, queryWeight]) => {
    const tf = entry.termMap[term] || 0;
    if (!tf) {
      return;
    }

    const idf = bm25.idf[term] || 0;
    const denominator = tf + k1 * (1 - b + b * ((entry.documentLength || 0) / avgLength));
    score += queryWeight * idf * ((tf * (k1 + 1)) / denominator);
  });

  return score;
}

function scoreSentenceAgainstQuery(sentence, queryMap) {
  const tokens = tokenizeWords(sentence);
  let score = 0;
  tokens.forEach((token) => {
    score += queryMap[token] || 0;
  });
  if (sentence.trim().endsWith('?')) {
    score *= 0.7;
  }
  return score;
}

function isAllowedKnowledgeSentence(sentence) {
  const cleanSentence = sanitizeReplyText(sentence);
  const tokens = tokenizeWords(cleanSentence);

  if (!cleanSentence || startsWithSpeakerLabel(cleanSentence)) {
    return false;
  }

  if (tokens.length < 6) {
    return false;
  }

  if (/\?\s*$/u.test(cleanSentence)) {
    return false;
  }

  if (/^(ответить|reply|главная|меню|голосовать|результаты|сезон\s+\d+)/iu.test(cleanSentence)) {
    return false;
  }

  return true;
}

function selectKnowledgeSentences(chunks, queryMap, maxSentences) {
  const candidates = [];
  const used = new Set();

  chunks.forEach((chunk) => {
    chunk.sentences.forEach((sentence) => {
      const cleanSentence = sanitizeReplyText(sentence);
      if (!isAllowedKnowledgeSentence(cleanSentence)) {
        return;
      }

      const normalized = cleanSentence.toLowerCase();
      if (used.has(normalized)) {
        return;
      }

      const score = scoreSentenceAgainstQuery(cleanSentence, queryMap) + (chunk.score || 0);
      if (score <= 0) {
        return;
      }

      used.add(normalized);
      candidates.push({ sentence: cleanSentence, score });
    });
  });

  return candidates
    .sort((left, right) => right.score - left.score)
    .slice(0, maxSentences)
    .map((candidate) => candidate.sentence);
}

function appendUniqueSentences(target, sentences, limit) {
  const seen = new Set(target.map((sentence) => sentence.toLowerCase()));

  sentences.forEach((sentence) => {
    const cleanSentence = sanitizeReplyText(sentence);
    const normalized = cleanSentence.toLowerCase();
    if (!cleanSentence || seen.has(normalized) || target.length >= limit) {
      return;
    }

    seen.add(normalized);
    target.push(cleanSentence);
  });

  return target;
}

function buildHybridReply({
  bestReply,
  neuralReply,
  knowledgeSentences,
  maxSentences,
}) {
  const selected = [];

  appendUniqueSentences(selected, knowledgeSentences, maxSentences);

  if (neuralReply) {
    appendUniqueSentences(
      selected,
      splitIntoSentences(neuralReply).slice(0, 2),
      maxSentences
    );
  }

  if ((bestReply?.score || 0) >= 0.32 && bestReply?.responseText) {
    appendUniqueSentences(
      selected,
      splitIntoSentences(bestReply.responseText).slice(0, 2),
      maxSentences
    );
  }

  return cleanText(selected.join(' '));
}

function buildGroundedFallback(chunkCandidates, maxSentences) {
  const selected = [];

  chunkCandidates.forEach((chunk) => {
    appendUniqueSentences(
      selected,
      chunk.sentences.filter((sentence) => isAllowedKnowledgeSentence(sentence)),
      maxSentences
    );
  });

  if (!selected.length && chunkCandidates[0]?.text) {
    appendUniqueSentences(
      selected,
      splitIntoSentences(chunkCandidates[0].text),
      maxSentences
    );
  }

  return cleanText(selected.join(' '));
}

function createExternalKnowledgeChunk(entry) {
  const text = cleanText(entry.content || entry.snippet || '').slice(0, 18000);
  const tokens = tokenizeWords(text);
  return {
    id: createId('web'),
    ownerId: entry.url,
    label: entry.title || entry.host || entry.url,
    text,
    sentences: splitIntoSentences(text),
    documentLength: tokens.length,
    termMap: createTermMap(tokens),
    score: 0,
    sourceType: 'web',
    url: entry.url,
  };
}

function extractHostLabel(url = '') {
  try {
    return new URL(url).host.replace(/^www\./u, '');
  } catch (_error) {
    return '';
  }
}

function buildReplyReferences(chunkCandidates = []) {
  const references = [];
  const seen = new Set();

  chunkCandidates.forEach((chunk) => {
    const key = cleanText(chunk.url || `${chunk.sourceType || 'knowledge'}:${chunk.label || chunk.ownerId || chunk.id}`);
    if (!key || seen.has(key) || references.length >= 4) {
      return;
    }

    seen.add(key);
    const excerptSource = chunk.sentences?.[0] || chunk.text || '';
    const excerpt = clampReplyCharacters(normalizeFinalReplyText(excerptSource), 260).content;

    references.push({
      type: chunk.sourceType === 'web' ? 'web' : 'knowledge',
      title: cleanText(chunk.label || chunk.ownerId || chunk.url || 'Источник'),
      url: chunk.url || null,
      host: extractHostLabel(chunk.url || ''),
      excerpt,
      score: Number((chunk.score || 0).toFixed(3)),
    });
  });

  return references;
}

function buildReplyPayload(rawContent, options = {}) {
  const {
    maxSentences = 0,
    maxCharacters = 0,
    userMessage = '',
    metadata = {},
    references = [],
  } = options;

  const finalized = finalizeReplyForOutput(rawContent, {
    maxSentences,
    userMessage,
  });
  const limited = clampReplyCharacters(finalized, maxCharacters);

  return {
    content: limited.content,
    metadata: {
      ...metadata,
      references,
      contentTruncated: limited.truncated,
      contentLength: limited.content.length,
      hasCodeBlocks: /```/u.test(limited.content),
    },
  };
}

async function fetchWebKnowledgeCandidates({
  userMessage,
  queryMap,
  runtimeConfig,
  preferGroundedKnowledge = false,
}) {
  if (!runtimeConfig.generation?.webSearchEnabled) {
    return [];
  }

  const maxResults = Math.max(
    1,
    (Number(runtimeConfig.generation.webSearchMaxResults) || 4) + (preferGroundedKnowledge ? 2 : 0)
  );
  const fetchPages = Math.max(
    1,
    (Number(runtimeConfig.generation.webSearchFetchPages) || 2) + (preferGroundedKnowledge ? 1 : 0)
  );
  const timeoutMs = Math.max(2000, Number(runtimeConfig.generation.webSearchTimeoutMs) || 12000);
  const preferredDomains = runtimeConfig.generation.webSearchPreferredDomains || '';
  const searchMaxResults = preferredDomains ? Math.max(maxResults, 8) : maxResults;
  const preferCyrillicSources = containsCyrillic(userMessage);
  const preferredDomainList = parsePreferredDomainsList(preferredDomains);
  const searchQueries = buildWebSearchQueries(userMessage, preferredDomains);
  const documents = [];
  const seenUrls = new Set();
  const maxCollectedDocuments = Math.max(searchMaxResults, maxResults);
  let hasPreferredDomainHit = false;

  for (const query of searchQueries) {
    if (
      documents.length >= maxCollectedDocuments &&
      (!preferredDomainList.length || hasPreferredDomainHit)
    ) {
      break;
    }

    const fetched = await fetchSearchDocuments(query, {
      maxResults: searchMaxResults,
      fetchPages,
      timeoutMs,
      preferredDomains,
    });

    fetched.forEach((entry) => {
      if (!entry?.url || seenUrls.has(entry.url) || documents.length >= maxCollectedDocuments) {
        return;
      }
      seenUrls.add(entry.url);
      documents.push(entry);
      if (isPreferredDomainUrl(entry.url, preferredDomainList)) {
        hasPreferredDomainHit = true;
      }
    });
  }

  if (!documents.length) {
    return [];
  }

  const rankedDocuments = preferredDomainList.length
    ? [...documents].sort((left, right) => {
      const leftPreferred = isPreferredDomainUrl(left.url, preferredDomainList) ? 1 : 0;
      const rightPreferred = isPreferredDomainUrl(right.url, preferredDomainList) ? 1 : 0;
      return rightPreferred - leftPreferred;
    })
    : documents;

  const mappedChunks = rankedDocuments
    .slice(0, maxResults)
    .map(createExternalKnowledgeChunk)
    .filter((chunk) => !isNoisyWebChunk(chunk))
    .filter((chunk) => Boolean(chunk.text));
  if (!mappedChunks.length) {
    return [];
  }

  const bm25 = buildBm25Index(mappedChunks);
  const scoredChunks = mappedChunks
    .map((chunk) => {
      const topicalRelevance = computePromptReplyRelevance(userMessage, chunk.text);
      const languageBonus = preferCyrillicSources
        ? (containsCyrillic(chunk.text) ? 0.25 : -0.18)
        : 0;
      return {
        ...chunk,
        topicalRelevance,
        score:
          computeBm25Score(queryMap, chunk, bm25, runtimeConfig) +
          topicalRelevance * 0.55 +
          languageBonus,
      };
    })
    .filter((chunk) => chunk.score > 0)
    .sort((left, right) => right.score - left.score);
  if (!scoredChunks.length) {
    return [];
  }

  const languagePreferredChunks = preferCyrillicSources
    ? scoredChunks.filter((chunk) => containsCyrillic(chunk.text))
    : [];
  const candidatePool = languagePreferredChunks.length ? languagePreferredChunks : scoredChunks;

  const topicalGate = preferGroundedKnowledge ? 0.14 : 0.08;
  const filteredChunks = candidatePool.filter((chunk) => chunk.topicalRelevance >= topicalGate);
  const candidateChunks = filteredChunks.length ? filteredChunks : candidatePool.slice(0, 1);

  return candidateChunks.slice(0, fetchPages);
}

function isPositiveMessageFeedback(metadata) {
  if (!metadata) {
    return false;
  }

  return metadata.userRating === 1;
}

function isNegativeMessageFeedback(metadata) {
  return metadata?.userRating === -1;
}

function extractDialoguePairsFromChat(chat) {
  const pairs = [];

  for (let index = 1; index < chat.messages.length; index += 1) {
    const currentMessage = chat.messages[index];
    const previousMessage = chat.messages[index - 1];
    if (previousMessage.role !== 'user' || currentMessage.role !== 'assistant') {
      continue;
    }

    if (currentMessage.metadata?.type === 'system') {
      continue;
    }

    if (!isPositiveMessageFeedback(currentMessage.metadata)) {
      continue;
    }

    pairs.push({
      id: createId('pair'),
      ownerId: chat.id,
      title: chat.title,
      origin: 'chat_feedback',
      score: currentMessage.metadata?.userRating === 1
        ? Math.max(0.86, Math.min(0.97, Number(currentMessage.metadata?.selfScore || 0.82) + 0.08))
        : Math.max(0.15, Math.min(0.94, Number(currentMessage.metadata?.selfScore || 0.7))),
      promptText: cleanText(previousMessage.content),
      responseText: cleanText(currentMessage.content),
      combinedText: cleanText(`Пользователь: ${previousMessage.content}\nАссистент: ${currentMessage.content}`),
    });
  }

  return pairs;
}

function collectBlockedReplies(state) {
  const blocked = new Set();
  state.chats.forEach((chat) => {
    chat.messages.forEach((message) => {
      if (message.role !== 'assistant') {
        return;
      }
      if (isNegativeMessageFeedback(message.metadata)) {
        blocked.add(cleanText(message.content).toLowerCase());
      }
    });
  });
  return blocked;
}

function buildTrainingTexts({
  sourceDocuments,
  sourceReplyMemories,
  replayReplyMemories,
  runtimeConfig,
}, options = {}) {
  const texts = [];
  const { shouldStop, onProgress } = options;
  const weightedReplayCount = replayReplyMemories.reduce((sum, pair) => (
    sum + (pair.origin === 'chat_feedback'
      ? Math.max(1, runtimeConfig.feedback.positiveReplayWeight)
      : 1)
  ), 0);
  const totalItems = Math.max(
    sourceDocuments.length + sourceReplyMemories.length + weightedReplayCount,
    1
  );
  let processedItems = 0;

  const reportProgress = (phase) => {
    if (typeof onProgress !== 'function') {
      return;
    }

    onProgress({
      phase,
      processedItems,
      totalItems,
      percent: Number(((processedItems / totalItems) * 100).toFixed(1)),
    });
  };

  sourceDocuments.forEach((documentEntry) => {
    throwIfTrainingPreparationStopped(shouldStop);
    const cleanSourceText = cleanText(documentEntry.text);
    if (cleanSourceText) {
      texts.push(structuredTrainingText(['__bos__', cleanSourceText, '__eos__']));
    }
    processedItems += 1;
    reportProgress('source_documents');
  });

  sourceReplyMemories.forEach((pair) => {
    throwIfTrainingPreparationStopped(shouldStop);
    texts.push(
      structuredTrainingText([
        '__bos__',
        '__usr__',
        pair.promptText,
        '__sep__',
        '__asst__',
        pair.responseText,
        '__eos__',
      ])
    );
    processedItems += 1;
    reportProgress('source_dialogues');
  });

  replayReplyMemories.forEach((pair) => {
    const weight = pair.origin === 'chat_feedback'
      ? Math.max(1, runtimeConfig.feedback.positiveReplayWeight)
      : 1;
    for (let repeatIndex = 0; repeatIndex < weight; repeatIndex += 1) {
      throwIfTrainingPreparationStopped(shouldStop);
      texts.push(
        structuredTrainingText([
          '__bos__',
          '__usr__',
          pair.promptText,
          '__sep__',
          '__asst__',
          pair.responseText,
          '__eos__',
        ])
      );
      processedItems += 1;
      reportProgress('replay_memories');
    }
  });

  return texts.filter(Boolean);
}

function createTrainingCorpusSignature(state, artifacts, options = {}) {
  const hash = crypto.createHash('sha1');
  const { shouldStop } = options;
  appendHash(hash, artifacts.sourceDocuments.length);
  appendHash(hash, artifacts.sourceReplyMemories.length);
  appendHash(hash, artifacts.replayReplyMemories.length);
  appendHash(hash, artifacts.trainingTexts.length);
  appendHash(hash, (artifacts.datasetFiles || []).length);
  appendHash(hash, artifacts.positiveFeedbackCount || 0);
  appendHash(hash, artifacts.negativeFeedbackCount || 0);

  artifacts.sourceDocuments.forEach((documentEntry) => {
    throwIfTrainingPreparationStopped(shouldStop);
    appendHash(hash, documentEntry.label);
    appendHash(hash, documentEntry.text);
  });

  artifacts.sourceReplyMemories.forEach((pair) => {
    throwIfTrainingPreparationStopped(shouldStop);
    appendHash(hash, pair.promptText);
    appendHash(hash, pair.responseText);
  });

  artifacts.replayReplyMemories.forEach((pair) => {
    throwIfTrainingPreparationStopped(shouldStop);
    appendHash(hash, pair.origin);
    appendHash(hash, pair.score);
    appendHash(hash, pair.promptText);
    appendHash(hash, pair.responseText);
  });

  (artifacts.datasetFiles || []).forEach((fileDescriptor) => {
    throwIfTrainingPreparationStopped(shouldStop);
    appendHash(hash, fileDescriptor.path || fileDescriptor);
    appendHash(hash, fileDescriptor.size || 0);
    appendHash(hash, fileDescriptor.label || '');
  });

  const settingsSnapshot = {
    ...buildTrainingSettingsSnapshot(state.settings.training),
    epochs: state.settings.training.epochs,
    batchSize: state.settings.training.batchSize,
  };

  appendHash(hash, JSON.stringify(settingsSnapshot));
  appendHash(hash, JSON.stringify(getRuntimeConfig().feedback || {}));
  return hash.digest('hex');
}

function resolveQueuedSources(state, options = {}) {
  if (Array.isArray(options.queuedSourcesOverride)) {
    return options.queuedSourcesOverride;
  }

  return options.includeQueuedSources === false ? [] : state.sources;
}

function resolveDatasetSourcePath(source = {}) {
  return cleanText(source?.contentPath || source?.datasetFilePath || '');
}

function isDatasetFileSource(source = {}) {
  return source?.type === 'dataset_file' && Boolean(resolveDatasetSourcePath(source));
}

function collectDatasetFileDescriptors(sources = []) {
  const descriptors = (Array.isArray(sources) ? sources : [])
    .filter((source) => isDatasetFileSource(source))
    .map((source) => ({
      path: resolveDatasetSourcePath(source),
      size: Math.max(Number(source.contentSize) || 0, 0),
      label: cleanText(source.label || '') || 'dataset_file',
    }))
    .filter((entry) => entry.path);

  return dedupeEntries(descriptors, (entry) => entry.path);
}

async function materializeTrainingSources(sources = [], options = {}) {
  const { shouldStop, onProgress } = options;
  const materialized = [];
  const totalSources = Math.max(Array.isArray(sources) ? sources.length : 0, 1);
  const inputSources = Array.isArray(sources) ? sources : [];

  for (let index = 0; index < inputSources.length; index += 1) {
    throwIfTrainingPreparationStopped(shouldStop);
    const source = inputSources[index];
    if (isDatasetFileSource(source)) {
      const datasetPath = resolveDatasetSourcePath(source);
      try {
        await fs.access(datasetPath);
      } catch (error) {
        if (error?.code === 'ENOENT') {
          throw new Error(
            `Файл датасета «${cleanText(source.label || '') || path.basename(datasetPath) || 'без имени'}» не найден на диске. ` +
            'Перезагрузите файл в очередь обучения.'
          );
        }
        throw error;
      }
      materialized.push({
        ...source,
        content: '',
        stats: source.stats && typeof source.stats === 'object'
          ? source.stats
          : computeStats(''),
        contentSize: Math.max(Number(source.contentSize) || 0, 0),
      });
    } else {
      const content = source?.content
        ? cleanText(source.content)
        : await readTrainingQueueSourceContent(source);

      if (content) {
        materialized.push({
          ...source,
          content,
          stats: source.stats && typeof source.stats === 'object'
            ? source.stats
            : computeStats(content),
          contentSize: Math.max(
            Number(source.contentSize) || 0,
            Buffer.byteLength(content, 'utf8')
          ),
        });
      }
    }

    if (typeof onProgress === 'function') {
      onProgress({
        stage: 'loading_queue_sources',
        processedSources: index + 1,
        totalSources,
        percent: Number((((index + 1) / totalSources) * 100).toFixed(1)),
        label: source?.label || '',
      });
    }

    await yieldToEventLoop();
  }

  return materialized;
}

async function buildTrainingSessionArtifactsAsync(state, options = {}) {
  const runtimeConfig = options.runtimeConfig || getRuntimeConfig();
  const queuedSources = resolveQueuedSources(state, options);
  const shouldStop = options.shouldStop;
  const progressCallback = typeof options.onProgress === 'function'
    ? options.onProgress
    : null;
  const materializedQueuedSources = await materializeTrainingSources(queuedSources, {
    shouldStop,
    onProgress: progressCallback,
  });
  const datasetFiles = collectDatasetFileDescriptors(materializedQueuedSources);
  const textQueuedSources = materializedQueuedSources.filter((source) => !isDatasetFileSource(source));
  const sourceDocumentsRaw = [];
  const sourceReplyMemories = [];
  for (let index = 0; index < textQueuedSources.length; index += 1) {
    throwIfTrainingPreparationStopped(shouldStop);
    const source = textQueuedSources[index];
    sourceDocumentsRaw.push(...buildSourceDocuments(source));
    sourceReplyMemories.push(...extractDialoguePairsFromSource(source));
    progressCallback?.({
      stage: 'extracting_dialogues',
      processedSources: index + 1,
      totalSources: Math.max(textQueuedSources.length, 1),
      percent: Number((((index + 1) / Math.max(textQueuedSources.length, 1)) * 100).toFixed(1)),
      label: source.label || '',
    });
    await yieldToEventLoop();
  }
  const sourceDocuments = dedupeEntries(
    sourceDocumentsRaw,
    (entry) => `${entry.label}\u241f${entry.text}`
  );
  const storedReplyMemories = normalizeStoredReplyMemories(state.knowledge.replyMemories || []);
  const chatReplyMemories = state.chats.flatMap((chat) => extractDialoguePairsFromChat(chat));
  const blockedReplies = collectBlockedReplies(state);

  const positiveFeedbackCount = state.chats
    .flatMap((chat) => chat.messages)
    .filter((message) => message.metadata?.userRating === 1)
    .length;
  const negativeFeedbackCount = state.chats
    .flatMap((chat) => chat.messages)
    .filter((message) => message.metadata?.userRating === -1)
    .length;

  const archivedReplayMemories = [...storedReplyMemories]
    .sort((left, right) => (right.score || 0) - (left.score || 0))
    .slice(0, ARCHIVED_REPLY_REPLAY_LIMIT);
  const replayReplyMemories = dedupeEntries(
    [...chatReplyMemories, ...archivedReplayMemories],
    (pair) => `${pair.promptText}\u241f${pair.responseText}\u241f${pair.origin}`
  )
    .map((pair) => ({
      ...pair,
      responseText: sanitizeReplyText(pair.responseText),
    }))
    .filter((pair) => pair.promptText && pair.responseText)
    .filter((pair) => !blockedReplies.has(pair.responseText.toLowerCase()))
    .filter((pair) => !isEchoReply(pair.promptText, pair.responseText));

  if (progressCallback) {
    progressCallback({
      stage: 'building_training_texts',
      processedSources: materializedQueuedSources.length,
      totalSources: Math.max(materializedQueuedSources.length, 1),
      percent: 0,
      label: '',
    });
  }

  const trainingTexts = buildTrainingTexts({
    sourceDocuments,
    sourceReplyMemories,
    replayReplyMemories,
    runtimeConfig,
  }, {
    shouldStop,
    onProgress: (progress) => {
      progressCallback?.({
        stage: 'building_training_texts',
        processedItems: progress.processedItems,
        totalItems: progress.totalItems,
        percent: progress.percent,
        phase: progress.phase,
      });
    },
  });

  throwIfTrainingPreparationStopped(shouldStop);

  return {
    materializedQueuedSources,
    sourceDocuments,
    sourceReplyMemories,
    replayReplyMemories,
    trainingTexts,
    datasetFiles,
    positiveFeedbackCount,
    negativeFeedbackCount,
    trainingCorpusSignature: createTrainingCorpusSignature(state, {
      sourceDocuments,
      sourceReplyMemories,
      replayReplyMemories,
      trainingTexts,
      datasetFiles,
      positiveFeedbackCount,
      negativeFeedbackCount,
    }, {
      shouldStop,
    }),
  };
}

function buildTrainingSessionArtifacts(state, options = {}) {
  const runtimeConfig = options.runtimeConfig || getRuntimeConfig();
  const queuedSources = resolveQueuedSources(state, options);
  const datasetFiles = collectDatasetFileDescriptors(queuedSources);
  const textQueuedSources = queuedSources.filter((source) => !isDatasetFileSource(source));
  const sourceDocuments = dedupeEntries(
    textQueuedSources.flatMap((source) => buildSourceDocuments(source)),
    (entry) => `${entry.label}\u241f${entry.text}`
  );
  const sourceReplyMemories = textQueuedSources.flatMap((source) => extractDialoguePairsFromSource(source));
  const storedReplyMemories = normalizeStoredReplyMemories(state.knowledge.replyMemories || []);
  const chatReplyMemories = state.chats.flatMap((chat) => extractDialoguePairsFromChat(chat));
  const blockedReplies = collectBlockedReplies(state);

  const positiveFeedbackCount = state.chats
    .flatMap((chat) => chat.messages)
    .filter((message) => message.metadata?.userRating === 1)
    .length;
  const negativeFeedbackCount = state.chats
    .flatMap((chat) => chat.messages)
    .filter((message) => message.metadata?.userRating === -1)
    .length;

  const archivedReplayMemories = [...storedReplyMemories]
    .sort((left, right) => (right.score || 0) - (left.score || 0))
    .slice(0, ARCHIVED_REPLY_REPLAY_LIMIT);
  const replayReplyMemories = dedupeEntries(
    [...chatReplyMemories, ...archivedReplayMemories],
    (pair) => `${pair.promptText}\u241f${pair.responseText}\u241f${pair.origin}`
  )
    .map((pair) => ({
      ...pair,
      responseText: sanitizeReplyText(pair.responseText),
    }))
    .filter((pair) => pair.promptText && pair.responseText)
    .filter((pair) => !blockedReplies.has(pair.responseText.toLowerCase()))
    .filter((pair) => !isEchoReply(pair.promptText, pair.responseText));

  const trainingTexts = buildTrainingTexts({
    sourceDocuments,
    sourceReplyMemories,
    replayReplyMemories,
    runtimeConfig,
  });

  return {
    sourceDocuments,
    sourceReplyMemories,
    replayReplyMemories,
    trainingTexts,
    datasetFiles,
    positiveFeedbackCount,
    negativeFeedbackCount,
    trainingCorpusSignature: createTrainingCorpusSignature(state, {
      sourceDocuments,
      sourceReplyMemories,
      replayReplyMemories,
      trainingTexts,
      datasetFiles,
      positiveFeedbackCount,
      negativeFeedbackCount,
    }),
  };
}

function buildKnowledgeArtifacts(state, options = {}) {
  const {
    includeQueuedSources = false,
    includeRetrievalArtifacts = true,
    includeTrainingArtifacts = false,
  } = options;
  const runtimeConfig = getRuntimeConfig();
  const queuedSources = resolveQueuedSources(state, {
    includeQueuedSources,
    queuedSourcesOverride: options.queuedSourcesOverride,
  });
  const sourceDocuments = queuedSources.flatMap((source) => buildSourceDocuments(source));
  const uniqueSourceDocuments = dedupeEntries(sourceDocuments, (entry) => `${entry.label}\u241f${entry.text}`);

  const dialogueMemoryConfig = runtimeConfig.dialogueMemory || {};
  const useSourceDialogues = dialogueMemoryConfig.useSourceDialogues !== false;
  const useChatDialogues = dialogueMemoryConfig.useChatDialogues === true;
  const responseOnlyIndex = dialogueMemoryConfig.responseOnlyIndex !== false;
  const minPairTopicalRelevance = Math.min(
    0.95,
    Math.max(0.02, Number(dialogueMemoryConfig.minPairTopicalRelevance) || 0.1)
  );

  const sourceReplyMemories = useSourceDialogues
    ? queuedSources.flatMap((source) => extractDialoguePairsFromSource(source))
    : [];
  const chatReplyMemories = useChatDialogues
    ? state.chats.flatMap((chat) => extractDialoguePairsFromChat(chat))
    : [];
  const storedReplyMemories = normalizeStoredReplyMemories(state.knowledge.replyMemories || []);
  const blockedReplies = collectBlockedReplies(state);

  const replyMemories = dedupeEntries(
    [...storedReplyMemories, ...sourceReplyMemories, ...chatReplyMemories],
    (pair) => `${pair.promptText}\u241f${pair.responseText}`
  )
    .map((pair) => ({
      ...pair,
      responseText: sanitizeReplyText(pair.responseText),
    }))
    .filter((pair) => pair.promptText && pair.responseText)
    .filter((pair) => computePromptReplyRelevance(pair.promptText, pair.responseText) >= minPairTopicalRelevance)
    .filter((pair) => !blockedReplies.has(pair.responseText.toLowerCase()))
    .filter((pair) => !isEchoReply(pair.promptText, pair.responseText));

  let tokenCount = 0;
  let sentenceCount = 0;

  uniqueSourceDocuments.forEach((documentEntry) => {
    const tokens = tokenizeWords(documentEntry.text);
    const sentences = splitIntoSentences(documentEntry.text);
    tokenCount += tokens.length;
    sentenceCount += sentences.length;
  });

  replyMemories.forEach((memory) => {
    tokenCount += tokenizeWords(memory.promptText).length;
    tokenCount += tokenizeWords(memory.responseText).length;
  });

  const positiveFeedbackCount = state.chats
    .flatMap((chat) => chat.messages)
    .filter((message) => message.metadata?.userRating === 1)
    .length;
  const negativeFeedbackCount = state.chats
    .flatMap((chat) => chat.messages)
    .filter((message) => message.metadata?.userRating === -1)
    .length;
  const archivedReplayMemories = storedReplyMemories
    .sort((left, right) => (right.score || 0) - (left.score || 0))
    .slice(0, ARCHIVED_REPLY_REPLAY_LIMIT);
  const trainingReplayMemories = [...chatReplyMemories, ...archivedReplayMemories]
    .map((pair) => ({
      ...pair,
      responseText: sanitizeReplyText(pair.responseText),
    }))
    .filter((pair) => pair.promptText && pair.responseText)
    .filter((pair) => computePromptReplyRelevance(pair.promptText, pair.responseText) >= minPairTopicalRelevance)
    .filter((pair) => !blockedReplies.has(pair.responseText.toLowerCase()))
    .filter((pair) => !isEchoReply(pair.promptText, pair.responseText));

  const artifacts = {
    sourceDocuments: uniqueSourceDocuments,
    sourceDocumentCount: uniqueSourceDocuments.length,
    replyMemories,
    blockedReplies,
    tokenCount,
    sentenceCount,
    replyPairCount: replyMemories.length,
    positiveFeedbackCount,
    negativeFeedbackCount,
  };

  if (includeRetrievalArtifacts) {
    const persistedKnowledgeChunks = dedupeEntries(
      normalizeStoredChunks(state.knowledge.chunks || []),
      (entry) => `${entry.label}\u241f${entry.text}`
    );
    const rawVocabulary = {};

    persistedKnowledgeChunks.forEach((chunk) => {
      const tokens = tokenizeWords(chunk.text);
      const sentences = splitIntoSentences(chunk.text);
      tokenCount += tokens.length;
      sentenceCount += sentences.length;
      tokens.forEach((token) => {
        rawVocabulary[token] = (rawVocabulary[token] || 0) + 1;
      });
    });

    uniqueSourceDocuments.forEach((documentEntry) => {
      tokenizeWords(documentEntry.text).forEach((token) => {
        rawVocabulary[token] = (rawVocabulary[token] || 0) + 1;
      });
    });

    replyMemories.forEach((memory) => {
      tokenizeWords(memory.promptText).forEach((token) => {
        rawVocabulary[token] = (rawVocabulary[token] || 0) + 1;
      });
      tokenizeWords(memory.responseText).forEach((token) => {
        rawVocabulary[token] = (rawVocabulary[token] || 0) + 1;
      });
    });

    const vocabularyEntries = Object.entries(rawVocabulary)
      .sort((left, right) => right[1] - left[1])
      .slice(0, state.settings.training.vocabularyLimit);
    const allowedTerms = new Set(vocabularyEntries.map(([term]) => term));
    const vocabulary = Object.fromEntries(vocabularyEntries);

    const persistedIndexedChunks = persistedKnowledgeChunks.map((chunk) => {
      const chunkTokens = tokenizeWords(chunk.text);
      return {
        ...chunk,
        sentences: splitIntoSentences(chunk.text),
        tokenCount: chunkTokens.length,
        documentLength: chunkTokens.length,
        termMap: createTermMap(chunkTokens, allowedTerms),
      };
    });
    const queuedSourceChunks = uniqueSourceDocuments.flatMap((documentEntry) =>
      createChunksForDocument(
        documentEntry,
        state.settings.training.chunkSize,
        state.settings.training.chunkOverlap,
        allowedTerms
      )
    );
    const chunks = dedupeEntries(
      [...persistedIndexedChunks, ...queuedSourceChunks],
      (entry) => `${entry.label}\u241f${entry.text}`
    );
    const indexedReplyMemories = replyMemories.map((memory) => {
      const indexedTokens = responseOnlyIndex
        ? tokenizeWords(memory.responseText)
        : [...tokenizeWords(memory.promptText), ...tokenizeWords(memory.responseText)];
      return {
        ...memory,
        termMap: createTermMap(indexedTokens, allowedTerms),
        documentLength: indexedTokens.length,
      };
    });

    artifacts.chunks = chunks;
    artifacts.replyMemories = indexedReplyMemories;
    artifacts.vocabulary = vocabulary;
    artifacts.bm25 = {
      knowledge: buildBm25Index(chunks),
      reply: buildBm25Index(indexedReplyMemories),
    };
    artifacts.topTerms = vocabularyEntries.slice(0, 12).map(([term, count]) => ({ term, count }));
    artifacts.replyPairCount = indexedReplyMemories.length;
  } else {
    artifacts.chunks = [];
    artifacts.vocabulary = {};
    artifacts.bm25 = createDefaultKnowledgeState().bm25;
    artifacts.topTerms = [];
  }

  if (includeTrainingArtifacts) {
    const trainingTexts = buildTrainingTexts({
      sourceDocuments: uniqueSourceDocuments,
      sourceReplyMemories,
      replayReplyMemories: trainingReplayMemories,
      runtimeConfig,
    });

    artifacts.trainingTexts = trainingTexts;
    artifacts.tokenizerTokenCount = trainingTexts.reduce(
      (sum, text) => sum + tokenizeForModel(text).length,
      0
    );
  } else {
    artifacts.trainingTexts = [];
    artifacts.tokenizerTokenCount = 0;
  }

  artifacts.tokenCount = tokenCount;
  artifacts.sentenceCount = sentenceCount;

  return {
    ...artifacts,
    corpusSignature: createCorpusSignature(state, artifacts),
  };
}

function selectContextMessages(chat, userMessage, runtimeConfig) {
  const includeAssistantMessages = runtimeConfig.context?.includeAssistantMessages !== false;
  const messages = chat.messages
    .filter((message) => message.metadata?.type !== 'system')
    .filter((message) => includeAssistantMessages || message.role !== 'assistant')
    .filter((message) => cleanText(message.content))
    .map((message, index) => ({
      ...message,
      index,
    }));

  const recentMessages = messages.slice(-12);
  const queryTokens = new Set(extractQueryTokens(userMessage, { fallbackToAll: true }));
  const scored = messages
    .filter((message) => message.content !== userMessage)
    .map((message) => {
      const tokens = extractQueryTokens(message.content, { fallbackToAll: false });
      let lexicalScore = 0;
      tokens.forEach((token) => {
        if (queryTokens.has(token)) {
          lexicalScore += 1;
        }
      });

      const recencyWeight = messages.length
        ? (message.index + 1) / messages.length
        : 0;

      return {
        message,
        score:
          lexicalScore * 1.15 +
          recencyWeight * 1.6 +
          (message.role === 'assistant' ? 0.08 : 0.28),
      };
    })
    .sort((left, right) => right.score - left.score)
    .slice(0, 14)
    .map((entry) => entry.message);

  const merged = new Map();
  [...recentMessages, ...scored].forEach((message) => {
    merged.set(message.id, message);
  });

  const ordered = [...merged.values()]
    .sort((left, right) => left.createdAt.localeCompare(right.createdAt))
    .slice(-runtimeConfig.context.maxMessages);

  const bounded = [];
  let totalCharacters = 0;

  for (let index = ordered.length - 1; index >= 0; index -= 1) {
    const message = ordered[index];
    const nextTotal = totalCharacters + message.content.length;
    if (bounded.length && nextTotal > runtimeConfig.context.maxCharacters) {
      continue;
    }

    bounded.unshift(message);
    totalCharacters = nextTotal;
  }

  return bounded;
}

function buildWeightedQueryMap(userMessage, chat, runtimeConfig) {
  const queryMap = {};
  addWeightedTokens(
    queryMap,
    extractQueryTokens(userMessage, { fallbackToAll: true }),
    3
  );

  const contextMessages = selectContextMessages(chat, userMessage, runtimeConfig);
  const reversedContext = [...contextMessages].reverse();
  reversedContext.forEach((message, index) => {
    const roleWeight = message.role === 'assistant'
      ? runtimeConfig.context.assistantWeight
      : runtimeConfig.context.userWeight;
    const topicalAlignment = computePromptReplyRelevance(userMessage, message.content);
    const relevanceWeight = topicalAlignment > 0
      ? 0.35 + Math.min(topicalAlignment * 1.35, 1)
      : 0;
    if (relevanceWeight <= 0) {
      return;
    }
    const weight = roleWeight * Math.pow(runtimeConfig.context.decay, index) * relevanceWeight;
    addWeightedTokens(
      queryMap,
      extractQueryTokens(message.content, { fallbackToAll: true }),
      weight
    );
  });

  return {
    queryMap,
    contextMessages,
  };
}

function buildPromptForGeneration(userMessage, contextMessages, chunkCandidates, replyCandidates, userLanguage = 'auto') {
  const promptParts = ['__bos__', '__ctx__'];
  const runtimeConfig = getRuntimeConfig();
  const normalizedSystemPrompt = cleanText(runtimeConfig.generation?.systemPrompt || '');

  if (normalizedSystemPrompt) {
    promptParts.push(`Системная инструкция: ${normalizedSystemPrompt.slice(0, 900)}`);
    promptParts.push('__sep__');
  }
  if (userLanguage === 'ru') {
    promptParts.push('Язык ответа: русский. Термины в коде можно оставлять на английском.');
    promptParts.push('__sep__');
  } else if (userLanguage === 'en') {
    promptParts.push('Response language: English.');
    promptParts.push('__sep__');
  }

  chunkCandidates.slice(0, 4).forEach((chunk) => {
    promptParts.push(chunk.text);
    promptParts.push('__sep__');
  });

  replyCandidates.slice(0, 2).forEach((pair) => {
    promptParts.push('__usr__');
    promptParts.push(pair.promptText);
    promptParts.push('__sep__');
    promptParts.push('__asst__');
    promptParts.push(pair.responseText);
    promptParts.push('__sep__');
  });

  contextMessages.slice(-8).forEach((message) => {
    promptParts.push(message.role === 'assistant' ? '__asst__' : '__usr__');
    promptParts.push(message.content);
    promptParts.push('__sep__');
  });

  promptParts.push('__usr__');
  promptParts.push(userMessage);
  promptParts.push('__sep__');
  promptParts.push('__asst__');

  return structuredTrainingText(promptParts);
}

function computeGroundedOverlap(replyTokens, candidateTexts) {
  if (!replyTokens.length || !candidateTexts.length) {
    return 0;
  }

  const groundedTokens = new Set(candidateTexts.flatMap((text) => tokenizeWords(text)));
  if (!groundedTokens.size) {
    return 0;
  }

  let shared = 0;
  replyTokens.forEach((token) => {
    if (groundedTokens.has(token)) {
      shared += 1;
    }
  });

  return shared / replyTokens.length;
}

function computeReplySimilarity(leftText, rightText) {
  const leftTokens = new Set(tokenizeWords(leftText));
  const rightTokens = new Set(tokenizeWords(rightText));

  if (!leftTokens.size || !rightTokens.size) {
    return 0;
  }

  let shared = 0;
  leftTokens.forEach((token) => {
    if (rightTokens.has(token)) {
      shared += 1;
    }
  });

  return (2 * shared) / (leftTokens.size + rightTokens.size);
}

function computeRecentReplyPenalty(replyText, recentAssistantReplies = []) {
  const maxSimilarity = recentAssistantReplies.reduce((maxValue, recentReply) => (
    Math.max(maxValue, computeReplySimilarity(replyText, recentReply))
  ), 0);

  if (maxSimilarity >= 0.96) {
    return 0.42;
  }

  if (maxSimilarity >= 0.88) {
    return 0.24;
  }

  if (maxSimilarity >= 0.78) {
    return 0.12;
  }

  return 0;
}

function hasLoopingPattern(tokens) {
  if (tokens.length < 6) {
    return false;
  }

  const lastThree = tokens.slice(-3).join(' ');
  const previousThree = tokens.slice(-6, -3).join(' ');
  return lastThree === previousThree;
}

function computeSelfScore({
  userMessage,
  replyText,
  chunkCandidates,
  replyCandidates,
  contextMessages,
  recentAssistantReplies = [],
}) {
  const cleanReply = sanitizeReplyText(replyText);
  const replyTokens = tokenizeWords(cleanReply);
  if (!cleanReply || !replyTokens.length) {
    return 0;
  }

  const groundedTexts = [
    ...chunkCandidates.map((chunk) => chunk.text),
    ...replyCandidates.slice(0, 3).map((pair) => pair.responseText),
  ];

  const groundedOverlap = computeGroundedOverlap(replyTokens, groundedTexts);
  const topicalAlignment = computePromptReplyRelevance(userMessage, cleanReply);
  const lengthScore = Math.min(replyTokens.length / 18, 1);
  const echoPenalty = isEchoReply(userMessage, cleanReply) ? 0.55 : 0;
  const loopPenalty = hasLoopingPattern(replyTokens) ? 0.4 : 0;
  const questionPenalty = cleanReply.endsWith('?') ? 0.12 : 0;
  const repeatPenalty = computeRecentReplyPenalty(cleanReply, recentAssistantReplies);

  const rawScore = 0.16 + groundedOverlap * 0.46 + topicalAlignment * 0.3 + lengthScore * 0.14
    - echoPenalty - loopPenalty - questionPenalty - repeatPenalty;
  const boundedRaw = Math.max(0.02, Math.min(0.985, rawScore));
  const confidence = Math.max(
    0,
    Math.min(
      1,
      groundedOverlap * 0.58 + topicalAlignment * 0.42 + Math.min(replyTokens.length / 48, 0.22)
    )
  );
  const smoothed = 0.08 + boundedRaw * 0.72 + confidence * 0.2;
  return Math.max(0.02, Math.min(0.985, smoothed));
}

function computeLowTopicalPenalty(relevanceScore) {
  if (relevanceScore < 0.12) {
    return 0.34;
  }
  if (relevanceScore < 0.2) {
    return 0.16;
  }
  return 0;
}

function chooseBestReplyCandidate({
  userMessage,
  userLanguage = 'auto',
  neuralReply,
  bestReply,
  knowledgeText,
  knowledgeSentences,
  maxSentences,
  chunkCandidates,
  webChunkCandidates = [],
  replyCandidates,
  contextMessages,
  recentAssistantReplies = [],
  preferGroundedKnowledge = false,
}) {
  const candidates = [];

  if (bestReply) {
    const content = sanitizeReplyText(bestReply.responseText);
    const promptSideRelevance = computePromptReplyRelevance(userMessage, bestReply.promptText);
    const responseSideRelevance = computePromptReplyRelevance(userMessage, content);
    const topicalRelevance = responseSideRelevance * 0.78 + promptSideRelevance * 0.22;
    const memoryPenalty = computeLowTopicalPenalty(topicalRelevance);
    if (responseSideRelevance >= 0.08) {
      candidates.push({
        mode: 'reply_memory',
        content,
        score: computeSelfScore({
          userMessage,
          replyText: content,
          chunkCandidates,
          replyCandidates,
          contextMessages,
          recentAssistantReplies,
        }) + bestReply.score * 0.03 + topicalRelevance * 0.46 - memoryPenalty - (preferGroundedKnowledge ? 0.2 : 0)
          - languageMismatchPenalty(userLanguage, content),
      });
    }
  }

  if (knowledgeText) {
    const topicalRelevance = computePromptReplyRelevance(userMessage, knowledgeText);
    candidates.push({
      mode: 'knowledge',
      content: cleanText(knowledgeText),
      score: computeSelfScore({
        userMessage,
        replyText: knowledgeText,
        chunkCandidates,
        replyCandidates,
        contextMessages,
        recentAssistantReplies,
      }) + topicalRelevance * 0.34 - computeLowTopicalPenalty(topicalRelevance) + (preferGroundedKnowledge ? 0.1 : 0)
        - languageMismatchPenalty(userLanguage, knowledgeText),
    });
  }

  const hybridReply = buildHybridReply({
    bestReply,
    neuralReply,
    knowledgeSentences,
    maxSentences,
  });

  if (hybridReply) {
    const topicalRelevance = computePromptReplyRelevance(userMessage, hybridReply);
    candidates.push({
      mode: 'hybrid',
      content: hybridReply,
      score: computeSelfScore({
        userMessage,
        replyText: hybridReply,
        chunkCandidates,
        replyCandidates,
        contextMessages,
        recentAssistantReplies,
      }) + (bestReply ? 0.02 : 0) + topicalRelevance * 0.34 - computeLowTopicalPenalty(topicalRelevance) - (preferGroundedKnowledge ? 0.14 : 0)
        - languageMismatchPenalty(userLanguage, hybridReply),
    });
  }

  if (neuralReply) {
    const topicalRelevance = computePromptReplyRelevance(userMessage, neuralReply);
    candidates.push({
      mode: 'neural',
      content: neuralReply,
      score: computeSelfScore({
        userMessage,
        replyText: neuralReply,
        chunkCandidates,
        replyCandidates,
        contextMessages,
        recentAssistantReplies,
      }) + 0.02 + topicalRelevance * 0.28 - computeLowTopicalPenalty(topicalRelevance) - (preferGroundedKnowledge ? 0.24 : 0)
        - languageMismatchPenalty(userLanguage, neuralReply),
    });
  }

  if (webChunkCandidates.length) {
    const webGrounded = buildGroundedFallback(webChunkCandidates, maxSentences);
    if (webGrounded) {
      const topicalRelevance = computePromptReplyRelevance(userMessage, webGrounded);
      candidates.push({
        mode: 'web_grounded',
        content: webGrounded,
        score: computeSelfScore({
          userMessage,
          replyText: webGrounded,
          chunkCandidates,
          replyCandidates,
          contextMessages,
          recentAssistantReplies,
        }) + 0.14 + topicalRelevance * 0.52 - (topicalRelevance < 0.12 ? 0.08 : 0) + (preferGroundedKnowledge ? 0.22 : 0)
          - languageMismatchPenalty(userLanguage, webGrounded),
      });
    }
  }

  const minimumRelevance = preferGroundedKnowledge ? 0.16 : 0.1;
  const filtered = candidates
    .filter((candidate) => candidate.content)
    .filter((candidate) => computePromptReplyRelevance(userMessage, candidate.content) >= minimumRelevance)
    .filter((candidate) => !isEchoReply(userMessage, candidate.content))
    .filter((candidate) => computeRecentReplyPenalty(candidate.content, recentAssistantReplies) < 0.42)
    .sort((left, right) => right.score - left.score);

  return filtered[0] || null;
}

function findMessageById(state, messageId) {
  for (const chat of state.chats) {
    for (let index = 0; index < chat.messages.length; index += 1) {
      if (chat.messages[index].id === messageId) {
        return {
          chat,
          message: chat.messages[index],
          messageIndex: index,
        };
      }
    }
  }

  return null;
}

function updateModelMetricsFromArtifacts(state, artifacts) {
  state.model.engine = createDefaultModelState().engine;
  state.knowledge.chunks = artifacts.chunks;
  state.knowledge.replyMemories = artifacts.replyMemories;
  state.knowledge.vocabulary = artifacts.vocabulary;
  state.knowledge.bm25 = artifacts.bm25;
  state.knowledge.blockedReplies = Array.from(artifacts.blockedReplies || []);

  state.model.sourceCount = state.sources.length;
  state.model.chatCount = state.chats.length;
  state.model.replyPairCount = artifacts.replyPairCount;
  state.model.chunkCount = artifacts.chunks.length;
  state.model.tokenCount = artifacts.tokenCount;
  state.model.sentenceCount = artifacts.sentenceCount;
  state.model.vocabularySize = Object.keys(artifacts.vocabulary).length;
  state.model.topTerms = artifacts.topTerms;
  state.model.positiveFeedbackCount = artifacts.positiveFeedbackCount;
  state.model.negativeFeedbackCount = artifacts.negativeFeedbackCount;
}

function resetTrainedModelState(state) {
  state.model.trainedEpochs = 0;
  state.model.targetEpochs = 0;
  state.model.batchesProcessed = 0;
  state.model.lastLoss = null;
  state.model.averageLoss = null;
  state.model.validationLoss = null;
  state.model.bestValidationLoss = null;
  state.model.perplexity = null;
  state.model.lastTrainingAt = null;
  state.model.corpusSignature = null;
  state.model.parameterCount = 0;
  state.model.averageSelfScore = null;
  state.training.history = [];
  state.training.progress = {
    currentEpoch: 0,
    totalEpochs: 0,
    currentBatch: 0,
    totalBatches: 0,
    percent: 0,
  };
  state.knowledge.languageModel = createDefaultKnowledgeState().languageModel;
}

function isTrainingLocked(state) {
  return (
    TRAINING_LOCKED_STATUSES.has(state.model.lifecycle) ||
    TRAINING_LOCKED_STATUSES.has(state.training.status) ||
    state.model.status === 'saving_checkpoint'
  );
}

function assertTrainingUnlocked(state, actionLabel) {
  if (!isTrainingLocked(state)) {
    return;
  }

  throw new Error(
    `${actionLabel} недоступно, пока идет обучение или сохранение чекпоинта. Дождитесь завершения или поставьте обучение на паузу.`
  );
}

function hasRetrainingImpact(previousSettings, nextSettings) {
  return RETRAINING_KEYS.some((key) => previousSettings.training?.[key] !== nextSettings.training?.[key]);
}

function hasCorpusRebuildImpact(previousSettings, nextSettings) {
  return CORPUS_REBUILD_KEYS.some((key) => previousSettings.training?.[key] !== nextSettings.training?.[key]);
}

function shouldUseNeuralGeneration({ bestReply, knowledgeSentences, state }) {
  const replyScore = Number(bestReply?.score || 0);
  const minSimilarity = Math.max(Number(state.settings.training.minSimilarity) || 0, 0);
  const maxSentences = Math.max(Number(state.settings.generation.maxReplySentences) || 1, 1);
  const hasGroundedKnowledge = knowledgeSentences.length >= Math.min(maxSentences, 2);

  if (replyScore >= Math.max(0.42, minSimilarity * 2.5)) {
    return false;
  }

  if (replyScore >= Math.max(0.28, minSimilarity * 1.75) && hasGroundedKnowledge) {
    return false;
  }

  return true;
}

function buildDashboard(state, chatId = null) {
  ensureChatAvailability(state);
  ensureModelRegistryState(state);
  const runtimeConfig = getRuntimeConfig();
  const selectedChat = state.chats.find((chat) => chat.id === chatId) || state.chats[0] || null;
  const activeChat = summarizeActiveChat(selectedChat);
  const ratedMessages = state.chats
    .flatMap((chat) => chat.messages)
    .filter((message) => message.role === 'assistant' && message.metadata?.userRating)
    .length;

  return {
    settings: state.settings,
    model: state.model,
    knowledge: {
      languageModel: state.knowledge.languageModel,
    },
    training: {
      ...state.training,
      availableStatuses: MODEL_STATUS_FLOW,
    },
    runtime: {
      contextStrategy: runtimeConfig.context.strategy,
      generatorBackend: getGeneratorBackend(),
      config: runtimeConfig,
      trainingExecutionMode: state.settings.training.executionMode,
      trainingBackend: state.model.computeBackendLabel || state.model.computeBackend,
      trainingBackendWarning: state.model.computeBackendWarning || '',
      storage: state.model.storage,
      ratedMessages,
    },
    modelRegistry: {
      activeModelId: state.modelRegistry?.activeModelId || null,
      items: (state.modelRegistry?.items || []).map(summarizeModelRegistryEntry),
    },
    sources: state.sources.map(summarizeSource),
    trainingQueues: {
      items: (state.trainingQueues?.items || []).map((queue, index) => summarizeTrainingQueue(queue, index)),
      runner: state.trainingQueues?.runner || createDefaultTrainingQueuesState().runner,
    },
    chats: state.chats.map(summarizeChat),
    activeChat,
  };
}

function buildRealtimeSnapshot(state) {
  const snapshot = buildDashboard(state);
  delete snapshot.activeChat;
  return snapshot;
}

function createModelEngine({ getState, updateState }) {
  let activeTrainingPromise = null;
  let activeTrainingQueueRunnerPromise = null;
  let activeTrainingStartPromise = null;
  let activeTrainingStartResolve = null;
  let activeTrainingStartReject = null;
  let activeTrainingWorker = null;
  let activeTrainingStopSignal = null;
  let activeTrainingTerminationMode = null;
  let activeTrainingRollbackSnapshot = null;
  let neuralRuntime = null;

  function hasActiveTrainingExecution() {
    return Boolean(
      activeTrainingPromise ||
      activeTrainingQueueRunnerPromise ||
      activeTrainingWorker ||
      activeTrainingStartPromise
    );
  }

  async function repairStaleExecutionStateIfNeeded(reason = 'stale_execution_state') {
    if (hasActiveTrainingExecution()) {
      return false;
    }

    const snapshot = getState();
    const queueRunner = snapshot.trainingQueues?.runner || null;
    const queueRunnerCanBeSanitized = queueRunner?.status !== 'error';
    const hasStaleQueueRunnerSnapshot = Boolean(
      queueRunner &&
      queueRunnerCanBeSanitized &&
      !queueRunner.active &&
      (
        queueRunner.status !== 'idle' ||
        (Number(queueRunner.totalQueues) || 0) > 0 ||
        (Number(queueRunner.currentQueueIndex) || 0) > 0 ||
        (Array.isArray(queueRunner.pendingQueueIds) && queueRunner.pendingQueueIds.length) ||
        (Array.isArray(queueRunner.completedQueueIds) && queueRunner.completedQueueIds.length)
      )
    );
    const hasStaleTrainingState =
      snapshot.training?.status === 'training' ||
      snapshot.model?.lifecycle === 'training' ||
      snapshot.model?.status === 'saving_checkpoint' ||
      snapshot.trainingQueues?.runner?.active ||
      hasStaleQueueRunnerSnapshot;

    if (!hasStaleTrainingState) {
      return false;
    }

    await updateState(async (state) => {
      activeTrainingRollbackSnapshot = null;
      const interruptedQueueId = state.trainingQueues?.runner?.currentQueueId || null;
      state.trainingQueues.items = (state.trainingQueues.items || []).map((queue) => (
        queue.status === 'running'
          ? {
            ...queue,
            status: Array.isArray(queue.sources) && queue.sources.length ? 'pending' : 'empty',
            updatedAt: nowIso(),
          }
          : queue
      ));
      if (state.trainingQueues?.runner?.active) {
        if (
          interruptedQueueId &&
          !state.trainingQueues.runner.pendingQueueIds.includes(interruptedQueueId)
        ) {
          state.trainingQueues.runner.pendingQueueIds = [
            interruptedQueueId,
            ...state.trainingQueues.runner.pendingQueueIds,
          ];
        }
        resetTrainingQueueRunnerState(state);
      }

      if (state.training.status === 'training' || state.model.lifecycle === 'training') {
        state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
        state.model.status = 'ready';
        state.training.status = 'idle';
        state.training.phase = reason;
        state.training.message = 'Предыдущее обучение не завершилось корректно. Сервер вернул состояние к последнему стабильному состоянию.';
        pushStatus(state, 'idle', reason, state.training.message, { updateTrainingState: false });
      }
    }, {
      writeSources: false,
      writeChats: false,
      writeArtifacts: false,
    });

    return true;
  }

  async function disposeNeuralRuntime() {
    disposeRuntime(neuralRuntime);
    neuralRuntime = null;
  }

  function createStopSignal() {
    activeTrainingStopSignal = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));
    Atomics.store(activeTrainingStopSignal, 0, 0);
    return activeTrainingStopSignal;
  }

  function createTrainingStartSignal() {
    activeTrainingStartPromise = new Promise((resolve, reject) => {
      activeTrainingStartResolve = resolve;
      activeTrainingStartReject = reject;
    });
    return activeTrainingStartPromise;
  }

  function resolveTrainingStartSignal() {
    if (activeTrainingStartResolve) {
      activeTrainingStartResolve();
    }
    activeTrainingStartResolve = null;
    activeTrainingStartReject = null;
  }

  function rejectTrainingStartSignal(error) {
    if (activeTrainingStartReject) {
      activeTrainingStartReject(error);
    }
    activeTrainingStartResolve = null;
    activeTrainingStartReject = null;
  }

  function requestTrainingStop() {
    if (activeTrainingStopSignal) {
      Atomics.store(activeTrainingStopSignal, 0, 1);
    }

    activeTrainingWorker?.postMessage({ type: 'stop' });
  }

  function isTrainingStopRequested() {
    return Boolean(activeTrainingStopSignal && Atomics.load(activeTrainingStopSignal, 0) === 1);
  }

  function getRollbackCheckpointPaths(storage) {
    const artifactDir = storage?.artifactDir || path.dirname(storage?.manifestPath || __dirname);
    const rollbackDir = path.join(artifactDir, 'rollback-checkpoint');
    const rollbackNeuralDir = path.join(rollbackDir, 'neural-model');

    return {
      rollbackDir,
      rollbackNeuralDir,
      snapshotPath: path.join(rollbackDir, 'state-snapshot.json'),
      manifestPath: path.join(rollbackDir, 'model-manifest.json'),
      knowledgeIndexPath: path.join(rollbackDir, 'knowledge-index.json'),
      languageModelPath: path.join(rollbackDir, 'language-model.json'),
      tokenizerPath: path.join(rollbackNeuralDir, 'tokenizer.json'),
      neuralWeightsPath: path.join(rollbackNeuralDir, 'weights.bin'),
      neuralSpecPath: path.join(rollbackNeuralDir, 'weights-spec.json'),
    };
  }

  function getCheckpointFilePairs(storage, rollbackPaths) {
    return [
      [storage?.manifestPath, rollbackPaths.manifestPath],
      [storage?.knowledgeIndexPath, rollbackPaths.knowledgeIndexPath],
      [storage?.languageModelPath, rollbackPaths.languageModelPath],
      [storage?.tokenizerPath, rollbackPaths.tokenizerPath],
      [storage?.neuralWeightsPath, rollbackPaths.neuralWeightsPath],
      [storage?.neuralSpecPath, rollbackPaths.neuralSpecPath],
    ].filter(([sourcePath, targetPath]) => Boolean(sourcePath && targetPath));
  }

  async function createRollbackCheckpoint(state) {
    const rollbackPaths = getRollbackCheckpointPaths(state.model.storage);
    const filePairs = getCheckpointFilePairs(state.model.storage, rollbackPaths);

    await fs.rm(rollbackPaths.rollbackDir, { recursive: true, force: true });
    await fs.mkdir(rollbackPaths.rollbackNeuralDir, { recursive: true });

    await Promise.all(filePairs.map(async ([sourcePath, targetPath]) => {
      await fs.copyFile(sourcePath, targetPath);
    }));

    const snapshotPayload = {
      createdAt: nowIso(),
      model: structuredClone(state.model),
      training: structuredClone(state.training),
      knowledge: structuredClone(state.knowledge),
    };
    await fs.writeFile(rollbackPaths.snapshotPath, JSON.stringify(snapshotPayload, null, 2), 'utf8');
    return snapshotPayload;
  }

  async function restoreRollbackCheckpoint(storage) {
    const rollbackPaths = getRollbackCheckpointPaths(storage);
    const filePairs = getCheckpointFilePairs(storage, rollbackPaths)
      .map(([sourcePath, targetPath]) => [targetPath, sourcePath]);

    const requiredPaths = [
      rollbackPaths.snapshotPath,
      ...filePairs.map(([sourcePath]) => sourcePath),
    ];

    await Promise.all(requiredPaths.map((targetPath) => fs.access(targetPath)));
    await Promise.all(filePairs.map(async ([sourcePath, targetPath]) => {
      await fs.copyFile(sourcePath, targetPath);
    }));

    const snapshotRaw = await fs.readFile(rollbackPaths.snapshotPath, 'utf8');
    return JSON.parse(snapshotRaw);
  }

  async function stopActiveTraining(terminationMode) {
    if (!activeTrainingWorker && !activeTrainingPromise) {
      return;
    }

    activeTrainingTerminationMode = terminationMode;
    requestTrainingStop();

    try {
      if (activeTrainingWorker) {
        await Promise.race([
          activeTrainingWorker.terminate(),
          delay(TRAINING_STOP_TIMEOUT_MS).then(() => {
            throw new Error('Остановка обучения заняла слишком много времени. Повторите действие или перезапустите сервер.');
          }),
        ]);
      }
    } catch (error) {
      if (!String(error?.message || '').includes('заняла слишком много времени')) {
        // Ignore worker termination failures and continue cleanup.
      } else {
        throw error;
      }
    }

    const pendingPromise = activeTrainingPromise;
    if (pendingPromise) {
      try {
        await pendingPromise;
      } catch (error) {
        // Errors are handled by the training promise handlers.
      }
    }

    activeTrainingWorker = null;
    activeTrainingPromise = null;
    activeTrainingStopSignal = null;
    activeTrainingTerminationMode = null;
  }

  async function restoreLastStableCheckpointState(state, options = {}) {
    const {
      phase = 'rollback_restored',
      restoredMessage = 'Дообучение остановлено. Восстановлен последний стабильный чекпоинт модели.',
      fallbackTrainedMessage = 'Подготовка или дообучение остановлены. Последний стабильный чекпоинт модели остался без изменений.',
      fallbackFreshMessage = 'Подготовка обучения остановлена до создания нового чекпоинта.',
    } = options;

    let snapshotToRestore = activeTrainingRollbackSnapshot;
    try {
      const restoredFromDisk = await restoreRollbackCheckpoint(state.model.storage);
      snapshotToRestore = restoredFromDisk || snapshotToRestore;
    } catch (error) {
      if (!snapshotToRestore) {
        state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
        state.model.status = 'ready';
        state.training.status = 'idle';
        state.training.phase = phase;
        state.training.message = state.model.trainedEpochs
          ? fallbackTrainedMessage
          : fallbackFreshMessage;
        pushStatus(state, 'idle', phase, state.training.message);
        return { restored: false };
      }
    }

    if (!snapshotToRestore?.model || !snapshotToRestore?.training || !snapshotToRestore?.knowledge) {
      throw new Error('Резервная копия чекпоинта повреждена. Откат невозможен.');
    }

    const preservedStorage = structuredClone(state.model.storage);
    const preservedArtifactFiles = [...(state.model.artifactFiles || [])];

    state.model = {
      ...createDefaultModelState(),
      ...snapshotToRestore.model,
      storage: preservedStorage,
      artifactFiles: preservedArtifactFiles,
    };
    state.training = {
      ...createDefaultTrainingState(),
      ...snapshotToRestore.training,
      requestedStop: false,
      updatedAt: nowIso(),
    };
    state.knowledge = {
      ...createDefaultKnowledgeState(),
      ...snapshotToRestore.knowledge,
    };

    state.model.exists = true;
    state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
    state.model.status = 'ready';
    state.training.status = 'idle';
    state.training.phase = phase;
    state.training.message = restoredMessage;
    pushStatus(state, 'idle', phase, state.training.message);
    return { restored: true };
  }

  function prepareTrainingQueuesState(state) {
    if (!state.trainingQueues || typeof state.trainingQueues !== 'object') {
      state.trainingQueues = createDefaultTrainingQueuesState();
    }
    if (!Array.isArray(state.trainingQueues.items)) {
      state.trainingQueues.items = [];
    }
    if (!state.trainingQueues.runner || typeof state.trainingQueues.runner !== 'object') {
      state.trainingQueues.runner = createDefaultTrainingQueuesState().runner;
    }

     const validQueueIds = new Set();
     state.trainingQueues.items = state.trainingQueues.items.map((queue, index) => {
      const normalizedSources = Array.isArray(queue.sources) ? queue.sources : [];
      const normalizedQueue = {
        ...queue,
        name: normalizeTrainingQueueName(queue.name, index + 1),
        sources: normalizedSources,
        status: normalizedSources.length
          ? (queue.status === 'completed' ? 'completed' : queue.status || 'pending')
          : 'empty',
        lastError: cleanText(queue.lastError || ''),
      };
      validQueueIds.add(normalizedQueue.id);
      return normalizedQueue;
    });

    const runner = state.trainingQueues.runner;
    runner.pendingQueueIds = Array.isArray(runner.pendingQueueIds)
      ? runner.pendingQueueIds.filter((queueId) => {
        const queue = state.trainingQueues.items.find((entry) => entry.id === queueId);
        return Boolean(queue && queue.sources.length && queue.status !== 'completed');
      })
      : [];
    runner.completedQueueIds = Array.isArray(runner.completedQueueIds)
      ? runner.completedQueueIds.filter((queueId) => validQueueIds.has(queueId))
      : [];
    runner.currentQueueId = validQueueIds.has(runner.currentQueueId) ? runner.currentQueueId : null;
    runner.lastCompletedQueueId = validQueueIds.has(runner.lastCompletedQueueId)
      ? runner.lastCompletedQueueId
      : null;
    runner.failedQueueId = validQueueIds.has(runner.failedQueueId) ? runner.failedQueueId : null;
    runner.lastError = cleanText(runner.lastError || '');

    if (!runner.active) {
      runner.status = runner.status === 'error' && runner.failedQueueId ? 'error' : 'idle';
      runner.currentQueueId = null;
      runner.currentQueueIndex = 0;
      runner.totalQueues = 0;
      runner.pendingQueueIds = [];
      runner.completedQueueIds = [];
      if (runner.status !== 'error') {
        runner.lastError = '';
      }
    } else {
      if (runner.currentQueueId) {
        runner.pendingQueueIds = runner.pendingQueueIds.filter((queueId) => queueId !== runner.currentQueueId);
      }
      runner.totalQueues = Math.max(
        Number(runner.totalQueues) || 0,
        runner.pendingQueueIds.length + runner.completedQueueIds.length + (runner.currentQueueId ? 1 : 0)
      );
    }
  }

function markQueueAsRunning(state, queueId) {
  prepareTrainingQueuesState(state);
  const queue = state.trainingQueues.items.find((entry) => entry.id === queueId);
  if (!queue) {
    throw new Error('Очередь обучения не найдена.');
    }

  const startedAt = nowIso();
  queue.status = 'running';
  queue.updatedAt = startedAt;
  queue.lastError = '';
  queue.lastRunAt = startedAt;
  state.trainingQueues.runner.pendingQueueIds = state.trainingQueues.runner.pendingQueueIds
    .filter((entryId) => entryId !== queueId);
  state.trainingQueues.runner.currentQueueId = queueId;
  state.trainingQueues.runner.currentQueueIndex = state.trainingQueues.runner.completedQueueIds.length + 1;
  state.trainingQueues.runner.updatedAt = startedAt;
  return queue;
}

  function markQueueAsPendingAfterInterruption(state, queueId, runnerStatus, lastError = '') {
    prepareTrainingQueuesState(state);
    const queue = state.trainingQueues.items.find((entry) => entry.id === queueId);
    const updatedAt = nowIso();

    if (queue) {
      queue.status = Array.isArray(queue.sources) && queue.sources.length ? 'pending' : 'empty';
      queue.updatedAt = updatedAt;
      queue.lastError = cleanText(lastError || '');
    }

    if (queueId && !state.trainingQueues.runner.pendingQueueIds.includes(queueId) && queue?.sources?.length) {
      state.trainingQueues.runner.pendingQueueIds = [queueId, ...state.trainingQueues.runner.pendingQueueIds];
    }

    state.trainingQueues.runner = {
      ...state.trainingQueues.runner,
      active: false,
      status: runnerStatus,
      currentQueueId: null,
      failedQueueId: runnerStatus === 'error' ? queueId : null,
      updatedAt,
      lastError: cleanText(lastError || ''),
    };
  }

  function markQueueAsCompleted(state, queueId, queueName, completedAt) {
    prepareTrainingQueuesState(state);
    const queue = state.trainingQueues.items.find((entry) => entry.id === queueId);
    if (queue) {
      queue.status = 'completed';
      queue.updatedAt = completedAt;
      queue.completedAt = completedAt;
      queue.lastRunAt = completedAt;
      queue.lastError = '';
    }

    state.trainingQueues.runner.pendingQueueIds = state.trainingQueues.runner.pendingQueueIds
      .filter((entryId) => entryId !== queueId);
    state.trainingQueues.runner.completedQueueIds = dedupeEntries(
      [...state.trainingQueues.runner.completedQueueIds, queueId],
      (entry) => entry
    );
    state.trainingQueues.runner.lastCompletedQueueId = queueId;
    state.trainingQueues.runner.failedQueueId = null;
    state.trainingQueues.runner.lastError = '';
    state.trainingQueues.runner.updatedAt = completedAt;

  if (!state.trainingQueues.runner.pendingQueueIds.length) {
      const completedQueueIds = [...state.trainingQueues.runner.completedQueueIds];
      const lastCompletedQueueId = queueId;
      const updatedAt = completedAt;
      pushStatus(
        state,
        'completed',
        'queue_runner_completed',
        `Автоочередь обучения завершена. Последней успешно применена очередь «${queueName}».`,
        { updateTrainingState: false }
      );
      state.trainingQueues.runner = {
        ...createDefaultTrainingQueuesState().runner,
        status: 'idle',
        active: false,
        updatedAt,
        lastCompletedQueueId,
        completedQueueIds,
      };
      return;
    }

    state.trainingQueues.runner.active = true;
    state.trainingQueues.runner.status = 'running';
    state.trainingQueues.runner.currentQueueId = null;
    state.trainingQueues.runner.currentQueueIndex = state.trainingQueues.runner.completedQueueIds.length;
  }

  async function removeModelArtifacts(storage) {
    const fileTargets = [
      storage?.manifestPath,
      storage?.knowledgeIndexPath,
      storage?.languageModelPath,
      storage?.tokenizerPath,
      storage?.neuralWeightsPath,
      storage?.neuralSpecPath,
    ].filter(Boolean);

    await Promise.all(fileTargets.map(async (targetPath) => {
      try {
        await fs.rm(targetPath, { force: true });
      } catch (error) {
        // Ignore missing files.
      }
    }));

    if (storage?.neuralModelDir) {
      try {
        await fs.mkdir(storage.neuralModelDir, { recursive: true });
      } catch (error) {
        // Ignore directory recreation failures.
      }
    }

    const rollbackPaths = getRollbackCheckpointPaths(storage);
    await fs.rm(rollbackPaths.rollbackDir, { recursive: true, force: true });
  }

  async function readCheckpointMetadata(storage) {
    const hasTokenizer = await pathExists(storage?.tokenizerPath);
    const hasWeights = await pathExists(storage?.neuralWeightsPath);
    const hasSpec = await pathExists(storage?.neuralSpecPath);
    if (!hasTokenizer || !hasWeights || !hasSpec) {
      return null;
    }

    const [tokenizerJson, specJson] = await Promise.all([
      readJsonFileIfExists(storage.tokenizerPath, null),
      readJsonFileIfExists(storage.neuralSpecPath, null),
    ]);

    if (!tokenizerJson || !specJson) {
      return null;
    }

    const manifest = specJson.manifest && typeof specJson.manifest === 'object'
      ? specJson.manifest
      : {};
    const vocabularySize = toPositiveNumber(
      manifest.vocabularySize,
      Array.isArray(tokenizerJson.idToToken) ? tokenizerJson.idToToken.length : 0
    );
    const parameterCount = toPositiveNumber(
      manifest.parameterCount,
      computeParameterCountFromSpecs(specJson.specs || [])
    );

    return {
      manifest,
      vocabularySize,
      parameterCount,
      tokenizerJson,
      specJson,
      hasTokenizer,
      hasWeights,
      hasSpec,
    };
  }

  async function reconcileCheckpointState(state, artifacts) {
    const checkpoint = await readCheckpointMetadata(state.model.storage);
    if (!checkpoint) {
      return { restoredFromCheckpoint: false, checkpoint: null };
    }

    const previousTrainedEpochs = Math.max(Number(state.model.trainedEpochs) || 0, 0);
    const checkpointTrainedEpochs = Math.max(Number(checkpoint.manifest?.trainedEpochs) || 0, 0);
    const restoredFromCheckpoint = previousTrainedEpochs <= 0 && checkpointTrainedEpochs > 0;
    const restoredCorpusSignature = cleanText(checkpoint.manifest?.corpusSignature || '');

    if (restoredFromCheckpoint) {
      const checkpointModelSettings = checkpoint.manifest?.modelSettings || {};
      state.settings = {
        ...state.settings,
        training: {
          ...state.settings.training,
          ...checkpointModelSettings,
        },
      };
      state.model.trainedEpochs = checkpointTrainedEpochs;
      state.model.targetEpochs = Math.max(
        Number(state.model.targetEpochs) || 0,
        checkpointTrainedEpochs
      );
      state.model.lastTrainingAt = checkpoint.manifest?.savedAt || state.model.lastTrainingAt;
      state.model.corpusSignature = restoredCorpusSignature || state.model.corpusSignature;
      state.model.lifecycle = 'trained';
      state.model.status = 'ready';
      state.model.exists = true;
      state.model.configSnapshot = {
        ...structuredClone(state.settings),
        training: {
          ...state.settings.training,
        },
      };
    }

    if (!state.model.corpusSignature && restoredCorpusSignature) {
      state.model.corpusSignature = restoredCorpusSignature;
    }

    if (checkpoint.parameterCount > 0) {
      state.model.parameterCount = checkpoint.parameterCount;
    }

    const fallbackVocabularySize = Math.max(
      checkpoint.vocabularySize,
      Number(state.model.vocabularySize) || 0,
      Object.keys(artifacts?.vocabulary || {}).length
    );
    state.knowledge.languageModel = {
      ...createDefaultKnowledgeState().languageModel,
      ...(state.knowledge.languageModel || {}),
      kind: 'tf-keras-llm',
      checkpointReady: true,
      tokenizerReady: true,
      vocabularySize: fallbackVocabularySize,
      parameterCount: checkpoint.parameterCount || Number(state.knowledge.languageModel?.parameterCount) || 0,
      trainingSequenceCount: Math.max(
        Number(state.knowledge.languageModel?.trainingSequenceCount) || 0,
        Number(checkpoint.manifest?.trainingSequenceCount) || 0
      ),
      corpusTokenCount: Math.max(
        Number(state.knowledge.languageModel?.corpusTokenCount) || 0,
        Number(checkpoint.manifest?.corpusTokenCount) || 0
      ),
      feedbackExampleCount: Math.max(
        Number(state.knowledge.languageModel?.feedbackExampleCount) || 0,
        Number(checkpoint.manifest?.feedbackExampleCount) || 0
      ),
      lastSavedAt: checkpoint.manifest?.savedAt || state.knowledge.languageModel?.lastSavedAt || null,
    };

    return { restoredFromCheckpoint, checkpoint };
  }

  async function readWeightsBufferIfExists(weightsPath) {
    const normalizedPath = normalizeOptionalPath(weightsPath);
    if (!(await pathExists(normalizedPath))) {
      return null;
    }

    try {
      return await fs.readFile(normalizedPath);
    } catch (_error) {
      return null;
    }
  }

  async function collectModelArtifactsForExport(storage) {
    const normalizedStorage = normalizeStoragePaths(storage);
    const [manifest, knowledgeIndex, languageModel, tokenizer, weightsSpec, weightsBuffer] = await Promise.all([
      readJsonFileIfExists(normalizedStorage.manifestPath, null),
      readJsonFileIfExists(normalizedStorage.knowledgeIndexPath, null),
      readJsonFileIfExists(normalizedStorage.languageModelPath, null),
      readJsonFileIfExists(normalizedStorage.tokenizerPath, null),
      readJsonFileIfExists(normalizedStorage.neuralSpecPath, null),
      readWeightsBufferIfExists(normalizedStorage.neuralWeightsPath),
    ]);

    return {
      manifest,
      knowledgeIndex,
      languageModel,
      tokenizer,
      weightsSpec,
      weightsBinBase64: weightsBuffer ? weightsBuffer.toString('base64') : '',
    };
  }

  function isStringSizeLimitError(error) {
    const message = cleanText(error?.message || '').toLowerCase();
    return message.includes('cannot create a string longer than') || message.includes('invalid string length');
  }

  async function writeImportedArtifacts(storage, artifactsPayload = {}) {
    const normalizedStorage = normalizeStoragePaths(storage);
    if (!normalizedStorage.artifactDir || !normalizedStorage.neuralModelDir) {
      throw new Error('Пути хранилища артефактов не определены.');
    }

    await fs.mkdir(normalizedStorage.artifactDir, { recursive: true });
    await fs.mkdir(normalizedStorage.neuralModelDir, { recursive: true });

    const writeTasks = [];
    if (artifactsPayload.manifest) {
      writeTasks.push(fs.writeFile(normalizedStorage.manifestPath, JSON.stringify(artifactsPayload.manifest, null, 2), 'utf8'));
    }
    if (artifactsPayload.knowledgeIndex) {
      writeTasks.push(fs.writeFile(normalizedStorage.knowledgeIndexPath, JSON.stringify(artifactsPayload.knowledgeIndex, null, 2), 'utf8'));
    }
    if (artifactsPayload.languageModel) {
      writeTasks.push(fs.writeFile(normalizedStorage.languageModelPath, JSON.stringify(artifactsPayload.languageModel, null, 2), 'utf8'));
    }
    if (artifactsPayload.tokenizer) {
      writeTasks.push(fs.writeFile(normalizedStorage.tokenizerPath, JSON.stringify(artifactsPayload.tokenizer, null, 2), 'utf8'));
    }
    if (artifactsPayload.weightsSpec) {
      writeTasks.push(fs.writeFile(normalizedStorage.neuralSpecPath, JSON.stringify(artifactsPayload.weightsSpec, null, 2), 'utf8'));
    }
    if (typeof artifactsPayload.weightsBinBase64 === 'string' && artifactsPayload.weightsBinBase64.length) {
      writeTasks.push(fs.writeFile(normalizedStorage.neuralWeightsPath, Buffer.from(artifactsPayload.weightsBinBase64, 'base64')));
    }

    await Promise.all(writeTasks);
  }

  async function buildModelPackagePayload(state, options = {}) {
    const includeKnowledgeSnapshot = options.includeKnowledgeSnapshot !== false;
    const includeArtifactsPayload = options.includeArtifactsPayload !== false;
    const artifactsPayload = includeArtifactsPayload
      ? await collectModelArtifactsForExport(state.model.storage)
      : {};
    const artifactPaths = normalizeStoragePaths(state.model.storage);
    const hasCheckpointArtifacts = Boolean(
      artifactsPayload.tokenizer &&
      artifactsPayload.weightsSpec &&
      artifactsPayload.weightsBinBase64
    );

    if (options.strictCheckpoint && !hasCheckpointArtifacts) {
      throw new Error('Экспорт недоступен: нейросетевой чекпоинт не найден или поврежден.');
    }

    return {
      format: MODEL_PACKAGE_FORMAT,
      version: MODEL_PACKAGE_VERSION,
      engine: state.model.engine,
      exportedAt: nowIso(),
      snapshot: {
        settings: structuredClone(state.settings),
        model: normalizeModelForExport(structuredClone(state.model)),
        knowledge: includeKnowledgeSnapshot
          ? structuredClone(state.knowledge)
          : {
            ...createDefaultKnowledgeState(),
            languageModel: {
              ...createDefaultKnowledgeState().languageModel,
              ...(state.knowledge?.languageModel || {}),
            },
          },
      },
      runtimeConfig: structuredClone(getRuntimeConfig()),
      embeddedArtifacts: includeArtifactsPayload,
      artifacts: hasCheckpointArtifacts
        ? {
          ...artifactsPayload,
          manifest: {
            engine: state.model.engine,
            updatedAt: state.meta?.updatedAt || nowIso(),
            trainedEpochs: state.model.trainedEpochs,
            parameterCount: state.model.parameterCount,
            vocabularySize: state.model.vocabularySize,
            sourceCount: state.model.sourceCount,
            replyPairCount: state.model.replyPairCount,
            corpusSignature: state.model.corpusSignature,
            trainingSequenceCount: state.knowledge.languageModel.trainingSequenceCount,
            corpusTokenCount: state.knowledge.languageModel.corpusTokenCount,
            feedbackExampleCount: state.knowledge.languageModel.feedbackExampleCount,
            checkpointReady: state.knowledge.languageModel.checkpointReady,
            tokenizerReady: state.knowledge.languageModel.tokenizerReady,
            modelSettings: {
              ...buildTrainingSettingsSnapshot(state.model.configSnapshot?.training || state.settings.training),
              ...(state.model.configSnapshot?.training || {}),
            },
            computeBackend: state.model.computeBackend,
            computeBackendLabel: state.model.computeBackendLabel,
            files: artifactPaths,
          },
          knowledgeIndex: {
            chunks: state.knowledge.chunks,
            replyMemories: state.knowledge.replyMemories,
            vocabulary: state.knowledge.vocabulary,
            bm25: state.knowledge.bm25,
          },
          languageModel: state.knowledge.languageModel,
        }
        : {
          manifest: {
            engine: state.model.engine,
            updatedAt: state.meta?.updatedAt || nowIso(),
            trainedEpochs: state.model.trainedEpochs,
            parameterCount: state.model.parameterCount,
            vocabularySize: state.model.vocabularySize,
            sourceCount: state.model.sourceCount,
            replyPairCount: state.model.replyPairCount,
            corpusSignature: state.model.corpusSignature,
            checkpointReady: false,
            tokenizerReady: false,
            files: artifactPaths,
          },
          knowledgeIndex: {
            chunks: state.knowledge.chunks,
            replyMemories: state.knowledge.replyMemories,
            vocabulary: state.knowledge.vocabulary,
            bm25: state.knowledge.bm25,
          },
          languageModel: state.knowledge.languageModel,
          tokenizer: null,
          weightsSpec: null,
          weightsBinBase64: '',
        },
    };
  }

  async function saveActiveModelPackage(state) {
    const activeItem = updateActiveModelRegistrySummary(state);
    if (!activeItem) {
      return null;
    }

    let payload;
    try {
      payload = await buildModelPackagePayload(state, {
        strictCheckpoint: false,
        includeKnowledgeSnapshot: false,
        includeArtifactsPayload: false,
      });
    } catch (error) {
      if (!isStringSizeLimitError(error)) {
        throw error;
      }
      log('warn', 'Model library package exceeded string limit; storing compact package.', {
        scope: 'model_library',
        modelId: activeItem.id,
      });
      payload = {
        format: MODEL_PACKAGE_FORMAT,
        version: MODEL_PACKAGE_VERSION,
        engine: state.model.engine,
        exportedAt: nowIso(),
        snapshot: {
          settings: structuredClone(state.settings),
          model: normalizeModelForExport(structuredClone(state.model)),
          knowledge: {
            ...createDefaultKnowledgeState(),
            languageModel: {
              ...createDefaultKnowledgeState().languageModel,
              ...(state.knowledge?.languageModel || {}),
            },
          },
        },
        runtimeConfig: structuredClone(getRuntimeConfig()),
        embeddedArtifacts: false,
        artifacts: {
          manifest: {
            engine: state.model.engine,
            updatedAt: state.meta?.updatedAt || nowIso(),
            trainedEpochs: state.model.trainedEpochs,
            parameterCount: state.model.parameterCount,
            vocabularySize: state.model.vocabularySize,
            sourceCount: state.model.sourceCount,
            replyPairCount: state.model.replyPairCount,
            corpusSignature: state.model.corpusSignature,
            checkpointReady: false,
            tokenizerReady: false,
            files: normalizeStoragePaths(state.model.storage),
          },
          knowledgeIndex: {
            chunks: [],
            replyMemories: [],
            vocabulary: {},
            bm25: createDefaultKnowledgeState().bm25,
          },
          languageModel: {
            ...createDefaultKnowledgeState().languageModel,
            ...(state.knowledge?.languageModel || {}),
          },
          tokenizer: null,
          weightsSpec: null,
          weightsBinBase64: '',
        },
      };
    }
    const packagePath = await writeModelLibraryPackage(activeItem.id, payload);
    activeItem.packagePath = packagePath;
    activeItem.updatedAt = nowIso();
    activeItem.lastUsedAt = nowIso();
    activeItem.hasCheckpoint = Boolean(
      Number(state.model?.trainedEpochs || 0) > 0 &&
      state.knowledge?.languageModel?.checkpointReady
    );
    return activeItem;
  }

  async function applyModelPackageToState(state, modelPackage, options = {}) {
    if (!modelPackage || typeof modelPackage !== 'object') {
      throw new Error('Некорректный пакет модели: ожидается JSON-объект.');
    }
    if (modelPackage.format !== MODEL_PACKAGE_FORMAT || Number(modelPackage.version) !== MODEL_PACKAGE_VERSION) {
      throw new Error('Неподдерживаемый формат пакета модели.');
    }

    const strictArtifacts = options.strictArtifacts !== false;
    const packageArtifacts = modelPackage.artifacts || {};
    const hasCheckpointArtifacts = Boolean(
      packageArtifacts.tokenizer &&
      packageArtifacts.weightsSpec &&
      packageArtifacts.weightsBinBase64
    );

    if (strictArtifacts && !hasCheckpointArtifacts) {
      throw new Error('Пакет не содержит обязательные артефакты чекпоинта (tokenizer/spec/weights).');
    }

    await disposeNeuralRuntime();
    activeTrainingRollbackSnapshot = null;

    if (hasCheckpointArtifacts) {
      await writeImportedArtifacts(state.model.storage, packageArtifacts);
    } else if (modelPackage.embeddedArtifacts !== false) {
      await removeModelArtifacts(state.model.storage);
    }

    if (modelPackage.runtimeConfig && typeof modelPackage.runtimeConfig === 'object') {
      await updateRuntimeConfig(modelPackage.runtimeConfig);
    }

    const incomingSettings = modelPackage.snapshot?.settings || {};
    state.settings = {
      training: {
        ...state.settings.training,
        ...(incomingSettings.training || {}),
      },
      generation: {
        ...state.settings.generation,
        ...(incomingSettings.generation || {}),
      },
    };

    const preservedStorage = structuredClone(state.model.storage);
    const preservedArtifactFiles = [...(state.model.artifactFiles || [])];
    const importedModel = modelPackage.snapshot?.model || {};
    const importedKnowledge = modelPackage.snapshot?.knowledge || {};

    state.model = {
      ...createDefaultModelState(),
      ...importedModel,
      storage: preservedStorage,
      artifactFiles: preservedArtifactFiles,
    };
    state.knowledge = {
      ...createDefaultKnowledgeState(),
      ...importedKnowledge,
    };
    if (packageArtifacts.languageModel && typeof packageArtifacts.languageModel === 'object') {
      state.knowledge.languageModel = {
        ...createDefaultKnowledgeState().languageModel,
        ...packageArtifacts.languageModel,
      };
    }

    const artifacts = await rebuildKnowledge(state);
    const { restoredFromCheckpoint } = await reconcileCheckpointState(state, artifacts);

    state.model.exists = true;
    state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
    state.model.status = 'ready';
    state.model.configSnapshot = structuredClone(state.settings);
    state.training = {
      ...createDefaultTrainingState(),
      status: 'idle',
      phase: options.phase || 'model_imported',
      message: restoredFromCheckpoint
        ? 'Модель загружена и восстановлена из чекпоинта.'
        : 'Модель загружена без нейросетевого чекпоинта. Ее можно дообучить или использовать как профиль.',
      updatedAt: nowIso(),
    };
    restoreQueuesAfterModelReset(state);
    updateActiveModelRegistrySummary(state);

    pushStatus(
      state,
      'idle',
      state.training.phase,
      state.training.message
    );
  }

  async function updateStoredCheckpointManifest(storage, updates = {}) {
    if (!storage?.neuralSpecPath) {
      return;
    }

    const existingBundle = await readJsonFileIfExists(storage.neuralSpecPath, null);
    if (!existingBundle || typeof existingBundle !== 'object') {
      return;
    }

    await writeJsonFile(storage.neuralSpecPath, {
      ...existingBundle,
      manifest: {
        ...(existingBundle.manifest || {}),
        ...updates,
      },
    });
  }

  async function ensureRuntimeLoaded(state) {
    if (neuralRuntime?.model) {
      return neuralRuntime;
    }

    if (!state.knowledge.languageModel?.checkpointReady) {
      return null;
    }

    neuralRuntime = await loadRuntime({
      storage: state.model.storage,
      settings: state.settings,
    });

    return neuralRuntime;
  }

  async function rebuildKnowledge(state, options = {}) {
    const artifacts = buildKnowledgeArtifacts(state, options);
    updateModelMetricsFromArtifacts(state, artifacts);
    return artifacts;
  }

  function shouldAutoClearSources(state) {
    return state.settings.training.autoClearSourcesAfterTraining !== false;
  }

  function appendSources(state, sources) {
    const preparedSources = sources
      .map((source) => {
        const text = cleanText(source.content);
        if (!text) {
          return null;
        }

        const fallbackStats = computeStats(text);
        const sourceStats = source?.stats && typeof source.stats === 'object' ? source.stats : null;
        const tokenCount = Math.max(
          Number(sourceStats?.tokenCount) || 0,
          Number(fallbackStats.tokenCount) || 0,
          0
        );
        const charCount = Math.max(
          Number(sourceStats?.charCount) || 0,
          Number(fallbackStats.charCount) || 0,
          0
        );
        const rowCount = Math.max(Number(sourceStats?.rowCount) || 0, 0);
        const columnCount = Math.max(Number(sourceStats?.columnCount) || 0, 0);
        const format = cleanText(sourceStats?.format || '');
        const columns = Array.isArray(sourceStats?.columns)
          ? sourceStats.columns.map((value) => cleanText(String(value))).filter(Boolean).slice(0, 64)
          : [];
        const contentPath = cleanText(source?.contentPath || source?.datasetFilePath || '');
        const contentSize = Math.max(
          Number(source?.contentSize) || 0,
          text ? Buffer.byteLength(text, 'utf8') : 0
        );
        const mergedStats = {
          tokenCount,
          charCount,
          ...(rowCount > 0 ? { rowCount } : {}),
          ...(columnCount > 0 ? { columnCount } : {}),
          ...(format ? { format } : {}),
          ...(columns.length ? { columns } : {}),
        };

        return {
          id: createId('source'),
          type: source.type,
          label: source.label,
          url: source.url || null,
          content: text,
          stats: mergedStats,
          addedAt: nowIso(),
          contentPath: contentPath || null,
          contentSize,
        };
      })
      .filter(Boolean);

    if (!preparedSources.length) {
      throw new Error('Источник не содержит текста.');
    }

    state.sources.unshift(...preparedSources);
    state.model.sourceCount = state.sources.length;
    if (state.model.exists) {
      state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
      state.model.status = 'ready';
    }

    return preparedSources;
  }

  async function ensureModelExists() {
    await updateState(async (state) => {
      ensureChatAvailability(state);
      if (!state.model.exists) {
        state.model = createDefaultModelState();
        state.training = createDefaultTrainingState();
        state.knowledge = createDefaultKnowledgeState();
        state.model.exists = true;
        state.model.lifecycle = 'ready_for_training';
        state.model.status = 'ready';
        pushStatus(state, 'idle', 'model_ready', 'Серверная нейросеть подготовлена к обучению.');
      } else {
        state.model.exists = true;
      }

      state.model.configSnapshot = structuredClone(state.settings);
    }, {
      writeSources: false,
      writeChats: false,
      writeArtifacts: false,
    });
  }

  function buildTrainingManifest(state, artifacts) {
    return {
      version: 1,
      corpusSignature: artifacts.trainingCorpusSignature,
      modelSettings: buildTrainingSettingsSnapshot(state.settings.training),
      vocabularySize: 0,
      parameterCount: 0,
      trainedEpochs: state.model.trainedEpochs,
      trainingSequenceCount: state.model.trainingItemCount,
      corpusTokenCount: 0,
      feedbackExampleCount: artifacts.positiveFeedbackCount,
      savedAt: nowIso(),
    };
  }

  async function runTrainingJob(options = {}) {
    if (activeTrainingPromise) {
      return activeTrainingPromise;
    }

    const queueContext = options.queueContext || null;
    const queuedSourcesOverride = Array.isArray(options.queuedSourcesOverride)
      ? options.queuedSourcesOverride
      : null;
    let trainingPayloadPathForCleanup = '';

    createTrainingStartSignal();
    const stopSignal = createStopSignal();
    activeTrainingPromise = (async () => {
      await updateState(async (state) => {
        state.model.exists = true;
        state.model.lifecycle = 'training';
        state.model.status = 'training';
        state.model.configSnapshot = structuredClone(state.settings);
        state.training = {
          ...state.training,
          ...createDefaultTrainingState(),
          status: 'training',
          phase: 'building_tokenizer',
          message: 'Сервер готовит корпус, токенизатор и отдельный воркер обучения.',
          startedAt: nowIso(),
          updatedAt: nowIso(),
          requestedStop: false,
        };
        pushStatus(state, 'training', 'building_tokenizer', 'Подготовка корпуса и запуск отдельного процесса обучения.');
      }, {
        writeSources: false,
        writeChats: false,
        writeArtifacts: false,
      });
      resolveTrainingStartSignal();

      const initialState = getState();
      let materializedQueuedSourcesForRun = null;
      let trainingPayloadPath = '';
      let lastPreparationProgressBucket = '';
      let lastPreparationPhase = '';
      const reportPreparationProgress = (progress = {}) => {
        const stage = progress.stage || 'building_training_corpus';
        const bucket = `${stage}:${progress.phase || ''}:${Math.floor((Number(progress.percent) || 0) / 10)}`;
        if (bucket === lastPreparationProgressBucket) {
          return;
        }
        lastPreparationProgressBucket = bucket;

        const stageMessageMap = {
          loading_queue_sources: progress.label
            ? `Подготовка очереди обучения: загрузка файла «${progress.label}» (${progress.processedSources}/${progress.totalSources}).`
            : `Подготовка очереди обучения: загружено файлов ${progress.processedSources}/${progress.totalSources}.`,
          extracting_dialogues: progress.label
            ? `Подготовка очереди обучения: разбор файла «${progress.label}» (${progress.processedSources}/${progress.totalSources}).`
            : `Подготовка очереди обучения: разобрано файлов ${progress.processedSources}/${progress.totalSources}.`,
          building_training_texts: progress.totalItems
            ? `Подготовка обучающего корпуса: ${progress.processedItems}/${progress.totalItems} элементов.`
            : 'Подготовка обучающего корпуса для воркера.',
        };
        const message = stageMessageMap[stage] || 'Подготовка корпуса и запуск отдельного процесса обучения.';
        const percent = stage === 'loading_queue_sources'
          ? Math.min(10, 2 + Math.round((Number(progress.percent) || 0) * 0.08))
          : stage === 'extracting_dialogues'
            ? Math.min(16, 10 + Math.round((Number(progress.percent) || 0) * 0.06))
            : Math.min(22, 16 + Math.round((Number(progress.percent) || 0) * 0.06));

        updateState(async (state) => {
          if (state.training.status !== 'training') {
            return;
          }
          state.training.phase = stage;
          state.training.message = message;
          state.training.updatedAt = nowIso();
          state.training.progress = {
            ...state.training.progress,
            currentEpoch: 0,
            totalEpochs: Math.max(Number(state.settings.training.epochs) || 1, 1),
            currentBatch: 0,
            totalBatches: 0,
            percent: Math.max(state.training.progress.percent || 0, percent),
          };
          if (lastPreparationPhase !== stage) {
            pushStatus(state, 'training', stage, message, { updateTrainingState: false });
            lastPreparationPhase = stage;
          }
        }, { persist: false }).catch(() => {});
      };

      let initialTrainingArtifacts;
      try {
        initialTrainingArtifacts = await buildTrainingSessionArtifactsAsync(initialState, {
          includeQueuedSources: true,
          queuedSourcesOverride,
          shouldStop: () => isTrainingStopRequested(),
          onProgress: reportPreparationProgress,
        });
        materializedQueuedSourcesForRun = initialTrainingArtifacts.materializedQueuedSources || null;
      } catch (error) {
        if (error?.code === 'TRAINING_PREPARATION_STOPPED') {
          return;
        }
        throw error;
      }

      if (isTrainingStopRequested()) {
        return;
      }

      const trainingTexts = initialTrainingArtifacts.trainingTexts;
      const datasetFiles = initialTrainingArtifacts.datasetFiles || [];

      if (!trainingTexts.length && !datasetFiles.length) {
        throw new Error('Для обучения не хватает данных. Добавьте источники или диалоговые пары.');
      }

      log('info', 'Training corpus prepared for worker startup.', {
        trainingTextCount: trainingTexts.length,
        datasetFileCount: datasetFiles.length,
        sourceDocumentCount: initialTrainingArtifacts.sourceDocuments.length,
        replayPairCount: initialTrainingArtifacts.replayReplyMemories.length,
      });

      activeTrainingRollbackSnapshot = null;
      const hasStableCheckpoint = Boolean(
        initialState.knowledge.languageModel?.checkpointReady &&
        (Number(initialState.model.trainedEpochs) || 0) > 0
      );
      if (hasStableCheckpoint) {
        try {
          activeTrainingRollbackSnapshot = await createRollbackCheckpoint(initialState);
        } catch (error) {
          throw new Error(
            `Не удалось создать резервную копию чекпоинта перед дообучением: ${cleanText(error.message || 'ошибка файловой системы')}. Повторите запуск обучения.`
          );
        }
      }

      if (isTrainingStopRequested()) {
        return;
      }

      const checkpointTrainingSettings = buildTrainingSettingsSnapshot(initialState.model.configSnapshot?.training || {});
      const requestedTrainingSettings = buildTrainingSettingsSnapshot(initialState.settings.training || {});
      const canResumeFromCheckpoint =
        initialState.knowledge.languageModel?.checkpointReady &&
        checkpointTrainingSettings.sequenceLength === requestedTrainingSettings.sequenceLength &&
        checkpointTrainingSettings.embeddingSize === requestedTrainingSettings.embeddingSize &&
        checkpointTrainingSettings.attentionHeads === requestedTrainingSettings.attentionHeads &&
        checkpointTrainingSettings.transformerLayers === requestedTrainingSettings.transformerLayers &&
        checkpointTrainingSettings.feedForwardSize === requestedTrainingSettings.feedForwardSize;
      const requestedResumeEpochOffset = canResumeFromCheckpoint
        ? Math.max(Number(initialState.model.trainedEpochs) || 0, 0)
        : 0;

      await disposeNeuralRuntime();

      if (isTrainingStopRequested()) {
        return;
      }

      const workerPath = path.join(__dirname, 'trainingWorker.js');
      const manifest = buildTrainingManifest(initialState, initialTrainingArtifacts);
      trainingPayloadPath = await createTrainingJobPayloadFile({
        trainingTexts,
        datasetFiles: datasetFiles.map((entry) => entry.path || entry).filter(Boolean),
        datasetOptions: {
          maxRecordsPerFile: MAX_DATASET_RECORDS_PER_FILE,
          maxCharsPerRecord: MAX_DATASET_CHARS_PER_RECORD,
          parquetBatchSize: DATASET_PARQUET_BATCH_SIZE,
        },
      });
      trainingPayloadPathForCleanup = trainingPayloadPath;

      await new Promise((resolve, reject) => {
        const worker = new Worker(workerPath, {
          workerData: {
            settings: initialState.settings,
            trainingPayloadPath,
            storage: initialState.model.storage,
            resumeFromCheckpoint: Boolean(canResumeFromCheckpoint),
            resumeEpochOffset: requestedResumeEpochOffset,
            manifest,
            positiveFeedbackCount: initialTrainingArtifacts.positiveFeedbackCount,
            stopSignalBuffer: stopSignal.buffer,
          },
        });

        activeTrainingWorker = worker;
        let finished = false;
        let workerQueue = Promise.resolve();
        let knownBatchesPerEpoch = 1;
        let lastPreparationStage = '';

        const finalizeFailure = (error) => {
          if (finished) {
            return;
          }

          if (
            activeTrainingTerminationMode === 'model_deleted' ||
            activeTrainingTerminationMode === 'rollback_to_checkpoint' ||
            activeTrainingTerminationMode === 'pause_requested'
          ) {
            finalizeSuccess();
            return;
          }

          finished = true;
          activeTrainingWorker = null;
          logError('Training worker failed.', error);
          reject(error);
        };

        const finalizeSuccess = () => {
          if (finished) {
            return;
          }

          finished = true;
          activeTrainingWorker = null;
          resolve();
        };

        const enqueueWorkerUpdate = (task) => {
          workerQueue = workerQueue.then(task).catch(finalizeFailure);
        };

        worker.on('message', (message) => {
          if (!message?.type || finished) {
            return;
          }

          if (message.type === 'error') {
            finalizeFailure(new Error(message.error?.message || 'Ошибка воркера обучения.'));
            return;
          }

          if (message.type === 'preparing') {
            enqueueWorkerUpdate(async () => {
              await updateState(async (state) => {
                state.training.status = 'training';
                state.training.phase = message.stage || 'building_tokenizer';
                state.training.message = cleanText(message.message || state.training.message || 'Подготовка обучения.');
                state.training.updatedAt = nowIso();
                state.training.progress = {
                  ...state.training.progress,
                  currentEpoch: 0,
                  totalEpochs: Math.max(Number(state.settings.training.epochs) || 1, 1),
                  currentBatch: 0,
                  totalBatches: 0,
                  percent: Math.max(Number(message.progress) || 0, state.training.progress.percent || 0),
                };

                if (lastPreparationStage !== state.training.phase) {
                  pushStatus(
                    state,
                    'training',
                    state.training.phase,
                    state.training.message,
                    { updateTrainingState: false }
                  );
                  lastPreparationStage = state.training.phase;
                }
              }, {
                persist: false,
              });
            });
            return;
          }

          if (message.type === 'prepared') {
            enqueueWorkerUpdate(async () => {
              knownBatchesPerEpoch = Math.max(message.batchesPerEpoch || 1, 1);
              await updateState(async (state) => {
                const batchSizeNote = message.batchSizeAdjusted
                  ? ` Размер батча автоматически скорректирован: ${message.requestedBatchSize} → ${message.effectiveBatchSize}, чтобы уменьшить число батчей и ускорить эпоху.`
                  : '';
                const backendWarningNote = message.backendWarning
                  ? ` ${cleanText(message.backendWarning)}`
                  : '';
                const fallbackReasonMap = {
                  checkpoint_unavailable: 'чекпоинт недоступен',
                  checkpoint_incompatible: 'чекпоинт несовместим',
                };
                const resumedEpochs = Math.max(Number(message.effectiveResumeEpochOffset) || 0, 0);
                const scheduledEpochs = Math.max(Number(message.requestedEpochs) || state.settings.training.epochs || 1, 1);
                const totalEpochsPlanned = Math.max(Number(message.effectiveEpochs) || (resumedEpochs + scheduledEpochs), 1);
                const resumeNote = message.resumedFromCheckpoint
                  ? resumedEpochs > 0
                    ? ` Дообучение продолжается с ${resumedEpochs + 1}-й эпохи: +${scheduledEpochs}, итоговый план ${totalEpochsPlanned} эпох.`
                    : ` Обучение стартует с новой модели на ${scheduledEpochs} эпох.`
                  : message.requestedResumeEpochOffset > 0
                    ? ` Возобновление чекпоинта недоступно (${fallbackReasonMap[message.resumeRestartReason] || 'изменился токенизатор'}), обучение перезапущено с нуля на ${scheduledEpochs} эпох.`
                    : '';
                state.model.trainingItemCount = message.trainingSampleCount;
                state.model.batchesPerEpoch = message.batchesPerEpoch;
                state.model.parameterCount = message.parameterCount;
                state.model.computeBackend = message.backendName || state.model.computeBackend;
                state.model.computeBackendLabel = message.backendLabel || state.model.computeBackendLabel;
                state.model.computeBackendWarning = message.backendWarning || '';
                state.model.targetEpochs = totalEpochsPlanned;
                state.training.phase = 'training_batches';
                state.training.message = `Подготовлено ${message.trainingSampleCount} обучающих окон, ${message.validationSampleCount} валидационных и ${message.batchesPerEpoch} батчей на эпоху. Backend: ${message.backendLabel || message.backendName}.${batchSizeNote}${resumeNote}${backendWarningNote}`;
                pushStatus(
                  state,
                  'training',
                  'training_batches',
                  `Нейросеть обучается в отдельном воркере: ${message.parameterCount} параметров, ${message.trainingSampleCount} обучающих окон и ${message.validationSampleCount} окон для валидации. Backend: ${message.backendLabel || message.backendName}.${batchSizeNote}${resumeNote}${backendWarningNote}`
                );
              }, {
                writeSources: false,
                writeChats: false,
                writeArtifacts: false,
              });
            });
            return;
          }

          if (message.type === 'heartbeat') {
            enqueueWorkerUpdate(async () => {
              await updateState(async (state) => {
                if (state.training.status !== 'training') {
                  return;
                }

                state.training.phase = 'training_batches';
                state.training.updatedAt = nowIso();
                state.training.message = `Обучение продолжается: эпоха ${message.epoch}/${message.effectiveEpochs}, выполняется батч ${message.batch}/${message.batchesThisEpoch}.`;
              }, { persist: false });
            });
            return;
          }

          if (message.type === 'batch') {
            enqueueWorkerUpdate(async () => {
              knownBatchesPerEpoch = Math.max(message.batchesThisEpoch || knownBatchesPerEpoch, 1);
              const totalEpochs = Math.max(message.effectiveEpochs || initialState.settings.training.epochs, 1);
              const totalBatches = Math.max(message.batchesThisEpoch * totalEpochs, 1);
              const completedBatches = Math.min(
                Math.max(
                  ((Math.max(message.epoch, 1) - 1) * Math.max(message.batchesThisEpoch, 1)) +
                    Math.max(message.batch, 1),
                  1
                ),
                totalBatches
              );
              await updateState(async (state) => {
                state.training.status = 'training';
                state.training.phase = 'training_batches';
                state.model.batchesPerEpoch = message.batchesThisEpoch;
                state.model.targetEpochs = totalEpochs;
                state.training.message = message.enforcedStepBudget
                  ? `Обучение идет: эпоха ${message.epoch}/${totalEpochs}, батч ${message.batch}/${message.batchesThisEpoch}. Минимальный бюджет обучения увеличил число эпох.`
                  : `Обучение идет: эпоха ${message.epoch}/${totalEpochs}, батч ${message.batch}/${message.batchesThisEpoch}.`;
                state.training.progress = {
                  currentEpoch: message.epoch,
                  totalEpochs,
                  currentBatch: message.batch,
                  totalBatches: message.batchesThisEpoch,
                  percent: Number(((completedBatches / totalBatches) * 100).toFixed(1)),
                };
                state.training.history = [...state.training.history, message.historyEntry].slice(-state.settings.training.maxHistoryPoints);
                state.training.updatedAt = nowIso();
                state.model.lastLoss = message.lastLoss;
                state.model.averageLoss = message.averageLoss;
                state.model.perplexity = message.averageLoss ? Number(Math.exp(message.averageLoss).toFixed(4)) : null;
                state.model.batchesProcessed = completedBatches;
              }, { persist: false });
            });
            return;
          }

          if (message.type === 'checkpointing') {
            enqueueWorkerUpdate(async () => {
              await updateState(async (state) => {
                state.model.lifecycle = 'syncing_knowledge';
                state.model.status = 'saving_checkpoint';
                state.training.phase = 'saving_checkpoint';
                state.training.message = 'Обучение завершено, сервер сохраняет веса модели и токенизатор.';
                pushStatus(
                  state,
                  'syncing_knowledge',
                  'saving_checkpoint',
                  'Сохранение нейросетевого чекпоинта и артефактов retriever.'
                );
              }, { persist: false });
            });
            return;
          }

          if (message.type === 'done') {
            enqueueWorkerUpdate(async () => {
              await disposeNeuralRuntime();
              await updateState(async (state) => {
                state.model.exists = true;
                state.model.lastLoss = message.trainingResult.lastLoss;
                state.model.averageLoss = message.trainingResult.averageLoss;
                state.model.validationLoss = message.trainingResult.validationLoss;
                state.model.bestValidationLoss = message.trainingResult.bestValidationLoss;
                state.model.perplexity = message.trainingResult.perplexity;
                state.model.lastTrainingAt = nowIso();
                state.model.batchesProcessed = message.trainingResult.processedBatches;
                state.model.targetEpochs = message.trainingResult.effectiveEpochs;
                state.model.parameterCount = message.parameterCount;

                const currentArtifacts = await rebuildKnowledge(state, {
                  includeQueuedSources: true,
                  queuedSourcesOverride: materializedQueuedSourcesForRun || queuedSourcesOverride,
                });

                state.model.trainedEpochs = message.trainingResult.completedEpochs;
                state.model.corpusSignature = currentArtifacts.corpusSignature;
                state.knowledge.languageModel = message.languageModel;
                state.model.lifecycle = message.trainingResult.stopRequested ? 'paused' : 'trained';
                state.model.status = 'ready';
                if (!message.trainingResult.stopRequested && queueContext?.queueId) {
                  markQueueAsCompleted(
                    state,
                    queueContext.queueId,
                    queueContext.queueName,
                    state.model.lastTrainingAt
                  );
                }
                if (message.trainingResult.stopRequested && queueContext?.queueId) {
                  markQueueAsPendingAfterInterruption(state, queueContext.queueId, 'paused');
                }
                if (!queueContext?.queueId && !message.trainingResult.stopRequested && shouldAutoClearSources(state)) {
                  await Promise.all((state.sources || []).map(async (source) => {
                    const datasetPath = resolveDatasetSourcePath(source);
                    if (!datasetPath) {
                      return;
                    }
                    try {
                      await fs.rm(datasetPath, { force: true });
                    } catch (_error) {
                      // Ignore cleanup failures for source dataset files.
                    }
                  }));
                  state.sources = [];
                  state.model.sourceCount = 0;
                }
                state.training.status = message.trainingResult.stopRequested ? 'paused' : 'completed';
                state.training.phase = message.trainingResult.stopRequested ? 'paused_by_user' : 'ready_for_chat';
                const remainingQueueCount = queueContext?.queueId
                  ? state.trainingQueues.runner.pendingQueueIds.length
                  : 0;
                state.training.message = message.trainingResult.stopRequested
                  ? 'Обучение остановлено пользователем. Последний чекпоинт сохранен.'
                  : queueContext?.queueId
                    ? remainingQueueCount
                      ? `Очередь «${queueContext.queueName}» успешно применена. Осталось очередей: ${remainingQueueCount}.`
                      : `Автоочередь завершена. Последней успешно применена очередь «${queueContext.queueName}».`
                    : shouldAutoClearSources(state)
                      ? 'Обучение завершено. Чекпоинт и индекс знаний сохранены, очередь источников очищена.'
                      : 'Обучение завершено. Нейросеть, токенизатор и индекс знаний сохранены на сервере.';
                state.training.progress = {
                  currentEpoch: message.trainingResult.completedEpochs,
                  totalEpochs: message.trainingResult.effectiveEpochs,
                  currentBatch: 0,
                  totalBatches: knownBatchesPerEpoch,
                  percent: message.trainingResult.stopRequested ? state.training.progress.percent : 100,
                };
                pushStatus(
                  state,
                  message.trainingResult.stopRequested ? 'paused' : 'completed',
                  message.trainingResult.stopRequested ? 'paused_by_user' : 'ready_for_chat',
                  state.training.message
                );

                await updateStoredCheckpointManifest(state.model.storage, {
                  corpusSignature: currentArtifacts.corpusSignature,
                  corpusTokenCount: message.languageModel?.corpusTokenCount || 0,
                  feedbackExampleCount: message.languageModel?.feedbackExampleCount || 0,
                  trainedEpochs: message.trainingResult.completedEpochs,
                  savedAt: state.model.lastTrainingAt,
                });
              }, {
                writeChats: false,
              });
              await updateState(async (state) => {
                ensureModelRegistryState(state);
                updateActiveModelRegistrySummary(state);
                await saveActiveModelPackage(state);
              }, {
                writeSources: false,
                writeChats: false,
              });
              log('info', 'Training worker finished successfully.', {
                completedEpochs: message.trainingResult.completedEpochs,
                processedBatches: message.trainingResult.processedBatches,
                stopRequested: message.trainingResult.stopRequested,
              });
              activeTrainingRollbackSnapshot = null;
              finalizeSuccess();
            });
          }
        });

        worker.on('error', finalizeFailure);
        worker.on('exit', (code) => {
          if (finished || code === 0) {
            return;
          }

          finalizeFailure(new Error(`Воркер обучения завершился с кодом ${code}.`));
        });
      });
    })().catch((error) => {
      rejectTrainingStartSignal(error);
      throw error;
    }).finally(async () => {
      try {
        await removeTrainingJobPayloadFile(trainingPayloadPathForCleanup);
      } catch (_error) {
        // Ignore temp payload cleanup failures.
      }
      activeTrainingPromise = null;
      activeTrainingStartPromise = null;
      activeTrainingWorker = null;
      activeTrainingStopSignal = null;
      activeTrainingTerminationMode = null;
    });

    return activeTrainingPromise;
  }

  async function runTrainingQueueCampaign() {
    if (activeTrainingQueueRunnerPromise) {
      return activeTrainingQueueRunnerPromise;
    }

    activeTrainingQueueRunnerPromise = (async () => {
      await updateState(async (state) => {
        prepareTrainingQueuesState(state);
        const pendingQueues = getPendingTrainingQueues(state);
        if (!pendingQueues.length) {
          throw new Error('Нет непустых очередей обучения для запуска.');
        }

        const startedAt = nowIso();
        state.trainingQueues.items = state.trainingQueues.items.map((queue) => ({
          ...queue,
          status: queue.status === 'completed'
            ? 'completed'
            : (Array.isArray(queue.sources) && queue.sources.length ? 'pending' : 'empty'),
          lastError: queue.status === 'completed' ? queue.lastError || '' : '',
          updatedAt: startedAt,
        }));
        state.trainingQueues.runner = {
          ...createDefaultTrainingQueuesState().runner,
          active: true,
          status: 'running',
          startedAt,
          updatedAt: startedAt,
          totalQueues: pendingQueues.length,
          pendingQueueIds: pendingQueues.map((queue) => queue.id),
          completedQueueIds: [],
        };

        pushStatus(
          state,
          'training',
          'queue_runner_started',
          `Запущена автоочередь обучения: ${pendingQueues.length} очередей будут применены последовательно.`,
          { updateTrainingState: false }
        );
      }, {
        writeSources: false,
        writeChats: false,
        writeArtifacts: false,
      });

      while (true) {
        const runnerState = getState().trainingQueues?.runner;
        if (!runnerState?.active) {
          break;
        }

        const nextQueueId = runnerState.pendingQueueIds?.[0];
        if (!nextQueueId) {
          break;
        }

        const currentState = getState();
        const nextQueue = (currentState.trainingQueues?.items || []).find((queue) => queue.id === nextQueueId);

        if (!nextQueue || !Array.isArray(nextQueue.sources) || !nextQueue.sources.length) {
          await updateState(async (state) => {
            prepareTrainingQueuesState(state);
            state.trainingQueues.runner.pendingQueueIds = state.trainingQueues.runner.pendingQueueIds
              .filter((queueId) => queueId !== nextQueueId);
            if (!state.trainingQueues.runner.pendingQueueIds.length) {
              state.trainingQueues.runner.active = false;
              state.trainingQueues.runner.status = 'completed';
              state.trainingQueues.runner.currentQueueId = null;
              state.trainingQueues.runner.updatedAt = nowIso();
            }
          }, {
            writeSources: false,
            writeChats: false,
            writeArtifacts: false,
          });
          continue;
        }

        await updateState(async (state) => {
          const queue = markQueueAsRunning(state, nextQueue.id);
          pushStatus(
            state,
            'training',
            'queue_runner_next',
            `Автоочередь запускает очередь ${state.trainingQueues.runner.currentQueueIndex}/${state.trainingQueues.runner.totalQueues}: «${queue.name}» (${queue.sources.length} файлов).`,
            { updateTrainingState: false }
          );
        }, {
          writeSources: false,
          writeChats: false,
          writeArtifacts: false,
        });

        try {
          await runTrainingJob({
            queuedSourcesOverride: cloneTrainingQueueSources(nextQueue.sources),
            queueContext: {
              queueId: nextQueue.id,
              queueName: nextQueue.name,
            },
          });
        } catch (error) {
          await updateState(async (state) => {
            prepareTrainingQueuesState(state);
            markQueueAsPendingAfterInterruption(state, nextQueue.id, 'error', error.message);
            const queue = state.trainingQueues.items.find((entry) => entry.id === nextQueue.id);
            if (queue) {
              queue.status = 'error';
              queue.updatedAt = nowIso();
              queue.lastError = cleanText(error.message || '');
            }
            state.trainingQueues.runner.status = 'error';
            state.trainingQueues.runner.failedQueueId = nextQueue.id;

            await disposeNeuralRuntime();
            await restoreLastStableCheckpointState(state, {
              phase: 'queue_runner_failed_rollback',
              restoredMessage: 'Автоочередь остановлена из-за ошибки. Восстановлен последний стабильный чекпоинт модели.',
              fallbackTrainedMessage: 'Автоочередь остановлена из-за ошибки. Последний стабильный чекпоинт модели остался без изменений.',
              fallbackFreshMessage: 'Автоочередь остановлена до создания нового чекпоинта.',
            });

            pushStatus(
              state,
              'error',
              'queue_runner_failed',
              `Автоочередь остановлена на очереди «${nextQueue.name}»: ${cleanText(error.message || 'ошибка обучения')}.`,
              { updateTrainingState: false }
            );
          }, {
            writeSources: false,
            writeChats: false,
          });
          break;
        }

        const stateAfterQueue = getState();
        if (
          !stateAfterQueue.trainingQueues?.runner?.active ||
          stateAfterQueue.training.status === 'paused' ||
          stateAfterQueue.trainingQueues.runner.status === 'error'
        ) {
          break;
        }
      }
    })().finally(() => {
      activeTrainingQueueRunnerPromise = null;
    });

    return activeTrainingQueueRunnerPromise;
  }

  async function renderReply({ state, chat, userMessage }) {
    const runtimeConfig = getRuntimeConfig();
    const userLanguage = detectMessageLanguage(userMessage);
    if (isSimpleGreetingMessage(userMessage)) {
      return buildReplyPayload(buildGreetingReply(userLanguage), {
        maxSentences: state.settings.generation.maxReplySentences,
        maxCharacters: state.settings.generation.maxReplyCharacters,
        userMessage,
        metadata: {
          mode: 'system',
          type: 'system',
          usedContextMessages: 0,
          matchedSourceLabels: [],
          matchedMemoryScore: 0,
          chunkMatches: 0,
          selfScore: 0.92,
          userRating: 0,
          usedWebSources: [],
          usedWebSourceEntries: [],
        },
      });
    }

    const { queryMap, contextMessages } = buildWeightedQueryMap(userMessage, chat, runtimeConfig);
    const recentAssistantReplies = (chat.messages || [])
      .filter((message) => message.role === 'assistant')
      .filter((message) => message.metadata?.type !== 'system')
      .slice(-4)
      .map((message) => sanitizeReplyText(message.content))
      .filter(Boolean);
    const blockedReplies = new Set((state.knowledge.blockedReplies || []).map((value) => value.toLowerCase()));
    const minSimilarity = Math.max(Number(state.settings.training.minSimilarity) || 0, 0);
    const dialogueMemoryConfig = runtimeConfig.dialogueMemory || {};
    const strongMatchThreshold = Math.max(
      minSimilarity,
      Number(dialogueMemoryConfig.strongMatchThreshold) || 1.08
    );
    const softMatchThreshold = Math.max(
      minSimilarity,
      Number(dialogueMemoryConfig.softMatchThreshold) || 0.62
    );
    const preferGroundedKnowledge = isTechnicalOrMathQuery(userMessage);
    const topicalGate = Math.max(0.08, minSimilarity * 0.75, preferGroundedKnowledge ? 0.14 : 0);

    const replyCandidates = (state.knowledge.replyMemories || [])
      .map((memory) => {
        const promptSideRelevance = computePromptReplyRelevance(userMessage, memory.promptText);
        const responseSideRelevance = computePromptReplyRelevance(userMessage, memory.responseText);
        const topicalRelevance = responseSideRelevance * 0.78 + promptSideRelevance * 0.22;

        return {
          ...memory,
          promptSideRelevance,
          responseSideRelevance,
          topicalRelevance,
          sourcePenalty: memory.origin === 'source' ? 0.16 : 0,
          score:
            computeBm25Score(queryMap, memory, state.knowledge.bm25.reply, runtimeConfig) *
            runtimeConfig.retrieval.assistantReplyBoost *
            (1 + (memory.score || 0) * 0.15) +
            topicalRelevance * 0.72 +
            responseSideRelevance * 0.2 -
            (memory.origin === 'source' ? 0.16 : 0),
        };
      })
      .filter((memory) => !isNoisyKnowledgeArtifactText(memory.promptText))
      .filter((memory) => !isNoisyKnowledgeArtifactText(memory.responseText))
      .filter((memory) => memory.score >= softMatchThreshold)
      .filter((memory) => memory.topicalRelevance >= topicalGate)
      .filter((memory) => memory.responseSideRelevance >= (preferGroundedKnowledge ? 0.12 : 0.06))
      .filter((memory) => !blockedReplies.has(memory.responseText.toLowerCase()))
      .filter((memory) => !isEchoReply(userMessage, memory.responseText))
      .filter((memory) => computeRecentReplyPenalty(memory.responseText, recentAssistantReplies) < 0.42)
      .sort((left, right) => right.score - left.score)
      .slice(0, runtimeConfig.retrieval.topReplyPairs);

    const localChunkCandidates = (state.knowledge.chunks || [])
      .map((chunk) => {
        const topicalRelevance = computePromptReplyRelevance(userMessage, chunk.text);
        return {
          ...chunk,
          topicalRelevance,
          score:
            computeBm25Score(queryMap, chunk, state.knowledge.bm25.knowledge, runtimeConfig) *
            runtimeConfig.retrieval.knowledgeChunkBoost +
            topicalRelevance * 0.45,
        };
      })
      .filter((chunk) => !isNoisyKnowledgeArtifactText(chunk.text))
      .filter((chunk) => chunk.score > 0)
      .filter((chunk) => chunk.topicalRelevance >= topicalGate)
      .sort((left, right) => right.score - left.score)
      .slice(0, state.settings.training.topKChunks);

    const bestReply = replyCandidates.find((memory) => memory.score >= strongMatchThreshold) || null;
    const topLocalChunkScore = Number(localChunkCandidates[0]?.score || 0);
    const localTopicalRelevance = Math.max(
      Number(bestReply?.topicalRelevance || 0),
      Number(localChunkCandidates[0]?.topicalRelevance || 0)
    );
    const localEvidenceScore = Math.max(Number(bestReply?.score || 0), topLocalChunkScore);
    const weakLocalEvidence =
      localEvidenceScore < Math.max(0.24, minSimilarity * 1.35) ||
      localTopicalRelevance < Math.max(0.14, topicalGate);
    let webChunkCandidates = [];

    if (runtimeConfig.generation?.webSearchEnabled && !shouldSkipWebSearchForMessage(userMessage)) {
      const forceWebSearch = shouldForceWebSearchForMessage(userMessage);
      const shouldUseWebSearch =
        forceWebSearch || preferGroundedKnowledge || weakLocalEvidence || !bestReply || localChunkCandidates.length < 2;
      if (!shouldUseWebSearch) {
        webChunkCandidates = [];
      } else {
        try {
          webChunkCandidates = await fetchWebKnowledgeCandidates({
            userMessage,
            queryMap,
            runtimeConfig,
            preferGroundedKnowledge,
          });
        } catch (error) {
          log('warn', 'Web search failed, continuing with local evidence.', {
            scope: 'generation',
            reason: cleanText(error?.message || ''),
            promptPreview: previewText(userMessage, 72),
          });
          webChunkCandidates = [];
        }
      }
    }

    if (webChunkCandidates.length && preferGroundedKnowledge) {
      webChunkCandidates = webChunkCandidates.map((chunk) => ({
        ...chunk,
        score: chunk.score * 1.42,
      }));
    } else if (webChunkCandidates.length && weakLocalEvidence) {
      webChunkCandidates = webChunkCandidates.map((chunk) => ({
        ...chunk,
        score: chunk.score * 1.28,
      }));
    }

    const chunkCandidates = [...localChunkCandidates, ...webChunkCandidates]
      .sort((left, right) => right.score - left.score)
      .slice(0, Math.max(state.settings.training.topKChunks, webChunkCandidates.length));
    const replyReferences = buildReplyReferences(chunkCandidates);
    const usedWebSourceEntries = replyReferences.filter((reference) => reference.type === 'web');

    const knowledgeSentences = selectKnowledgeSentences(
      chunkCandidates,
      queryMap,
      state.settings.generation.maxReplySentences
    );
    const knowledgeText = knowledgeSentences.join(' ');

    let neuralReply = '';
    const allowNeuralGeneration =
      shouldUseNeuralGeneration({ bestReply, knowledgeSentences, state }) &&
      !(preferGroundedKnowledge && chunkCandidates.length > 0);

    if (allowNeuralGeneration) {
      try {
        await ensureRuntimeLoaded(state);
        if (neuralRuntime?.model) {
          const promptText = buildPromptForGeneration(
            userMessage,
            contextMessages,
            chunkCandidates,
            replyCandidates,
            userLanguage
          );
          const runtimeAlignedSettings = neuralRuntime.manifest?.modelSettings
            ? {
              ...state.settings,
              training: {
                ...state.settings.training,
                ...neuralRuntime.manifest.modelSettings,
              },
            }
            : state.settings;
          const generated = await generateText({
            runtime: neuralRuntime,
            promptText,
            settings: runtimeAlignedSettings,
          });
          neuralReply = sanitizeReplyText(generated.text);
        }
      } catch (_error) {
        neuralReply = '';
      }
    }

    let candidate = chooseBestReplyCandidate({
      userMessage,
      userLanguage,
      neuralReply,
      bestReply,
      knowledgeText,
      knowledgeSentences,
      maxSentences: state.settings.generation.maxReplySentences,
      chunkCandidates,
      webChunkCandidates,
      replyCandidates,
      contextMessages,
      recentAssistantReplies,
      preferGroundedKnowledge,
    });

    if (
      candidate &&
      preferGroundedKnowledge &&
      ['neural', 'reply_memory', 'hybrid'].includes(candidate.mode)
    ) {
      const groundedCandidateText = buildGroundedFallback(
        chunkCandidates,
        state.settings.generation.maxReplySentences
      );
      if (groundedCandidateText) {
        const groundedRelevance = computePromptReplyRelevance(userMessage, groundedCandidateText);
        const candidateRelevance = computePromptReplyRelevance(userMessage, candidate.content);
        if (groundedRelevance >= candidateRelevance + 0.06) {
          candidate = {
            mode: webChunkCandidates.length ? 'web_grounded' : 'knowledge',
            content: groundedCandidateText,
          };
        }
      }
    }

    if (!candidate) {
      const groundedFallback = buildGroundedFallback(
        chunkCandidates,
        state.settings.generation.maxReplySentences
      );

      if (groundedFallback) {
        const fallbackScore = computeSelfScore({
          userMessage,
          replyText: groundedFallback,
          chunkCandidates,
          replyCandidates,
          contextMessages,
          recentAssistantReplies,
        });

        return buildReplyPayload(groundedFallback, {
          maxSentences: state.settings.generation.maxReplySentences,
          maxCharacters: state.settings.generation.maxReplyCharacters,
          userMessage,
          references: replyReferences,
          metadata: {
            mode: 'knowledge',
            usedContextMessages: contextMessages.length,
            matchedSourceLabels: chunkCandidates.map((chunk) => chunk.label).slice(0, 4),
            matchedMemoryScore: bestReply?.score || 0,
            chunkMatches: chunkCandidates.length,
            selfScore: Number(fallbackScore.toFixed(3)),
            userRating: 0,
            usedWebSources: webChunkCandidates.map((chunk) => chunk.url).filter(Boolean).slice(0, 4),
            usedWebSourceEntries,
          },
        });
      }

      const fallbackMessage = runtimeConfig.generation?.webSearchEnabled
        ? 'Надежный ответ не найден ни в локальной базе, ни в веб-источниках. Уточните вопрос (например, версию языка, фреймворк или конкретную задачу).'
        : 'Пока в базе знаний не нашлось достаточно надежного ответа. Включите веб-поиск или добавьте профильные источники и переобучите модель.';

      return buildReplyPayload(fallbackMessage, {
        maxSentences: state.settings.generation.maxReplySentences,
        maxCharacters: state.settings.generation.maxReplyCharacters,
        userMessage,
        references: replyReferences,
        metadata: {
          mode: 'fallback',
          usedContextMessages: contextMessages.length,
          matchedSourceLabels: chunkCandidates.map((chunk) => chunk.label).slice(0, 4),
          matchedMemoryScore: bestReply?.score || 0,
          chunkMatches: chunkCandidates.length,
          selfScore: 0,
          userRating: 0,
          usedWebSources: webChunkCandidates.map((chunk) => chunk.url).filter(Boolean).slice(0, 4),
          usedWebSourceEntries,
        },
      });
    }

    const selfScore = computeSelfScore({
      userMessage,
      replyText: candidate.content,
      chunkCandidates,
      replyCandidates,
      contextMessages,
      recentAssistantReplies,
    });

    return buildReplyPayload(candidate.content, {
      maxSentences: state.settings.generation.maxReplySentences,
      maxCharacters: state.settings.generation.maxReplyCharacters,
      userMessage,
      references: replyReferences,
      metadata: {
        mode: candidate.mode,
        usedContextMessages: contextMessages.length,
        matchedSourceLabels: chunkCandidates.map((chunk) => chunk.label).slice(0, 4),
        matchedMemoryScore: bestReply?.score || 0,
        chunkMatches: chunkCandidates.length,
        selfScore: Number(selfScore.toFixed(3)),
        userRating: 0,
        usedWebSources: webChunkCandidates.map((chunk) => chunk.url).filter(Boolean).slice(0, 4),
        usedWebSourceEntries,
      },
    });
  }

  return {
    getOperationalStatus() {
      return {
        trainingPromiseActive: Boolean(activeTrainingPromise),
        trainingStartPending: Boolean(activeTrainingStartPromise),
        trainingWorkerActive: Boolean(activeTrainingWorker),
        queueRunnerActive: Boolean(activeTrainingQueueRunnerPromise),
        rollbackSnapshotReady: Boolean(activeTrainingRollbackSnapshot),
        runtimeLoaded: Boolean(neuralRuntime?.model),
      };
    },

    async hydrateState() {
      return updateState(async (state) => {
        ensureChatAvailability(state);
        ensureModelRegistryState(state);
        await pruneBrokenModelRegistryItems(state);
        prepareTrainingQueuesState(state);
        const artifacts = await rebuildKnowledge(state);
        const { restoredFromCheckpoint } = await reconcileCheckpointState(state, artifacts);
        await ensureRuntimeLoaded(state);
        const activeRegistryItem = updateActiveModelRegistrySummary(state);
        if (activeRegistryItem && !(await modelLibraryPackageExists(activeRegistryItem.id))) {
          await saveActiveModelPackage(state);
        }

        if (restoredFromCheckpoint) {
          pushStatus(
            state,
            'completed',
            'checkpoint_state_restored',
            'Сервер восстановил состояние обученной модели из сохраненного чекпоинта после перезапуска.',
            { updateTrainingState: false }
          );
        }

        if (state.training.status === 'training') {
          state.training.status = 'idle';
          state.training.phase = 'server_restarted';
          state.training.message = 'Сервер был перезапущен во время обучения. Запустите обучение снова, чтобы продолжить от последнего чекпоинта.';
          state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
          state.model.status = 'ready';
          pushStatus(state, 'idle', 'server_restarted', state.training.message);
        }

        const interruptedQueueId = state.trainingQueues.runner?.currentQueueId || null;
        state.trainingQueues.items = (state.trainingQueues.items || []).map((queue) => {
          if (queue.status !== 'running') {
            return queue;
          }

          return {
            ...queue,
            status: Array.isArray(queue.sources) && queue.sources.length ? 'pending' : 'empty',
            updatedAt: nowIso(),
          };
        });
        if (state.trainingQueues.runner?.active) {
          if (
            interruptedQueueId &&
            !state.trainingQueues.runner.pendingQueueIds.includes(interruptedQueueId)
          ) {
            state.trainingQueues.runner.pendingQueueIds = [
              interruptedQueueId,
              ...state.trainingQueues.runner.pendingQueueIds,
            ];
          }

          state.trainingQueues.runner.active = false;
          state.trainingQueues.runner.status = 'idle';
          state.trainingQueues.runner.currentQueueId = null;
          state.trainingQueues.runner.currentQueueIndex = 0;
          state.trainingQueues.runner.totalQueues = 0;
          state.trainingQueues.runner.pendingQueueIds = [];
          state.trainingQueues.runner.completedQueueIds = [];
          state.trainingQueues.runner.updatedAt = nowIso();
          state.trainingQueues.runner.lastError = '';

          pushStatus(
            state,
            'idle',
            'queue_runner_interrupted',
            'Сервер был перезапущен во время автоочереди обучения. Последний стабильный чекпоинт сохранен, оставшиеся очереди можно запустить повторно.',
            { updateTrainingState: false }
          );
        }

        if (state.model.trainedEpochs > 0) {
          const signatureMismatch =
            !state.model.corpusSignature || state.model.corpusSignature !== artifacts.corpusSignature;

          if (signatureMismatch) {
            state.model.exists = true;
            state.model.lifecycle = 'trained';
            state.model.status = 'ready';
            pushStatus(
              state,
              'idle',
              'model_rehydrated_retrain_recommended',
              'Сервер обнаружил изменения корпуса после последнего чекпоинта. Модель сохранена и доступна, но для учета новых данных рекомендуется дообучение.',
              { updateTrainingState: false }
            );
          } else if (!neuralRuntime?.model) {
            state.model.lifecycle = 'trained';
            state.model.status = 'ready';
            pushStatus(
              state,
              'completed',
              'checkpoint_restore_partial',
              'Сервер восстановил корпус и метаданные модели, но нейросетевой чекпоинт не загрузился. Ответы будут опираться на память чата и базу знаний, пока вы не переобучите или не удалите модель.',
              { updateTrainingState: false }
            );
          } else {
            state.model.parameterCount = neuralRuntime.parameterCount || state.model.parameterCount;
          }
        }
      });
    },

    async getDashboard(chatId) {
      await repairStaleExecutionStateIfNeeded();
      return buildDashboard(getState(), chatId);
    },

    async getRealtimeSnapshot() {
      await repairStaleExecutionStateIfNeeded();
      return buildRealtimeSnapshot(getState());
    },

    async createChat() {
      await repairStaleExecutionStateIfNeeded();
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Создание чата');
        ensureChatAvailability(state);
        state.chats.unshift(createEmptyChat());
      }, {
        writeSources: false,
        writeArtifacts: false,
      }).then((state) => buildDashboard(state, state.chats[0]?.id));
    },

    async deleteChat(chatId) {
      await repairStaleExecutionStateIfNeeded();
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Удаление чата');
        state.chats = state.chats.filter((chat) => chat.id !== chatId);
        ensureChatAvailability(state);
      }, {
        writeSources: false,
        writeArtifacts: false,
      }).then((state) => buildDashboard(state, state.chats[0]?.id));
    },

    async addSource({ type, label, content, url = null }) {
      await repairStaleExecutionStateIfNeeded();
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Изменение источников');
        appendSources(state, [{ type, label, content, url }]);

        pushStatus(
          state,
          'idle',
          'source_added',
          `Источник ${label} добавлен в очередь обучения. Текущая обученная модель сохранена, новые данные войдут после следующего запуска обучения.`
        );
      }, {
        writeChats: false,
        writeArtifacts: false,
      }).then((state) => buildDashboard(state));
    },

    async addSources(sources) {
      await repairStaleExecutionStateIfNeeded();
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Изменение источников');
        const preparedSources = appendSources(state, sources);
        pushStatus(
          state,
          'idle',
          'sources_added',
          `Добавлено ${preparedSources.length} источников в очередь обучения. Текущая обученная модель сохранена, новые данные войдут после следующего запуска обучения.`
        );
      }, {
        writeChats: false,
        writeArtifacts: false,
      }).then((state) => buildDashboard(state));
    },

    async removeSource(sourceId) {
      await repairStaleExecutionStateIfNeeded();
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Изменение источников');
        const removedSource = state.sources.find((source) => source.id === sourceId) || null;
        const removedDatasetPath = resolveDatasetSourcePath(removedSource || {});
        state.sources = state.sources.filter((source) => source.id !== sourceId);
        state.model.sourceCount = state.sources.length;
        if (state.model.exists) {
          state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
          state.model.status = 'ready';
        }
        if (removedDatasetPath) {
          try {
            await fs.rm(removedDatasetPath, { force: true });
          } catch (_error) {
            // Ignore cleanup failures for removed source dataset files.
          }
        }

        pushStatus(state, 'idle', 'source_removed', 'Источник удален из очереди обучения.');
      }, {
        writeChats: false,
        writeArtifacts: false,
      }).then((state) => buildDashboard(state));
    },

    async createTrainingQueue(name) {
      await repairStaleExecutionStateIfNeeded();
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Изменение очередей обучения');
        prepareTrainingQueuesState(state);
        const queueId = createId('training_queue');
        const queueIndex = state.trainingQueues.items.length + 1;
        const queueName = normalizeTrainingQueueName(name, queueIndex);
        const timestamp = nowIso();

        state.trainingQueues.items.push({
          id: queueId,
          name: queueName,
          status: 'pending',
          createdAt: timestamp,
          updatedAt: timestamp,
          completedAt: null,
          lastRunAt: null,
          lastError: '',
          sources: [],
        });

        pushStatus(
          state,
          'idle',
          'training_queue_created',
          `Создана очередь обучения «${queueName}».`,
          { updateTrainingState: false }
        );
      }, {
        writeSources: false,
        writeChats: false,
        writeArtifacts: false,
      }).then((state) => buildDashboard(state));
    },

    async addSourcesToTrainingQueue(queueId, sources) {
      const preparedSources = [];
      for (const source of sources) {
        const datasetPath = cleanText(source.datasetFilePath || '');
        if (datasetPath) {
          const sourceId = createId('queued_source');
          const persistedDataset = await persistTrainingQueueDatasetFile(queueId, sourceId, datasetPath);
          preparedSources.push({
            id: sourceId,
            type: 'dataset_file',
            label: cleanText(source.label || '') || path.basename(datasetPath),
            url: source.url || null,
            stats: source.stats && typeof source.stats === 'object'
              ? source.stats
              : computeStats(''),
            addedAt: nowIso(),
            contentPath: persistedDataset.contentPath,
            contentSize: Math.max(
              Number(source.contentSize) || 0,
              Number(persistedDataset.contentSize) || 0
            ),
          });
          continue;
        }

        const text = cleanText(source.content);
        if (!text) {
          continue;
        }

        const sourceId = createId('queued_source');
        const persistedSource = await writeTrainingQueueSourceContent(queueId, sourceId, text);
        preparedSources.push({
          id: sourceId,
          type: source.type,
          label: source.label,
          url: source.url || null,
          stats: computeStats(text),
          addedAt: nowIso(),
          contentPath: persistedSource.contentPath,
          contentSize: persistedSource.contentSize,
        });
      }

      if (!preparedSources.length) {
        throw new Error('Очередь не пополнена: файлы не содержат текста.');
      }

      try {
        const snapshot = await updateState(async (state) => {
          assertTrainingUnlocked(state, 'Изменение очередей обучения');
          prepareTrainingQueuesState(state);
          const queue = state.trainingQueues.items.find((entry) => entry.id === queueId);
          if (!queue) {
            throw new Error('Очередь обучения не найдена.');
          }

          queue.sources.unshift(...preparedSources);
          queue.status = 'pending';
          queue.updatedAt = nowIso();
          queue.lastError = '';

          pushStatus(
            state,
            'idle',
            'training_queue_sources_added',
            `В очередь «${queue.name}» добавлено ${preparedSources.length} файлов.`,
            { updateTrainingState: false }
          );
        }, {
          writeSources: false,
          writeChats: false,
          writeArtifacts: false,
        });

        return buildDashboard(snapshot);
      } catch (error) {
        await Promise.all(preparedSources.map(async (source) => {
          try {
            await removeTrainingQueueSourceContent(source);
          } catch (_cleanupError) {
            // Ignore cleanup failures after rejected queue write.
          }
        }));
        throw error;
      }
    },

    async removeTrainingQueueSource(queueId, sourceId) {
      let removedSource = null;
      const snapshot = await updateState(async (state) => {
        assertTrainingUnlocked(state, 'Изменение очередей обучения');
        prepareTrainingQueuesState(state);
        const queue = state.trainingQueues.items.find((entry) => entry.id === queueId);
        if (!queue) {
          pushStatus(
            state,
            'idle',
            'training_queue_source_missing',
            'Файл уже отсутствует в очереди обучения или очередь была удалена.',
            { updateTrainingState: false }
          );
          return;
        }

        const sourceExists = (queue.sources || []).some((source) => source.id === sourceId);
        removedSource = (queue.sources || []).find((source) => source.id === sourceId) || null;
        queue.sources = (queue.sources || []).filter((source) => source.id !== sourceId);
        queue.updatedAt = nowIso();
        queue.status = queue.sources.length ? 'pending' : 'empty';
        queue.lastError = '';
        prepareTrainingQueuesState(state);

        pushStatus(
          state,
          'idle',
          sourceExists ? 'training_queue_source_removed' : 'training_queue_source_missing',
          sourceExists
            ? `Из очереди «${queue.name}» удален файл.`
            : `Файл уже отсутствовал в очереди «${queue.name}».`,
          { updateTrainingState: false }
        );
      }, {
        writeSources: false,
        writeChats: false,
        writeArtifacts: false,
      });

      try {
        await removeTrainingQueueSourceContent(removedSource);
        await cleanupTrainingQueueStorage(getState().trainingQueues);
      } catch (error) {
        logError('Failed to cleanup removed training queue source file.', error, {
          scope: 'training_queue_storage',
          queueId,
          sourceId,
        });
      }
      return buildDashboard(snapshot);
    },

    async deleteTrainingQueue(queueId) {
      let removedSources = [];
      const snapshot = await updateState(async (state) => {
        assertTrainingUnlocked(state, 'Изменение очередей обучения');
        prepareTrainingQueuesState(state);
        const queue = state.trainingQueues.items.find((entry) => entry.id === queueId);
        if (!queue) {
          pushStatus(
            state,
            'idle',
            'training_queue_missing',
            'Очередь уже отсутствует и была просто убрана из локального состояния.',
            { updateTrainingState: false }
          );
          return;
        }

        removedSources = cloneTrainingQueueSources(queue.sources || []);
        state.trainingQueues.items = state.trainingQueues.items.filter((entry) => entry.id !== queueId);
        state.trainingQueues.runner.pendingQueueIds = state.trainingQueues.runner.pendingQueueIds
          .filter((entryId) => entryId !== queueId);
        state.trainingQueues.runner.completedQueueIds = state.trainingQueues.runner.completedQueueIds
          .filter((entryId) => entryId !== queueId);
        if (state.trainingQueues.runner.currentQueueId === queueId) {
          state.trainingQueues.runner.currentQueueId = null;
        }
        state.trainingQueues.runner.updatedAt = nowIso();
        prepareTrainingQueuesState(state);

        pushStatus(
          state,
          'idle',
          'training_queue_deleted',
          `Очередь «${queue.name}» удалена.`,
          { updateTrainingState: false }
        );
      }, {
        writeSources: false,
        writeChats: false,
        writeArtifacts: false,
      });

      try {
        await Promise.all((removedSources || []).map((source) => removeTrainingQueueSourceContent(source)));
        await removeTrainingQueueDirectory(queueId);
        await cleanupTrainingQueueStorage(getState().trainingQueues);
      } catch (error) {
        logError('Failed to cleanup deleted training queue directory.', error, {
          scope: 'training_queue_storage',
          queueId,
        });
      }
      return buildDashboard(snapshot);
    },

    async updateSettings(partialSettings) {
      await repairStaleExecutionStateIfNeeded();
      const currentSettings = structuredClone(getState().settings);
      const mergedTrainingSettings = {
        ...currentSettings.training,
        ...(partialSettings.training || {}),
      };
      const resolvedTrainingSettings = resolveTrainingSettings(mergedTrainingSettings);
      const nextSettings = {
        training: {
          ...mergedTrainingSettings,
          ...resolvedTrainingSettings,
          hiddenSize: resolvedTrainingSettings.feedForwardSize,
          recurrentLayers: resolvedTrainingSettings.transformerLayers,
        },
        generation: {
          ...currentSettings.generation,
          ...(partialSettings.generation || {}),
        },
      };
      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Изменение настроек');
        const previousSettings = structuredClone(state.settings);
        state.settings = nextSettings;

        if (hasRetrainingImpact(previousSettings, state.settings)) {
          state.model.lifecycle = state.model.trainedEpochs ? 'trained' : (state.model.exists ? 'ready_for_training' : 'not_created');
          state.model.status = state.model.exists ? 'ready' : 'idle';
          pushStatus(
            state,
            'idle',
            'settings_updated_retrain_recommended',
            'Изменены архитектурные или корпусные параметры. Текущий чекпоинт сохранен и доступен, но для применения новой конфигурации рекомендуется запустить дообучение.'
          );
        } else {
          pushStatus(state, 'idle', 'settings_updated', 'Настройки сервера обновлены.');
        }

        if (hasCorpusRebuildImpact(previousSettings, state.settings)) {
          await rebuildKnowledge(state);
        }
      }, {
        writeSources: false,
        writeChats: false,
        writeArtifacts: false,
      }).then((state) => buildDashboard(state));
    },

    async createFreshModel(name = '') {
      await repairStaleExecutionStateIfNeeded();
      await updateState(async (state) => {
        assertTrainingUnlocked(state, 'Создание модели');
        ensureModelRegistryState(state);
        await saveActiveModelPackage(state);
        await disposeNeuralRuntime();
        activeTrainingRollbackSnapshot = null;

        const nextModelItem = createFreshModelRegistryItem(state, name);
        nextModelItem.packagePath = getModelLibraryPackagePath(nextModelItem.id);
        state.modelRegistry.items.push(nextModelItem);
        state.modelRegistry.activeModelId = nextModelItem.id;

        state.model = createDefaultModelState();
        state.training = createDefaultTrainingState();
        state.knowledge = createDefaultKnowledgeState();
        state.model.exists = true;
        state.model.lifecycle = 'ready_for_training';
        state.model.status = 'ready';
        state.model.configSnapshot = structuredClone(state.settings);
        restoreQueuesAfterModelReset(state);
        updateActiveModelRegistrySummary(state);
        pushStatus(state, 'idle', 'fresh_model_created', 'Создана новая модель. Предыдущая сохранена в библиотеке моделей.');
        await rebuildKnowledge(state);
        await saveActiveModelPackage(state);
      }, {
        writeSources: false,
        writeChats: false,
      });
      return buildDashboard(getState());
    },

    async pauseTraining() {
      await repairStaleExecutionStateIfNeeded();
      if (!activeTrainingPromise && !activeTrainingQueueRunnerPromise) {
        throw new Error('Сейчас нет активного обучения, которое можно поставить на паузу.');
      }

      if (!activeTrainingPromise && activeTrainingQueueRunnerPromise) {
        await updateState(async (state) => {
          if (state.trainingQueues.runner?.active) {
            markQueueAsPendingAfterInterruption(
              state,
              state.trainingQueues.runner.currentQueueId,
              'paused'
            );
          }
          state.model.lifecycle = state.model.trainedEpochs ? 'paused' : 'ready_for_training';
          state.model.status = 'ready';
          state.training.status = 'paused';
          state.training.phase = 'paused_by_user';
          state.training.message = 'Автоочередь обучения остановлена пользователем между очередями.';
          pushStatus(state, 'paused', 'paused_by_user', state.training.message);
        }, {
          writeSources: false,
          writeChats: false,
          writeArtifacts: false,
        });
        return buildDashboard(getState());
      }

      await updateState(async (state) => {
        state.training.requestedStop = true;
        pushStatus(state, 'training', 'pause_requested', 'Пауза будет выполнена после завершения текущей стадии подготовки или текущего батча и сохранения чекпоинта.');
      }, { persist: false });
      await stopActiveTraining('pause_requested');
      await updateState(async (state) => {
        if (state.trainingQueues.runner?.active) {
          markQueueAsPendingAfterInterruption(
            state,
            state.trainingQueues.runner.currentQueueId,
            'paused'
          );
        }
        if (state.training.status === 'training' || state.model.lifecycle === 'training') {
          state.model.lifecycle = state.model.trainedEpochs ? 'paused' : 'ready_for_training';
          state.model.status = 'ready';
          state.training.status = 'paused';
          state.training.phase = 'paused_by_user';
          state.training.message = 'Подготовка или обучение остановлены пользователем до сохранения нового чекпоинта.';
          pushStatus(state, 'paused', 'paused_by_user', state.training.message);
        }
      }, {
        writeSources: false,
        writeChats: false,
      });
      return buildDashboard(getState());
    },

    async resetModel() {
      await repairStaleExecutionStateIfNeeded();
      await stopActiveTraining('model_deleted');

      await updateState(async (state) => {
        ensureModelRegistryState(state);
        await disposeNeuralRuntime();
        await removeModelArtifacts(state.model.storage);
        activeTrainingRollbackSnapshot = null;
        state.model = createDefaultModelState();
        state.training = createDefaultTrainingState();
        state.knowledge = createDefaultKnowledgeState();
        restoreQueuesAfterModelReset(state);
        pushStatus(state, 'idle', 'model_deleted', 'Обучение остановлено, артефакты модели удалены. Источники и чаты сохранены.');
        await rebuildKnowledge(state);
        updateActiveModelRegistrySummary(state);
        await saveActiveModelPackage(state);
      }, {
        writeSources: false,
        writeChats: false,
      });
      return buildDashboard(getState());
    },

    async rollbackTrainingToCheckpoint() {
      await repairStaleExecutionStateIfNeeded();
      if (!activeTrainingWorker && !activeTrainingPromise && !activeTrainingQueueRunnerPromise) {
        throw new Error('Откат доступен только во время активного обучения.');
      }

      if (!activeTrainingWorker && !activeTrainingPromise && activeTrainingQueueRunnerPromise) {
        await updateState(async (state) => {
          if (state.trainingQueues.runner?.active) {
            markQueueAsPendingAfterInterruption(
              state,
              state.trainingQueues.runner.currentQueueId,
              'rolled_back'
            );
          }
          state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
          state.model.status = 'ready';
          state.training.status = 'idle';
          state.training.phase = 'rollback_restored';
          state.training.message = 'Автоочередь остановлена между очередями. Модель оставлена на последней успешно примененной очереди.';
          pushStatus(state, 'idle', 'rollback_restored', state.training.message);
        }, {
          writeSources: false,
          writeChats: false,
          writeArtifacts: false,
        });
        return buildDashboard(getState());
      }

      await stopActiveTraining('rollback_to_checkpoint');

      await updateState(async (state) => {
        await disposeNeuralRuntime();
        const queueId = state.trainingQueues.runner?.currentQueueId || null;
        await restoreLastStableCheckpointState(state);
        if (state.trainingQueues.runner?.active) {
          markQueueAsPendingAfterInterruption(state, queueId, 'rolled_back');
        }
      }, {
        writeSources: false,
        writeChats: false,
      });

      activeTrainingRollbackSnapshot = null;
      return buildDashboard(getState());
    },

    async exportModelPackage() {
      const state = getState();
      if (!state.model.exists) {
        throw new Error('Модель еще не создана. Экспорт недоступен.');
      }

      if (isTrainingLocked(state) || activeTrainingPromise) {
        throw new Error('Экспорт модели недоступен во время обучения или сохранения чекпоинта.');
      }

      const payload = await buildModelPackagePayload(state, {
        strictCheckpoint: true,
      });

      const fileStamp = nowIso().replace(/[:.]/g, '-');
      return {
        fileName: `ai-model-${fileStamp}.aistudio.json`,
        contentType: 'application/json; charset=utf-8',
        buffer: Buffer.from(JSON.stringify(payload), 'utf8'),
      };
    },

    async importModelPackage(modelPackage) {
      const currentState = getState();
      if (isTrainingLocked(currentState) || activeTrainingPromise) {
        throw new Error('Импорт модели недоступен во время обучения или сохранения чекпоинта.');
      }

      await updateState(async (state) => {
        assertTrainingUnlocked(state, 'Импорт модели');
        ensureModelRegistryState(state);
        await saveActiveModelPackage(state);
        const nextRegistryItem = createModelRegistryItemFromState(
          state,
          modelPackage.snapshot?.model?.exportName || modelPackage.snapshot?.model?.engine || 'Импортированная модель'
        );
        nextRegistryItem.packagePath = getModelLibraryPackagePath(nextRegistryItem.id);
        state.modelRegistry.items.push(nextRegistryItem);
        state.modelRegistry.activeModelId = nextRegistryItem.id;
        await applyModelPackageToState(state, modelPackage, {
          strictArtifacts: true,
          phase: 'model_imported',
        });
        updateActiveModelRegistrySummary(state);
        await saveActiveModelPackage(state);
      }, {
        writeSources: false,
        writeChats: false,
      });

      await updateState(async (state) => {
        try {
          await ensureRuntimeLoaded(state);
          if (neuralRuntime?.model) {
            state.model.parameterCount = neuralRuntime.parameterCount || state.model.parameterCount;
          }
        } catch (_error) {
          state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
          state.model.status = 'ready';
          pushStatus(
            state,
            'completed',
            'checkpoint_restore_partial',
            'Пакет импортирован, но нейросетевой чекпоинт не удалось загрузить в память. Проверьте совместимость артефактов.',
            { updateTrainingState: false }
          );
        }
      }, {
        writeSources: false,
        writeChats: false,
        writeArtifacts: false,
      });

      return buildDashboard(getState());
    },

    async updateRuntimeSettings(partialRuntimeConfig) {
      await repairStaleExecutionStateIfNeeded();
      const currentState = getState();
      if (isTrainingLocked(currentState) || activeTrainingPromise) {
        throw new Error('Изменение runtime-настроек недоступно во время обучения.');
      }

      await updateState(async (state) => {
        assertTrainingUnlocked(state, 'Изменение runtime-настроек');
        ensureModelRegistryState(state);
        await updateRuntimeConfig(partialRuntimeConfig || {});
        updateActiveModelRegistrySummary(state);
        pushStatus(
          state,
          'idle',
          'runtime_settings_updated',
          'Обновлены настройки генератора и веб-поиска.',
          { updateTrainingState: false }
        );
      }, {
        writeSources: false,
        writeChats: false,
        writeArtifacts: false,
      });

      return buildDashboard(getState());
    },

    async selectModel(modelId) {
      await repairStaleExecutionStateIfNeeded();
      const currentState = getState();
      if (isTrainingLocked(currentState) || activeTrainingPromise) {
        throw new Error('Переключение моделей недоступно во время обучения.');
      }

      if (!cleanText(modelId)) {
        throw new Error('Идентификатор модели не передан.');
      }

      const existingModel = (currentState.modelRegistry?.items || []).find((item) => item.id === modelId);
      if (!existingModel) {
        throw new Error('Модель из библиотеки не найдена.');
      }

      if (currentState.modelRegistry?.activeModelId === modelId) {
        return buildDashboard(currentState);
      }

      let modelPackage = null;
      try {
        modelPackage = await readModelLibraryPackage(modelId);
      } catch (error) {
        if (error?.code === 'ENOENT') {
          await updateState(async (state) => {
            ensureModelRegistryState(state);
            state.modelRegistry.items = (state.modelRegistry.items || []).filter((item) => item.id !== modelId);
            if (state.modelRegistry.activeModelId === modelId) {
              state.modelRegistry.activeModelId = state.modelRegistry.items[0]?.id || null;
            }
            pushStatus(
              state,
              'error',
              'model_package_missing',
              'Выбранная модель повреждена или ее пакет отсутствует. Запись удалена из библиотеки.'
            );
          }, {
            writeSources: false,
            writeChats: false,
            writeArtifacts: false,
          });
          throw new Error('Файл выбранной модели отсутствует на диске. Запись удалена из библиотеки, выберите другую модель.');
        }
        throw error;
      }

      await updateState(async (state) => {
        assertTrainingUnlocked(state, 'Переключение модели');
        ensureModelRegistryState(state);
        await saveActiveModelPackage(state);

        state.modelRegistry.activeModelId = modelId;
        await applyModelPackageToState(state, modelPackage, {
          strictArtifacts: false,
          phase: 'model_selected',
        });

        const selectedItem = state.modelRegistry.items.find((item) => item.id === modelId) || null;
        if (selectedItem) {
          selectedItem.lastUsedAt = nowIso();
        }

        pushStatus(
          state,
          'idle',
          'model_selected',
          `Активна модель «${selectedItem?.name || 'без названия'}».`,
          { updateTrainingState: false }
        );
      }, {
        writeSources: false,
        writeChats: false,
      });

      await updateState(async (state) => {
        try {
          await ensureRuntimeLoaded(state);
          if (neuralRuntime?.model) {
            state.model.parameterCount = neuralRuntime.parameterCount || state.model.parameterCount;
          }
        } catch (_error) {
          // Keep the selected model active even if runtime stays unloaded.
        }
      }, {
        writeSources: false,
        writeChats: false,
        writeArtifacts: false,
      });

      return buildDashboard(getState());
    },

    async deleteModelFromLibrary(modelId) {
      await repairStaleExecutionStateIfNeeded();
      const currentState = getState();
      if (isTrainingLocked(currentState) || activeTrainingPromise) {
        throw new Error('Удаление модели из библиотеки недоступно во время обучения.');
      }

      if (currentState.modelRegistry?.activeModelId === modelId) {
        throw new Error('Сначала переключитесь на другую модель, затем удаляйте текущую.');
      }

      await updateState(async (state) => {
        ensureModelRegistryState(state);
        const nextItems = (state.modelRegistry.items || []).filter((item) => item.id !== modelId);
        if (nextItems.length === (state.modelRegistry.items || []).length) {
          throw new Error('Модель из библиотеки не найдена.');
        }
        state.modelRegistry.items = nextItems;
        pushStatus(
          state,
          'idle',
          'model_deleted_from_library',
          'Модель удалена из библиотеки.',
          { updateTrainingState: false }
        );
      }, {
        writeSources: false,
        writeChats: false,
        writeArtifacts: false,
      });

      await removeModelLibraryPackage(modelId);
      return buildDashboard(getState());
    },

    async trainModel() {
      await repairStaleExecutionStateIfNeeded();
      await ensureModelExists();
      const state = getState();
      if (hasPendingTrainingQueues(state)) {
        runTrainingQueueCampaign().catch((error) => {
          logError('Training queue runner failed.', error);
        });
      } else {
        const trainingPromise = runTrainingJob();
        trainingPromise.catch(async (error) => {
          logError('Training promise failed.', error);
          await updateState(async (nextState) => {
            activeTrainingRollbackSnapshot = null;
            nextState.model.lifecycle = 'error';
            nextState.model.status = 'error';
            nextState.training.status = 'error';
            nextState.training.phase = 'training_failed';
            pushStatus(nextState, 'error', 'training_failed', error.message);
          }, {
            writeSources: false,
            writeChats: false,
            writeArtifacts: false,
          });
        });
      }
      if (activeTrainingStartPromise) {
        await Promise.race([
          activeTrainingStartPromise.catch(() => null),
          delay(600),
        ]);
      }
      return buildDashboard(getState());
    },

    async sendMessage(chatId, content) {
      const normalizedContent = cleanText(content);
      if (!normalizedContent) {
        throw new Error('Сообщение пустое.');
      }

      const snapshot = await updateState(async (state) => {
        assertTrainingUnlocked(state, 'Отправка сообщений');
        ensureChatAvailability(state);
        const chat = state.chats.find((entry) => entry.id === chatId) || state.chats[0];

        const userMessage = {
          id: createId('msg'),
          role: 'user',
          content: normalizedContent,
          createdAt: nowIso(),
        };

        chat.messages.push(userMessage);

        if (!state.model.exists || !state.model.trainedEpochs) {
          chat.messages.push({
            id: createId('msg'),
            role: 'assistant',
            content: 'Модель пока не обучена. Сначала добавьте данные и запустите обучение.',
            createdAt: nowIso(),
            metadata: {
              mode: 'system',
              selfScore: 0,
              userRating: 0,
            },
          });
        } else {
          state.model.lifecycle = 'generating_reply';
          state.model.status = 'busy';
          pushStatus(
            state,
            'syncing_knowledge',
            'generating_reply',
            'Сервер формирует ответ по обученной нейросети и индексу знаний.',
            { updateTrainingState: false }
          );

          const reply = await renderReply({
            state,
            chat,
            userMessage: normalizedContent,
          });

          chat.messages.push({
            id: createId('msg'),
            role: 'assistant',
            content: reply.content,
            createdAt: nowIso(),
            metadata: reply.metadata,
          });

          const assistantMessages = chat.messages
            .filter((message) => message.role === 'assistant' && message.metadata?.selfScore !== undefined)
            .map((message) => Number(message.metadata.selfScore || 0));
          state.model.averageSelfScore = assistantMessages.length
            ? Number((assistantMessages.reduce((sum, value) => sum + value, 0) / assistantMessages.length).toFixed(3))
            : state.model.averageSelfScore;
          state.model.lifecycle = 'trained';
          state.model.status = 'ready';
          state.model.lastGenerationAt = nowIso();
        }

        chat.title = inferChatTitle(chat.messages);
        chat.updatedAt = nowIso();
      }, {
        writeSources: false,
        writeArtifacts: false,
      });

      return buildDashboard(snapshot, chatId);
    },

    async rateMessage(messageId, score) {
      const normalizedScore = score > 0 ? 1 : -1;

      return updateState(async (state) => {
        assertTrainingUnlocked(state, 'Оценка ответов');
        const located = findMessageById(state, messageId);
        if (!located) {
          throw new Error('Сообщение для оценки не найдено.');
        }

        if (located.message.role !== 'assistant' || located.message.metadata?.type === 'system') {
          throw new Error('Оценивать можно только ответы модели.');
        }

        located.message.metadata = {
          ...(located.message.metadata || {}),
          userRating: normalizedScore,
          ratedAt: nowIso(),
        };

        if (normalizedScore === 1) {
          state.model.positiveFeedbackCount += 1;
          pushStatus(
            state,
            'learning_from_feedback',
            'feedback_recorded',
            'Положительная оценка сохранена. Этот ответ войдет в память модели и в следующий обучающий прогон.',
            { updateTrainingState: false }
          );
        } else {
          state.model.negativeFeedbackCount += 1;
          pushStatus(
            state,
            'learning_from_feedback',
            'feedback_recorded',
            'Отрицательная оценка сохранена. Этот ответ будет подавляться в будущих ответах и исключен из обучающих примеров.',
            { updateTrainingState: false }
          );
        }

        await rebuildKnowledge(state);
        state.model.lifecycle = state.model.trainedEpochs ? 'trained' : 'ready_for_training';
        state.model.status = 'ready';
      }, {
        writeSources: false,
      }).then((state) => buildDashboard(state));
    },
  };
}

module.exports = {
  createModelEngine,
};
