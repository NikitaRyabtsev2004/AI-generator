const fs = require('fs/promises');
const path = require('path');
const Database = require('better-sqlite3');
const { log, logError } = require('../lib/logger');
const {
  createDefaultKnowledgeState,
  createDefaultModelState,
  createDefaultModelRegistryState,
  createDefaultState,
  createDefaultTrainingState,
  createDefaultTrainingQueueRunnerState,
  createDefaultTrainingQueuesState,
} = require('../lib/config');
const {
  ensureModelLibraryLayout,
  getModelLibraryDir,
  setActiveModelLibraryUser,
} = require('./modelLibraryStorage');
const { RUNTIME_CONFIG_PATH } = require('./runtimeConfig');
const { cleanText, computeStats, inferChatTitle } = require('../lib/text');
const {
  TRAINING_JOB_STORAGE_DIR,
  TRAINING_QUEUE_STORAGE_DIR,
  cleanupTrainingQueueStorage,
  ensureTrainingQueueStorageLayout,
  externalizeTrainingQueueSources,
} = require('./trainingQueueStorage');

const DATA_DIR = path.join(__dirname, '..', 'data');
const ARTIFACT_DIR = path.join(__dirname, '..', 'artifacts');
const USER_WORKSPACES_DIR = path.join(DATA_DIR, 'user-workspaces');
const DB_FILE = path.join(DATA_DIR, 'studio.db');
const LEGACY_STORE_FILE = path.join(DATA_DIR, 'studio-store.json');
const KNOWLEDGE_INDEX_FILE = path.join(ARTIFACT_DIR, 'knowledge-index.json');
const LANGUAGE_MODEL_FILE = path.join(ARTIFACT_DIR, 'language-model.json');
const MODEL_MANIFEST_FILE = path.join(ARTIFACT_DIR, 'model-manifest.json');
const NEURAL_MODEL_DIR = path.join(ARTIFACT_DIR, 'neural-model');
const TOKENIZER_FILE = path.join(NEURAL_MODEL_DIR, 'tokenizer.json');
const NEURAL_WEIGHTS_FILE = path.join(NEURAL_MODEL_DIR, 'weights.bin');
const NEURAL_SPEC_FILE = path.join(NEURAL_MODEL_DIR, 'weights-spec.json');
const SLOW_PERSIST_THRESHOLD_MS = 500;

let db = null;
let state = null;
let updateQueue = Promise.resolve();
let lastTrainingQueueMetaSignature = '';
let lastTrainingQueueSourcesSignature = '';
let lastTrainingQueueRunnerSignature = '';
let activeWorkspaceUserId = 'legacy-local';

function normalizeWorkspaceUserId(userId) {
  const raw = cleanText(String(userId || '')).toLowerCase();
  if (!raw) {
    return 'legacy-local';
  }
  const sanitized = raw.replace(/[^a-z0-9_-]/gu, '-').replace(/-+/gu, '-').slice(0, 80);
  return sanitized || 'legacy-local';
}

function getWorkspaceArtifactsRoot(userId = activeWorkspaceUserId) {
  const normalizedUserId = normalizeWorkspaceUserId(userId);
  return path.join(ARTIFACT_DIR, 'users', normalizedUserId);
}

function safeStateSignature(value) {
  try {
    return JSON.stringify(value);
  } catch (_error) {
    return JSON.stringify({
      meta: value?.meta?.updatedAt || null,
      sourceCount: Array.isArray(value?.sources) ? value.sources.length : 0,
      chatCount: Array.isArray(value?.chats) ? value.chats.length : 0,
      queueCount: Array.isArray(value?.trainingQueues?.items) ? value.trainingQueues.items.length : 0,
      modelLifecycle: value?.model?.lifecycle || '',
      trainedEpochs: Number(value?.model?.trainedEpochs || 0),
    });
  }
}

function getArtifactPaths(userId = activeWorkspaceUserId) {
  const workspaceArtifactsRoot = getWorkspaceArtifactsRoot(userId);
  const neuralModelDir = path.join(workspaceArtifactsRoot, 'neural-model');
  return {
    databasePath: DB_FILE,
    artifactDir: workspaceArtifactsRoot,
    manifestPath: path.join(workspaceArtifactsRoot, 'model-manifest.json'),
    knowledgeIndexPath: path.join(workspaceArtifactsRoot, 'knowledge-index.json'),
    languageModelPath: path.join(workspaceArtifactsRoot, 'language-model.json'),
    neuralModelDir,
    tokenizerPath: path.join(neuralModelDir, 'tokenizer.json'),
    neuralWeightsPath: path.join(neuralModelDir, 'weights.bin'),
    neuralSpecPath: path.join(neuralModelDir, 'weights-spec.json'),
    runtimeConfigPath: RUNTIME_CONFIG_PATH,
    trainingQueueStorageDir: TRAINING_QUEUE_STORAGE_DIR,
    trainingJobStorageDir: TRAINING_JOB_STORAGE_DIR,
    modelLibraryDir: getModelLibraryDir(userId),
  };
}

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

function attachStorageInfo(nextState) {
  nextState.model.storage = {
    ...nextState.model.storage,
    ...getArtifactPaths(),
  };
  nextState.model.artifactFiles = [
    getArtifactPaths().manifestPath,
    getArtifactPaths().knowledgeIndexPath,
    getArtifactPaths().languageModelPath,
    getArtifactPaths().tokenizerPath,
    getArtifactPaths().neuralSpecPath,
    getArtifactPaths().neuralWeightsPath,
    getArtifactPaths().databasePath,
    getArtifactPaths().runtimeConfigPath,
  ];
  return nextState;
}

function pickKnownKeys(input, defaults) {
  if (!input || typeof input !== 'object') {
    return {};
  }

  return Object.keys(defaults).reduce((result, key) => {
    if (input[key] !== undefined) {
      result[key] = input[key];
    }
    return result;
  }, {});
}

function normalizeSource(source = {}) {
  const content = cleanText(source.content || '');
  const contentPath = cleanText(source.contentPath || '');
  const fallbackStats = computeStats(content);
  const sourceStats = source.stats && typeof source.stats === 'object'
    ? source.stats
    : fallbackStats;
  const tokenCount = Math.max(
    Number(sourceStats.tokenCount) || 0,
    Number(fallbackStats.tokenCount) || 0,
    0
  );
  const charCount = Math.max(
    Number(sourceStats.charCount) || 0,
    Number(fallbackStats.charCount) || 0,
    0
  );
  const rowCount = Math.max(Number(sourceStats.rowCount) || 0, 0);
  const columnCount = Math.max(Number(sourceStats.columnCount) || 0, 0);
  const format = cleanText(sourceStats.format || '');
  const columns = Array.isArray(sourceStats.columns)
    ? sourceStats.columns.map((value) => cleanText(String(value))).filter(Boolean).slice(0, 64)
    : [];
  const stats = {
    tokenCount,
    charCount,
    ...(rowCount > 0 ? { rowCount } : {}),
    ...(columnCount > 0 ? { columnCount } : {}),
    ...(format ? { format } : {}),
    ...(columns.length ? { columns } : {}),
  };

  return {
    ...source,
    label: cleanText(source.label || '') || source.label || 'Источник',
    url: source.url || null,
    content,
    stats,
    contentPath: contentPath || null,
    contentSize: Math.max(
      Number(source.contentSize) || 0,
      content ? Buffer.byteLength(content, 'utf8') : 0
    ),
  };
}

function normalizeTrainingQueueSource(source = {}) {
  const contentPath = cleanText(source.contentPath || '');
  const inlineContent = contentPath ? '' : cleanText(source.content || '');

  return {
    ...source,
    label: cleanText(source.label || '') || source.label || 'Источник',
    url: source.url || null,
    stats: source.stats && typeof source.stats === 'object'
      ? source.stats
      : computeStats(inlineContent),
    addedAt: source.addedAt || new Date().toISOString(),
    contentPath: contentPath || null,
    contentSize: Math.max(
      Number(source.contentSize) || 0,
      inlineContent ? Buffer.byteLength(inlineContent, 'utf8') : 0
    ),
    ...(inlineContent ? { content: inlineContent } : {}),
  };
}

function normalizeMessage(message = {}) {
  return {
    ...message,
    content: cleanText(message.content || ''),
  };
}

function normalizeChat(chat = {}) {
  const messages = Array.isArray(chat.messages) ? chat.messages.map(normalizeMessage) : [];

  return {
    ...chat,
    title: cleanText(chat.title || '') || inferChatTitle(messages),
    messages,
  };
}

function normalizeRecentStatus(entry = {}) {
  return {
    ...entry,
    message: cleanText(entry.message || ''),
  };
}

function normalizeTraining(training) {
  const defaults = createDefaultTrainingState();
  const nextTraining = deepMerge(defaults, training);
  nextTraining.message = cleanText(nextTraining.message || defaults.message);
  nextTraining.recentStatuses = Array.isArray(nextTraining.recentStatuses)
    ? nextTraining.recentStatuses.map(normalizeRecentStatus)
    : [];
  return nextTraining;
}

function normalizeTrainingQueueRunner(runner) {
  const defaults = createDefaultTrainingQueueRunnerState();
  const nextRunner = deepMerge(defaults, runner);
  nextRunner.lastError = cleanText(nextRunner.lastError || '');
  nextRunner.pendingQueueIds = Array.isArray(nextRunner.pendingQueueIds)
    ? nextRunner.pendingQueueIds.filter(Boolean)
    : [];
  nextRunner.completedQueueIds = Array.isArray(nextRunner.completedQueueIds)
    ? nextRunner.completedQueueIds.filter(Boolean)
    : [];
  return nextRunner;
}

function normalizeTrainingQueueItem(queue = {}) {
  return {
    ...queue,
    name: cleanText(queue.name || '') || 'Очередь обучения',
    status: cleanText(queue.status || '') || 'pending',
    lastError: cleanText(queue.lastError || ''),
    sources: Array.isArray(queue.sources) ? queue.sources.map(normalizeTrainingQueueSource) : [],
  };
}

function normalizeTrainingQueues(trainingQueues) {
  const defaults = createDefaultTrainingQueuesState();
  const nextTrainingQueues = deepMerge(defaults, trainingQueues);
  nextTrainingQueues.items = Array.isArray(nextTrainingQueues.items)
    ? nextTrainingQueues.items.map(normalizeTrainingQueueItem)
    : [];
  nextTrainingQueues.runner = normalizeTrainingQueueRunner(nextTrainingQueues.runner);
  return nextTrainingQueues;
}

function normalizeModelRegistry(modelRegistry) {
  const defaults = createDefaultModelRegistryState();
  const nextModelRegistry = deepMerge(defaults, modelRegistry);
  nextModelRegistry.activeModelId = cleanText(nextModelRegistry.activeModelId || '') || null;
  nextModelRegistry.items = Array.isArray(nextModelRegistry.items)
    ? nextModelRegistry.items
      .map((item) => ({
        ...item,
        id: cleanText(item?.id || ''),
        name: cleanText(item?.name || '') || 'Модель',
        kind: cleanText(item?.kind || '') || 'local',
        api: {
          provider: cleanText(item?.api?.provider || ''),
          endpoint: cleanText(item?.api?.endpoint || ''),
          model: cleanText(item?.api?.model || ''),
          hasKey: Boolean(item?.api?.hasKey),
        },
        packagePath: cleanText(item?.packagePath || ''),
        hasCheckpoint: Boolean(item?.hasCheckpoint),
        summary: {
          lifecycle: cleanText(item?.summary?.lifecycle || '') || 'not_created',
          trainedEpochs: Number(item?.summary?.trainedEpochs) || 0,
          parameterCount: Number(item?.summary?.parameterCount) || 0,
          vocabularySize: Number(item?.summary?.vocabularySize) || 0,
          tokenCount: Number(item?.summary?.tokenCount) || 0,
          sourceCount: Number(item?.summary?.sourceCount) || 0,
          replyPairCount: Number(item?.summary?.replyPairCount) || 0,
          backend: cleanText(item?.summary?.backend || '') || 'neural',
        },
      }))
      .filter((item) => item.id)
    : [];
  return nextModelRegistry;
}

function safeParseJson(rawValue, fallbackValue = undefined, contextLabel = 'snapshot row') {
  if (rawValue === null || rawValue === undefined) {
    return fallbackValue;
  }

  const normalizedRawValue = typeof rawValue === 'string' ? rawValue.trim() : '';
  if (!normalizedRawValue) {
    log('warn', 'Snapshot row contained empty JSON payload and was ignored.', {
      scope: 'store',
      contextLabel,
    });
    return fallbackValue;
  }

  try {
    return JSON.parse(normalizedRawValue);
  } catch (error) {
    logError('Failed to parse persisted JSON snapshot row. Falling back to defaults.', error, {
      scope: 'store',
      contextLabel,
    });
    return fallbackValue;
  }
}

function computeTrainingQueueMetaSignature(trainingQueues = {}) {
  const items = Array.isArray(trainingQueues.items) ? trainingQueues.items : [];
  return JSON.stringify(items.map((queue, index) => ({
    id: queue.id,
    name: queue.name,
    status: queue.status,
    createdAt: queue.createdAt,
    updatedAt: queue.updatedAt,
    completedAt: queue.completedAt || null,
    lastRunAt: queue.lastRunAt || null,
    lastError: queue.lastError || '',
    sourceCount: Array.isArray(queue.sources) ? queue.sources.length : 0,
    sortOrder: index,
  })));
}

function computeTrainingQueueSourcesSignature(trainingQueues = {}) {
  const items = Array.isArray(trainingQueues.items) ? trainingQueues.items : [];
  return JSON.stringify(items.map((queue, queueIndex) => ({
    queueId: queue.id,
    queueIndex,
    sources: Array.isArray(queue.sources)
      ? queue.sources.map((source, sourceIndex) => ({
        id: source.id,
        label: source.label,
        type: source.type,
        url: source.url || null,
        addedAt: source.addedAt,
        contentPath: source.contentPath || null,
        contentSize: Math.max(Number(source.contentSize) || 0, 0),
        sourceIndex,
      }))
      : [],
  })));
}

function computeTrainingQueueRunnerSignature(trainingQueues = {}) {
  return JSON.stringify(trainingQueues.runner || createDefaultTrainingQueuesState().runner);
}

function resetTrainingQueuePersistSignatures(trainingQueues = {}) {
  lastTrainingQueueMetaSignature = computeTrainingQueueMetaSignature(trainingQueues);
  lastTrainingQueueSourcesSignature = computeTrainingQueueSourcesSignature(trainingQueues);
  lastTrainingQueueRunnerSignature = computeTrainingQueueRunnerSignature(trainingQueues);
}

function normalizePersistedState(nextState) {
  return {
    ...nextState,
    sources: Array.isArray(nextState.sources) ? nextState.sources.map(normalizeSource) : [],
    modelRegistry: normalizeModelRegistry(nextState.modelRegistry),
    trainingQueues: normalizeTrainingQueues(nextState.trainingQueues),
    chats: Array.isArray(nextState.chats) ? nextState.chats.map(normalizeChat) : [],
    training: normalizeTraining(nextState.training),
  };
}

function hydrateState(rawState) {
  const defaults = createDefaultState();
  const nextState = normalizePersistedState({
    meta: deepMerge(defaults.meta, rawState?.meta),
    settings: deepMerge(defaults.settings, rawState?.settings),
    sources: Array.isArray(rawState?.sources) ? rawState.sources : [],
    modelRegistry: deepMerge(createDefaultModelRegistryState(), rawState?.modelRegistry),
    trainingQueues: deepMerge(createDefaultTrainingQueuesState(), rawState?.trainingQueues),
    chats: Array.isArray(rawState?.chats) ? rawState.chats : [],
    model: deepMerge(createDefaultModelState(), rawState?.model),
    training: deepMerge(createDefaultTrainingState(), rawState?.training),
    knowledge: deepMerge(createDefaultKnowledgeState(), rawState?.knowledge),
  });

  return attachStorageInfo(nextState);
}

function migrateLegacyState(rawLegacy) {
  const defaults = createDefaultState();

  return hydrateState({
    meta: {
      ...defaults.meta,
      createdAt: rawLegacy?.meta?.createdAt || defaults.meta.createdAt,
      updatedAt: rawLegacy?.meta?.updatedAt || defaults.meta.updatedAt,
      version: defaults.meta.version,
    },
    settings: {
      training: {
        ...defaults.settings.training,
        ...pickKnownKeys(rawLegacy?.settings?.training, defaults.settings.training),
      },
      generation: {
        ...defaults.settings.generation,
        ...pickKnownKeys(rawLegacy?.settings?.generation, defaults.settings.generation),
      },
    },
    sources: Array.isArray(rawLegacy?.sources) ? rawLegacy.sources : [],
    modelRegistry: createDefaultModelRegistryState(),
    trainingQueues: createDefaultTrainingQueuesState(),
    chats: Array.isArray(rawLegacy?.chats) ? rawLegacy.chats : [],
    model: createDefaultModelState(),
    training: createDefaultTrainingState(),
    knowledge: createDefaultKnowledgeState(),
  });
}

async function ensureStorageLayout() {
  const artifactPaths = getArtifactPaths();
  setActiveModelLibraryUser(activeWorkspaceUserId);
  await fs.mkdir(DATA_DIR, { recursive: true });
  await fs.mkdir(ARTIFACT_DIR, { recursive: true });
  await fs.mkdir(USER_WORKSPACES_DIR, { recursive: true });
  await fs.mkdir(artifactPaths.artifactDir, { recursive: true });
  await fs.mkdir(artifactPaths.neuralModelDir, { recursive: true });
  await ensureTrainingQueueStorageLayout();
  await ensureModelLibraryLayout();

  if (!db) {
    db = new Database(DB_FILE);
    db.pragma('journal_mode = PERSIST');
    db.pragma('temp_store = MEMORY');
    db.pragma('synchronous = NORMAL');
    db.pragma('busy_timeout = 5000');
    db.pragma('foreign_keys = ON');
    db.exec(`
      CREATE TABLE IF NOT EXISTS app_snapshot (
        name TEXT PRIMARY KEY,
        json_value TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS user_workspaces (
        user_id TEXT PRIMARY KEY,
        state_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS sources (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        label TEXT NOT NULL,
        url TEXT,
        content TEXT NOT NULL,
        content_path TEXT,
        content_size INTEGER DEFAULT 0,
        stats_json TEXT NOT NULL,
        added_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        metadata_json TEXT,
        sort_order INTEGER NOT NULL,
        FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS training_queues (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        last_run_at TEXT,
        last_error TEXT,
        sort_order INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS training_queue_sources (
        id TEXT PRIMARY KEY,
        queue_id TEXT NOT NULL,
        type TEXT NOT NULL,
        label TEXT NOT NULL,
        url TEXT,
        content TEXT NOT NULL,
        stats_json TEXT NOT NULL,
        added_at TEXT NOT NULL,
        sort_order INTEGER NOT NULL,
        FOREIGN KEY (queue_id) REFERENCES training_queues(id) ON DELETE CASCADE
      );
    `);

    const queueSourceColumns = db.prepare('PRAGMA table_info(training_queue_sources)').all();
    const knownQueueSourceColumns = new Set(queueSourceColumns.map((column) => column.name));
    if (!knownQueueSourceColumns.has('content_path')) {
      db.exec('ALTER TABLE training_queue_sources ADD COLUMN content_path TEXT;');
    }
    if (!knownQueueSourceColumns.has('content_size')) {
      db.exec('ALTER TABLE training_queue_sources ADD COLUMN content_size INTEGER DEFAULT 0;');
    }

    const sourceColumns = db.prepare('PRAGMA table_info(sources)').all();
    const knownSourceColumns = new Set(sourceColumns.map((column) => column.name));
    if (!knownSourceColumns.has('content_path')) {
      db.exec('ALTER TABLE sources ADD COLUMN content_path TEXT;');
    }
    if (!knownSourceColumns.has('content_size')) {
      db.exec('ALTER TABLE sources ADD COLUMN content_size INTEGER DEFAULT 0;');
    }
  }
}

function getSnapshotRow(name) {
  const row = db.prepare('SELECT json_value FROM app_snapshot WHERE name = ?').get(name);
  return row ? safeParseJson(row.json_value, undefined, `app_snapshot:${name}`) : undefined;
}

function setSnapshotRow(name, value) {
  db.prepare(`
    INSERT INTO app_snapshot(name, json_value)
    VALUES (?, ?)
    ON CONFLICT(name) DO UPDATE SET json_value = excluded.json_value
  `).run(name, JSON.stringify(value));
}

function readTrainingQueuesFromDatabase() {
  const queueRows = db.prepare(`
    SELECT id, name, status, created_at, updated_at, completed_at, last_run_at, last_error, sort_order
    FROM training_queues
    ORDER BY sort_order ASC, updated_at ASC
  `).all();

  const sourceRows = db.prepare(`
    SELECT id, queue_id, type, label, url, content, content_path, content_size, stats_json, added_at, sort_order
    FROM training_queue_sources
    ORDER BY queue_id ASC, sort_order ASC, added_at ASC
  `).all();

  const items = queueRows.map((queueRow) => ({
    id: queueRow.id,
    name: queueRow.name,
    status: queueRow.status,
    createdAt: queueRow.created_at,
    updatedAt: queueRow.updated_at,
    completedAt: queueRow.completed_at || null,
    lastRunAt: queueRow.last_run_at || null,
    lastError: queueRow.last_error || '',
    sources: sourceRows
      .filter((sourceRow) => sourceRow.queue_id === queueRow.id)
      .map((sourceRow) => ({
        id: sourceRow.id,
        type: sourceRow.type,
        label: sourceRow.label,
        url: sourceRow.url || null,
        ...(cleanText(sourceRow.content_path || '')
          ? {}
          : { content: cleanText(sourceRow.content || '') }),
        contentPath: cleanText(sourceRow.content_path || '') || null,
        contentSize: Math.max(
          Number(sourceRow.content_size) || 0,
          sourceRow.content ? Buffer.byteLength(String(sourceRow.content), 'utf8') : 0
        ),
        stats: safeParseJson(sourceRow.stats_json, {}, `training_queue_source:${sourceRow.id}:stats_json`) || {},
        addedAt: sourceRow.added_at,
      })),
  }));

  return items;
}

function getTableRowCount(tableName) {
  const row = db.prepare(`SELECT COUNT(*) AS count FROM ${tableName}`).get();
  return Number(row?.count || 0);
}

async function readJsonIfExists(filePath, fallbackValue) {
  try {
    const raw = await fs.readFile(filePath, 'utf8');
    return JSON.parse(raw);
  } catch (error) {
    return fallbackValue;
  }
}

async function loadArtifacts() {
  const artifactPaths = getArtifactPaths();
  const defaultKnowledge = createDefaultKnowledgeState();
  const knowledgeIndex = await readJsonIfExists(artifactPaths.knowledgeIndexPath, {
    chunks: defaultKnowledge.chunks,
    replyMemories: defaultKnowledge.replyMemories,
    vocabulary: defaultKnowledge.vocabulary,
    bm25: defaultKnowledge.bm25,
  });
  const languageModel = await readJsonIfExists(
    artifactPaths.languageModelPath,
    defaultKnowledge.languageModel
  );

  return {
    chunks: Array.isArray(knowledgeIndex.chunks) ? knowledgeIndex.chunks : [],
    replyMemories: Array.isArray(knowledgeIndex.replyMemories)
      ? knowledgeIndex.replyMemories
      : [],
    vocabulary: knowledgeIndex.vocabulary || {},
    bm25: knowledgeIndex.bm25 || defaultKnowledge.bm25,
    languageModel: languageModel || defaultKnowledge.languageModel,
  };
}

async function saveArtifacts(currentState) {
  const artifactPaths = getArtifactPaths();
  const compactChunks = (Array.isArray(currentState.knowledge.chunks) ? currentState.knowledge.chunks : [])
    .map((chunk, index) => ({
      id: chunk?.id || `knowledge_chunk_${index}`,
      sourceId: chunk?.sourceId || `knowledge_chunk_${index}`,
      sourceType: chunk?.sourceType || 'knowledge',
      sourceKind: chunk?.sourceKind || 'knowledge',
      label: cleanText(chunk?.label || '') || 'Архив знаний',
      text: cleanText(chunk?.text || ''),
    }))
    .filter((entry) => Boolean(entry.text));
  const compactReplyMemories = (Array.isArray(currentState.knowledge.replyMemories)
    ? currentState.knowledge.replyMemories
    : [])
    .map((entry, index) => {
      const promptText = cleanText(entry?.promptText || '');
      const responseText = cleanText(entry?.responseText || '');
      if (!promptText || !responseText) {
        return null;
      }
      return {
        id: entry?.id || `knowledge_pair_${index}`,
        ownerId: entry?.ownerId || 'knowledge',
        title: cleanText(entry?.title || '') || 'Архив знаний',
        origin: entry?.origin || 'knowledge',
        score: Number(entry?.score || 0.78),
        promptText,
        responseText,
        combinedText: cleanText(entry?.combinedText || `Пользователь: ${promptText}\nАссистент: ${responseText}`),
      };
    })
    .filter(Boolean);
  const knowledgeIndex = {
    chunks: compactChunks,
    replyMemories: compactReplyMemories,
    vocabulary: {},
    bm25: createDefaultKnowledgeState().bm25,
  };
  const manifest = {
    engine: currentState.model.engine,
    updatedAt: currentState.meta.updatedAt,
    trainedEpochs: currentState.model.trainedEpochs,
    parameterCount: currentState.model.parameterCount,
    vocabularySize: currentState.model.vocabularySize,
    sourceCount: currentState.model.sourceCount,
    replyPairCount: currentState.model.replyPairCount,
    corpusSignature: currentState.model.corpusSignature,
    trainingSequenceCount: currentState.knowledge.languageModel.trainingSequenceCount,
    corpusTokenCount: currentState.knowledge.languageModel.corpusTokenCount,
    feedbackExampleCount: currentState.knowledge.languageModel.feedbackExampleCount,
    checkpointReady: currentState.knowledge.languageModel.checkpointReady,
    tokenizerReady: currentState.knowledge.languageModel.tokenizerReady,
    modelSettings: {
      ...currentState.model.configSnapshot?.training,
    },
    computeBackend: currentState.model.computeBackend,
    computeBackendLabel: currentState.model.computeBackendLabel,
    files: artifactPaths,
  };

  await fs.mkdir(artifactPaths.artifactDir, { recursive: true });
  await fs.mkdir(artifactPaths.neuralModelDir, { recursive: true });
  await fs.writeFile(artifactPaths.knowledgeIndexPath, JSON.stringify(knowledgeIndex), 'utf8');
  await fs.writeFile(
    artifactPaths.languageModelPath,
    JSON.stringify(currentState.knowledge.languageModel),
    'utf8'
  );
  await fs.writeFile(artifactPaths.manifestPath, JSON.stringify(manifest), 'utf8');
}

function serializeWorkspaceState(currentState) {
  const snapshot = structuredClone(currentState || createDefaultState());
  if (snapshot?.model && typeof snapshot.model === 'object') {
    delete snapshot.model.storage;
    delete snapshot.model.artifactFiles;
  }
  return snapshot;
}

function readWorkspaceSnapshot(userId = activeWorkspaceUserId) {
  const normalizedUserId = normalizeWorkspaceUserId(userId);
  const row = db.prepare(`
    SELECT state_json
    FROM user_workspaces
    WHERE user_id = ?
    LIMIT 1
  `).get(normalizedUserId);

  if (!row) {
    return null;
  }

  return safeParseJson(
    row.state_json,
    null,
    `user_workspaces:${normalizedUserId}`
  );
}

function readWorkspaceStateForUser(userId = activeWorkspaceUserId) {
  if (!db) {
    throw new Error('Store is not initialized.');
  }

  const workspaceSnapshot = readWorkspaceSnapshot(userId);
  if (!workspaceSnapshot) {
    return null;
  }

  return attachStorageInfo(hydrateState(workspaceSnapshot));
}

function writeWorkspaceSnapshot(currentState, userId = activeWorkspaceUserId) {
  const normalizedUserId = normalizeWorkspaceUserId(userId);
  const snapshot = serializeWorkspaceState(currentState);
  db.prepare(`
    INSERT INTO user_workspaces(user_id, state_json, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(user_id) DO UPDATE SET
      state_json = excluded.state_json,
      updated_at = excluded.updated_at
  `).run(
    normalizedUserId,
    JSON.stringify(snapshot),
    new Date().toISOString()
  );
}

function normalizePersistOptions(options = {}) {
  return {
    writeSources: options.writeSources !== false,
    writeChats: options.writeChats !== false,
    writeArtifacts: options.writeArtifacts !== false,
    writeTrainingQueueRunner: options.writeTrainingQueueRunner !== false,
    writeTrainingQueueMeta: options.writeTrainingQueueMeta !== false,
    writeTrainingQueueSources: options.writeTrainingQueueSources !== false,
  };
}

function writeStateToDatabase(currentState, options = {}) {
  const persistOptions = normalizePersistOptions(options);
  const nextTrainingQueueMetaSignature = computeTrainingQueueMetaSignature(currentState.trainingQueues);
  const nextTrainingQueueSourcesSignature = computeTrainingQueueSourcesSignature(currentState.trainingQueues);
  const nextTrainingQueueRunnerSignature = computeTrainingQueueRunnerSignature(currentState.trainingQueues);
  const shouldWriteTrainingQueueRunner = persistOptions.writeTrainingQueueRunner &&
    nextTrainingQueueRunnerSignature !== lastTrainingQueueRunnerSignature;
  const shouldWriteTrainingQueueMeta = persistOptions.writeTrainingQueueMeta &&
    nextTrainingQueueMetaSignature !== lastTrainingQueueMetaSignature;
  const shouldWriteTrainingQueueSources = persistOptions.writeTrainingQueueSources &&
    nextTrainingQueueSourcesSignature !== lastTrainingQueueSourcesSignature;
  db.exec('BEGIN');
  try {
    setSnapshotRow('meta', currentState.meta);
    setSnapshotRow('settings', currentState.settings);
    setSnapshotRow('model', currentState.model);
    setSnapshotRow('modelRegistry', currentState.modelRegistry || createDefaultModelRegistryState());
    setSnapshotRow('training', currentState.training);
    setSnapshotRow('trainingQueues', {
      migratedToTables: true,
      updatedAt: currentState.meta.updatedAt,
    });
    if (shouldWriteTrainingQueueRunner) {
      setSnapshotRow(
        'trainingQueueRunner',
        currentState.trainingQueues?.runner || createDefaultTrainingQueuesState().runner
      );
    }

    if (persistOptions.writeSources) {
      db.exec('DELETE FROM sources;');

      const insertSource = db.prepare(`
        INSERT INTO sources(id, type, label, url, content, content_path, content_size, stats_json, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);
      currentState.sources.forEach((source) => {
        insertSource.run(
          source.id,
          source.type,
          source.label,
          source.url || null,
          source.content,
          source.contentPath || null,
          Math.max(Number(source.contentSize) || 0, 0),
          JSON.stringify(source.stats || {}),
          source.addedAt
        );
      });
    }

    if (persistOptions.writeChats) {
      db.exec('DELETE FROM messages; DELETE FROM chats;');

      const insertChat = db.prepare(`
        INSERT INTO chats(id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
      `);
      const insertMessage = db.prepare(`
        INSERT INTO messages(id, chat_id, role, content, created_at, metadata_json, sort_order)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `);

      currentState.chats.forEach((chat) => {
        insertChat.run(chat.id, chat.title, chat.createdAt, chat.updatedAt);
        chat.messages.forEach((message, index) => {
          insertMessage.run(
            message.id,
            chat.id,
            message.role,
            message.content,
            message.createdAt,
            JSON.stringify(message.metadata || null),
            index
          );
        });
      });
    }

    if (shouldWriteTrainingQueueMeta) {
      db.exec('DELETE FROM training_queues;');
      const insertQueue = db.prepare(`
        INSERT INTO training_queues(id, name, status, created_at, updated_at, completed_at, last_run_at, last_error, sort_order)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      (currentState.trainingQueues?.items || []).forEach((queue, index) => {
        insertQueue.run(
          queue.id,
          queue.name,
          queue.status || 'pending',
          queue.createdAt || currentState.meta.updatedAt,
          queue.updatedAt || currentState.meta.updatedAt,
          queue.completedAt || null,
          queue.lastRunAt || null,
          cleanText(queue.lastError || ''),
          index
        );
      });
    }

    if (shouldWriteTrainingQueueSources) {
      db.exec('DELETE FROM training_queue_sources;');
      const insertQueueSource = db.prepare(`
        INSERT INTO training_queue_sources(
          id,
          queue_id,
          type,
          label,
          url,
          content,
          content_path,
          content_size,
          stats_json,
          added_at,
          sort_order
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      (currentState.trainingQueues?.items || []).forEach((queue) => {
        (queue.sources || []).forEach((source, index) => {
          insertQueueSource.run(
            source.id,
            queue.id,
            source.type,
            source.label,
            source.url || null,
            '',
            source.contentPath || null,
            Math.max(Number(source.contentSize) || 0, 0),
            JSON.stringify(source.stats || {}),
            source.addedAt,
            index
          );
        });
      });
    }
    db.exec('COMMIT');
    lastTrainingQueueMetaSignature = nextTrainingQueueMetaSignature;
    lastTrainingQueueSourcesSignature = nextTrainingQueueSourcesSignature;
    lastTrainingQueueRunnerSignature = nextTrainingQueueRunnerSignature;
  } catch (error) {
    try {
      db.exec('ROLLBACK');
    } catch (rollbackError) {
      // Ignore rollback failures and surface the original error.
    }
    throw error;
  }
}

function readStateFromDatabase() {
  const legacyTrainingQueues = getSnapshotRow('trainingQueues');
  const persistedTrainingQueueRunner = getSnapshotRow('trainingQueueRunner');
  const persistedTrainingQueueItems = readTrainingQueuesFromDatabase();
  const baseState = hydrateState({
    meta: getSnapshotRow('meta'),
    settings: getSnapshotRow('settings'),
    modelRegistry: getSnapshotRow('modelRegistry'),
    model: getSnapshotRow('model'),
    training: getSnapshotRow('training'),
    trainingQueues: {
      items: persistedTrainingQueueItems.length
        ? persistedTrainingQueueItems
        : (legacyTrainingQueues?.items || []),
      runner: persistedTrainingQueueRunner || legacyTrainingQueues?.runner || createDefaultTrainingQueuesState().runner,
    },
  });

  baseState.sources = db.prepare(`
    SELECT id, type, label, url, content, content_path, content_size, stats_json, added_at
    FROM sources
    ORDER BY added_at DESC
  `).all().map((row) => ({
    id: row.id,
    type: row.type,
    label: row.label,
    url: row.url,
    content: row.content,
    contentPath: cleanText(row.content_path || '') || null,
    contentSize: Math.max(
      Number(row.content_size) || 0,
      row.content ? Buffer.byteLength(String(row.content), 'utf8') : 0
    ),
    stats: safeParseJson(row.stats_json, {}, `source:${row.id}:stats_json`) || {},
    addedAt: row.added_at,
  }));

  const chatRows = db.prepare(`
    SELECT id, title, created_at, updated_at
    FROM chats
    ORDER BY updated_at DESC
  `).all();

  const messageRows = db.prepare(`
    SELECT id, chat_id, role, content, created_at, metadata_json, sort_order
    FROM messages
    ORDER BY chat_id, sort_order ASC
  `).all();

  baseState.chats = chatRows.map((chatRow) => ({
    id: chatRow.id,
    title: chatRow.title,
    createdAt: chatRow.created_at,
    updatedAt: chatRow.updated_at,
    messages: messageRows
      .filter((messageRow) => messageRow.chat_id === chatRow.id)
      .map((messageRow) => ({
        id: messageRow.id,
        role: messageRow.role,
        content: messageRow.content,
        createdAt: messageRow.created_at,
        metadata: messageRow.metadata_json
          ? safeParseJson(messageRow.metadata_json, null, `message:${messageRow.id}:metadata_json`)
          : null,
      })),
  }));

  return normalizePersistedState(baseState);
}

async function importLegacyStateIfNeeded() {
  const row = db.prepare('SELECT COUNT(*) AS count FROM app_snapshot').get();
  if (row.count > 0) {
    return;
  }

  let initialState = hydrateState(createDefaultState());
  try {
    const rawLegacy = await fs.readFile(LEGACY_STORE_FILE, 'utf8');
    initialState = migrateLegacyState(JSON.parse(rawLegacy));
  } catch (error) {
    initialState = hydrateState(createDefaultState());
  }

  writeStateToDatabase(initialState);
  await saveArtifacts(initialState);
}

async function repairStateFromLegacyIfNeeded() {
  if (!state) {
    return;
  }

  const currentVersion = Number(state.meta?.version || 0);
  const needsSourceRepair = currentVersion < 5 && (!Array.isArray(state.sources) || !state.sources.length);
  const needsChatRepair = currentVersion < 5 && (!Array.isArray(state.chats) || !state.chats.length);

  if (!needsSourceRepair && !needsChatRepair) {
    if (currentVersion < 6) {
      state.meta.version = 6;
      writeStateToDatabase(state);
    }
    return;
  }

  try {
    const rawLegacy = await fs.readFile(LEGACY_STORE_FILE, 'utf8');
    const migratedLegacyState = migrateLegacyState(JSON.parse(rawLegacy));
    let repaired = false;

    if (needsSourceRepair && migratedLegacyState.sources.length) {
      state.sources = migratedLegacyState.sources;
      repaired = true;
    }

    if (needsChatRepair && migratedLegacyState.chats.length) {
      state.chats = migratedLegacyState.chats;
      repaired = true;
    }

    if (repaired) {
      state.model = createDefaultModelState();
      state.model.exists = true;
      state.model.lifecycle = 'ready_for_training';
      state.model.status = 'ready';
      state.model.configSnapshot = structuredClone(state.settings);
      state.training = createDefaultTrainingState();
      state.training.message = 'Источники и состояние восстановлены из резервного снимка. Модель готова к новому обучению.';
      state.knowledge = createDefaultKnowledgeState();
    }

    state.meta.version = 6;
    writeStateToDatabase(state);
  } catch (error) {
    if (currentVersion < 6) {
      state.meta.version = 6;
      writeStateToDatabase(state);
    }
  }
}

async function initializeStore() {
  setActiveModelLibraryUser(activeWorkspaceUserId);
  await ensureStorageLayout();
  await importLegacyStateIfNeeded();
  const legacyTrainingQueuesSnapshot = getSnapshotRow('trainingQueues');
  const loadedState = readStateFromDatabase();
  const currentSnapshot = safeStateSignature(loadedState);
  state = normalizePersistedState(loadedState);
  const externalizedTrainingQueues = await externalizeTrainingQueueSources(state.trainingQueues);
  await cleanupTrainingQueueStorage(state.trainingQueues);
  if (externalizedTrainingQueues) {
    log('info', 'Training queue sources were migrated out of SQLite blobs into file-backed storage.', {
      scope: 'store',
      queueCount: state.trainingQueues?.items?.length || 0,
    });
  }
  const hasLegacyTrainingQueues =
    Array.isArray(legacyTrainingQueuesSnapshot?.items) &&
    legacyTrainingQueuesSnapshot.items.length > 0;
  const hasTrainingQueueTableRows = getTableRowCount('training_queues') > 0;
  if (safeStateSignature(state) !== currentSnapshot || externalizedTrainingQueues) {
    writeStateToDatabase(state);
  } else if (hasLegacyTrainingQueues && !hasTrainingQueueTableRows) {
    writeStateToDatabase(state);
  }
  if (externalizedTrainingQueues) {
    try {
      db.exec('VACUUM');
    } catch (error) {
      logError('Failed to compact SQLite database after training queue migration.', error, {
        scope: 'store',
      });
    }
  }
  await repairStateFromLegacyIfNeeded();
  const artifacts = await loadArtifacts();
  state.knowledge = deepMerge(createDefaultKnowledgeState(), artifacts);
  state = attachStorageInfo(state);
  resetTrainingQueuePersistSignatures(state.trainingQueues);
  writeWorkspaceSnapshot(state, activeWorkspaceUserId);
  return state;
}

function getState() {
  if (!state) {
    throw new Error('Store is not initialized.');
  }

  return state;
}

async function persistState(options = {}) {
  if (!state) {
    return;
  }

  const startedAt = Date.now();
  const persistOptions = normalizePersistOptions(options);
  state.meta.updatedAt = new Date().toISOString();
  state = attachStorageInfo(state);

  try {
    writeStateToDatabase(state, persistOptions);
    if (persistOptions.writeArtifacts) {
      await saveArtifacts(state);
    }
    writeWorkspaceSnapshot(state, activeWorkspaceUserId);
  } catch (error) {
    logError('Failed to persist application state.', error, {
      writeSources: persistOptions.writeSources,
      writeChats: persistOptions.writeChats,
      writeArtifacts: persistOptions.writeArtifacts,
    });
    throw error;
  }

  const durationMs = Date.now() - startedAt;
  if (durationMs >= SLOW_PERSIST_THRESHOLD_MS) {
    log('warn', 'State persistence was slow.', {
      durationMs,
      writeSources: persistOptions.writeSources,
      writeChats: persistOptions.writeChats,
      writeArtifacts: persistOptions.writeArtifacts,
    });
  }
}

function enqueueStoreOperation(operation) {
  const run = updateQueue
    .catch(() => undefined)
    .then(operation);
  updateQueue = run.catch(() => undefined);
  return run;
}

async function switchWorkspaceUserInternal(normalizedUserId) {
  if (!state) {
    throw new Error('Store is not initialized.');
  }

  if (normalizedUserId === activeWorkspaceUserId) {
    return state;
  }

  await persistState();
  activeWorkspaceUserId = normalizedUserId;
  setActiveModelLibraryUser(normalizedUserId);
  await ensureStorageLayout();

  const workspaceSnapshot = readWorkspaceSnapshot(normalizedUserId);
  if (workspaceSnapshot) {
    state = attachStorageInfo(hydrateState(workspaceSnapshot));
  } else {
    state = attachStorageInfo(hydrateState(createDefaultState()));
  }

  const artifacts = await loadArtifacts();
  state.knowledge = deepMerge(createDefaultKnowledgeState(), artifacts);
  state = attachStorageInfo(state);

  writeStateToDatabase(state);
  resetTrainingQueuePersistSignatures(state.trainingQueues);
  writeWorkspaceSnapshot(state, normalizedUserId);
  return state;
}

async function switchWorkspaceUser(userId) {
  const normalizedUserId = normalizeWorkspaceUserId(userId);
  return enqueueStoreOperation(async () => switchWorkspaceUserInternal(normalizedUserId));
}

function getActiveWorkspaceUserId() {
  return activeWorkspaceUserId;
}

async function replaceState(nextState, options = {}) {
  state = attachStorageInfo(hydrateState(nextState));
  if (options.persist !== false) {
    await persistState(options);
  }
  return state;
}

async function updateState(mutator, options = {}) {
  const targetUserId = options?.userId
    ? normalizeWorkspaceUserId(options.userId)
    : null;

  return enqueueStoreOperation(async () => {
    if (targetUserId) {
      await switchWorkspaceUserInternal(targetUserId);
    }

    const maybeNextState = await mutator(state);
    if (maybeNextState) {
      state = attachStorageInfo(hydrateState(maybeNextState));
    } else {
      state = attachStorageInfo(state);
    }

    if (options.persist !== false) {
      await persistState(options);
    }

    return state;
  });
}

module.exports = {
  DB_FILE,
  MODEL_MANIFEST_FILE,
  KNOWLEDGE_INDEX_FILE,
  LANGUAGE_MODEL_FILE,
  getArtifactPaths,
  getActiveWorkspaceUserId,
  getState,
  switchWorkspaceUser,
  initializeStore,
  persistState,
  readWorkspaceStateForUser,
  replaceState,
  updateState,
};
