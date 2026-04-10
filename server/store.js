const fs = require('fs/promises');
const path = require('path');
const Database = require('better-sqlite3');
const {
  createDefaultKnowledgeState,
  createDefaultModelState,
  createDefaultState,
  createDefaultTrainingState,
} = require('./config');
const { RUNTIME_CONFIG_PATH } = require('./runtimeConfig');
const { cleanText, computeStats, inferChatTitle } = require('./text');

const DATA_DIR = path.join(__dirname, 'data');
const ARTIFACT_DIR = path.join(__dirname, 'artifacts');
const DB_FILE = path.join(DATA_DIR, 'studio.db');
const LEGACY_STORE_FILE = path.join(DATA_DIR, 'studio-store.json');
const KNOWLEDGE_INDEX_FILE = path.join(ARTIFACT_DIR, 'knowledge-index.json');
const LANGUAGE_MODEL_FILE = path.join(ARTIFACT_DIR, 'language-model.json');
const MODEL_MANIFEST_FILE = path.join(ARTIFACT_DIR, 'model-manifest.json');
const NEURAL_MODEL_DIR = path.join(ARTIFACT_DIR, 'neural-model');
const TOKENIZER_FILE = path.join(NEURAL_MODEL_DIR, 'tokenizer.json');
const NEURAL_WEIGHTS_FILE = path.join(NEURAL_MODEL_DIR, 'weights.bin');
const NEURAL_SPEC_FILE = path.join(NEURAL_MODEL_DIR, 'weights-spec.json');

let db = null;
let state = null;
let updateQueue = Promise.resolve();

function getArtifactPaths() {
  return {
    databasePath: DB_FILE,
    artifactDir: ARTIFACT_DIR,
    manifestPath: MODEL_MANIFEST_FILE,
    knowledgeIndexPath: KNOWLEDGE_INDEX_FILE,
    languageModelPath: LANGUAGE_MODEL_FILE,
    neuralModelDir: NEURAL_MODEL_DIR,
    tokenizerPath: TOKENIZER_FILE,
    neuralWeightsPath: NEURAL_WEIGHTS_FILE,
    neuralSpecPath: NEURAL_SPEC_FILE,
    runtimeConfigPath: RUNTIME_CONFIG_PATH,
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

  return {
    ...source,
    label: cleanText(source.label || '') || source.label || 'Источник',
    url: source.url || null,
    content,
    stats: computeStats(content),
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

function normalizePersistedState(nextState) {
  return {
    ...nextState,
    sources: Array.isArray(nextState.sources) ? nextState.sources.map(normalizeSource) : [],
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
    chats: Array.isArray(rawLegacy?.chats) ? rawLegacy.chats : [],
    model: createDefaultModelState(),
    training: createDefaultTrainingState(),
    knowledge: createDefaultKnowledgeState(),
  });
}

async function ensureStorageLayout() {
  await fs.mkdir(DATA_DIR, { recursive: true });
  await fs.mkdir(ARTIFACT_DIR, { recursive: true });
  await fs.mkdir(NEURAL_MODEL_DIR, { recursive: true });

  if (!db) {
    db = new Database(DB_FILE);
    db.exec(`
      CREATE TABLE IF NOT EXISTS app_snapshot (
        name TEXT PRIMARY KEY,
        json_value TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS sources (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        label TEXT NOT NULL,
        url TEXT,
        content TEXT NOT NULL,
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
    `);
  }
}

function getSnapshotRow(name) {
  const row = db.prepare('SELECT json_value FROM app_snapshot WHERE name = ?').get(name);
  return row ? JSON.parse(row.json_value) : undefined;
}

function setSnapshotRow(name, value) {
  db.prepare(`
    INSERT INTO app_snapshot(name, json_value)
    VALUES (?, ?)
    ON CONFLICT(name) DO UPDATE SET json_value = excluded.json_value
  `).run(name, JSON.stringify(value));
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
  const defaultKnowledge = createDefaultKnowledgeState();
  const knowledgeIndex = await readJsonIfExists(KNOWLEDGE_INDEX_FILE, {
    chunks: defaultKnowledge.chunks,
    replyMemories: defaultKnowledge.replyMemories,
    vocabulary: defaultKnowledge.vocabulary,
    bm25: defaultKnowledge.bm25,
  });
  const languageModel = await readJsonIfExists(
    LANGUAGE_MODEL_FILE,
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
  const knowledgeIndex = {
    chunks: currentState.knowledge.chunks,
    replyMemories: currentState.knowledge.replyMemories,
    vocabulary: currentState.knowledge.vocabulary,
    bm25: currentState.knowledge.bm25,
  };
  const manifest = {
    engine: currentState.model.engine,
    updatedAt: currentState.meta.updatedAt,
    trainedEpochs: currentState.model.trainedEpochs,
    sourceCount: currentState.model.sourceCount,
    replyPairCount: currentState.model.replyPairCount,
    corpusSignature: currentState.model.corpusSignature,
    files: artifactPaths,
  };

  await fs.writeFile(KNOWLEDGE_INDEX_FILE, JSON.stringify(knowledgeIndex, null, 2), 'utf8');
  await fs.writeFile(
    LANGUAGE_MODEL_FILE,
    JSON.stringify(currentState.knowledge.languageModel, null, 2),
    'utf8'
  );
  await fs.writeFile(MODEL_MANIFEST_FILE, JSON.stringify(manifest, null, 2), 'utf8');
}

function writeStateToDatabase(currentState) {
  db.exec('BEGIN');
  try {
    setSnapshotRow('meta', currentState.meta);
    setSnapshotRow('settings', currentState.settings);
    setSnapshotRow('model', currentState.model);
    setSnapshotRow('training', currentState.training);

    db.exec('DELETE FROM messages; DELETE FROM chats; DELETE FROM sources;');

    const insertSource = db.prepare(`
      INSERT INTO sources(id, type, label, url, content, stats_json, added_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    currentState.sources.forEach((source) => {
      insertSource.run(
        source.id,
        source.type,
        source.label,
        source.url || null,
        source.content,
        JSON.stringify(source.stats || {}),
        source.addedAt
      );
    });

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
    db.exec('COMMIT');
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
  const baseState = hydrateState({
    meta: getSnapshotRow('meta'),
    settings: getSnapshotRow('settings'),
    model: getSnapshotRow('model'),
    training: getSnapshotRow('training'),
  });

  baseState.sources = db.prepare(`
    SELECT id, type, label, url, content, stats_json, added_at
    FROM sources
    ORDER BY added_at DESC
  `).all().map((row) => ({
    id: row.id,
    type: row.type,
    label: row.label,
    url: row.url,
    content: row.content,
    stats: JSON.parse(row.stats_json || '{}'),
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
        metadata: messageRow.metadata_json ? JSON.parse(messageRow.metadata_json) : null,
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
    if (currentVersion < 5) {
      state.meta.version = 5;
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

    state.meta.version = 5;
    writeStateToDatabase(state);
  } catch (error) {
    if (currentVersion < 5) {
      state.meta.version = 5;
      writeStateToDatabase(state);
    }
  }
}

async function initializeStore() {
  await ensureStorageLayout();
  await importLegacyStateIfNeeded();
  const loadedState = readStateFromDatabase();
  const currentSnapshot = JSON.stringify(loadedState);
  state = normalizePersistedState(loadedState);
  if (JSON.stringify(state) !== currentSnapshot) {
    writeStateToDatabase(state);
  }
  await repairStateFromLegacyIfNeeded();
  const artifacts = await loadArtifacts();
  state.knowledge = deepMerge(createDefaultKnowledgeState(), artifacts);
  state = attachStorageInfo(state);
  return state;
}

function getState() {
  if (!state) {
    throw new Error('Store is not initialized.');
  }

  return state;
}

async function persistState() {
  if (!state) {
    return;
  }

  state.meta.updatedAt = new Date().toISOString();
  state = attachStorageInfo(state);
  writeStateToDatabase(state);
  await saveArtifacts(state);
}

async function replaceState(nextState, options = {}) {
  state = attachStorageInfo(hydrateState(nextState));
  if (options.persist !== false) {
    await persistState();
  }
  return state;
}

async function updateState(mutator, options = {}) {
  updateQueue = updateQueue.then(async () => {
    const maybeNextState = await mutator(state);
    if (maybeNextState) {
      state = attachStorageInfo(hydrateState(maybeNextState));
    } else {
      state = attachStorageInfo(state);
    }

    if (options.persist !== false) {
      await persistState();
    }

    return state;
  });

  return updateQueue;
}

module.exports = {
  DB_FILE,
  MODEL_MANIFEST_FILE,
  KNOWLEDGE_INDEX_FILE,
  LANGUAGE_MODEL_FILE,
  getArtifactPaths,
  getState,
  initializeStore,
  persistState,
  replaceState,
  updateState,
};
