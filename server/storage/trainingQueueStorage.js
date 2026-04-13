const fs = require('fs/promises');
const path = require('path');
const { cleanText } = require('../lib/text');

const DATA_DIR = path.join(__dirname, '..', 'data');
const TRAINING_QUEUE_STORAGE_DIR = path.join(DATA_DIR, 'training-queue-sources');
const TRAINING_JOB_STORAGE_DIR = path.join(DATA_DIR, 'training-jobs');
const STALE_TRAINING_JOB_MAX_AGE_MS = 1000 * 60 * 60 * 24;

function normalizePathSegment(value, fallback = 'item') {
  const normalized = String(value || fallback).trim().replace(/[^a-zA-Z0-9._-]+/g, '_');
  return normalized || fallback;
}

function getTrainingQueueDirectory(queueId) {
  return path.join(TRAINING_QUEUE_STORAGE_DIR, normalizePathSegment(queueId, 'queue'));
}

function getTrainingQueueSourceFilePath(queueId, sourceId) {
  return path.join(
    getTrainingQueueDirectory(queueId),
    `${normalizePathSegment(sourceId, 'source')}.txt`
  );
}

async function removeDirectoryIfEmpty(directoryPath) {
  if (!directoryPath) {
    return;
  }

  try {
    const entries = await fs.readdir(directoryPath);
    if (!entries.length) {
      await fs.rmdir(directoryPath);
    }
  } catch (_error) {
    // Ignore missing or busy directories.
  }
}

async function cleanupStaleTrainingJobFiles(maxAgeMs = STALE_TRAINING_JOB_MAX_AGE_MS) {
  await fs.mkdir(TRAINING_JOB_STORAGE_DIR, { recursive: true });
  const entries = await fs.readdir(TRAINING_JOB_STORAGE_DIR, { withFileTypes: true });
  const expirationTime = Date.now() - Math.max(Number(maxAgeMs) || 0, 0);

  await Promise.all(entries.map(async (entry) => {
    if (!entry.isFile() || !entry.name.endsWith('.json')) {
      return;
    }

    const targetPath = path.join(TRAINING_JOB_STORAGE_DIR, entry.name);
    try {
      const stats = await fs.stat(targetPath);
      if (stats.mtimeMs < expirationTime) {
        await fs.rm(targetPath, { force: true });
      }
    } catch (_error) {
      // Ignore races with concurrent cleanup.
    }
  }));
}

async function ensureTrainingQueueStorageLayout() {
  await fs.mkdir(TRAINING_QUEUE_STORAGE_DIR, { recursive: true });
  await fs.mkdir(TRAINING_JOB_STORAGE_DIR, { recursive: true });
  await cleanupStaleTrainingJobFiles();
}

async function writeTrainingQueueSourceContent(queueId, sourceId, content) {
  const normalizedContent = cleanText(content || '');
  const targetPath = getTrainingQueueSourceFilePath(queueId, sourceId);
  await fs.mkdir(path.dirname(targetPath), { recursive: true });
  await fs.writeFile(targetPath, normalizedContent, 'utf8');

  return {
    contentPath: targetPath,
    contentSize: Buffer.byteLength(normalizedContent, 'utf8'),
  };
}

async function readTrainingQueueSourceContent(source = {}) {
  if (source.contentPath) {
    try {
      const raw = await fs.readFile(source.contentPath, 'utf8');
      return cleanText(raw);
    } catch (_error) {
      // Fall back to inline content if it still exists.
    }
  }

  return cleanText(source.content || '');
}

async function removeTrainingQueueSourceContent(source = {}) {
  if (!source.contentPath) {
    return;
  }

  try {
    await fs.rm(source.contentPath, { force: true });
  } catch (_error) {
    // Ignore missing files.
  }

  await removeDirectoryIfEmpty(path.dirname(source.contentPath));
}

async function removeTrainingQueueDirectory(queueId) {
  await fs.rm(getTrainingQueueDirectory(queueId), {
    recursive: true,
    force: true,
  });
}

async function externalizeTrainingQueueSources(trainingQueues = {}) {
  await ensureTrainingQueueStorageLayout();
  let changed = false;
  const items = Array.isArray(trainingQueues.items) ? trainingQueues.items : [];

  for (const queue of items) {
    const queueId = queue?.id;
    const sources = Array.isArray(queue?.sources) ? queue.sources : [];

    for (const source of sources) {
      if (!queueId || !source?.id) {
        continue;
      }

      if (source.contentPath) {
        if (!Number.isFinite(Number(source.contentSize)) || Number(source.contentSize) <= 0) {
          try {
            const stats = await fs.stat(source.contentPath);
            source.contentSize = stats.size;
            changed = true;
          } catch (_error) {
            const inlineContent = cleanText(source.content || '');
            if (inlineContent) {
              const persisted = await writeTrainingQueueSourceContent(queueId, source.id, inlineContent);
              source.contentPath = persisted.contentPath;
              source.contentSize = persisted.contentSize;
              delete source.content;
              changed = true;
            }
          }
        }
        continue;
      }

      const inlineContent = cleanText(source.content || '');
      if (!inlineContent) {
        delete source.content;
        continue;
      }

      const persisted = await writeTrainingQueueSourceContent(queueId, source.id, inlineContent);
      source.contentPath = persisted.contentPath;
      source.contentSize = persisted.contentSize;
      delete source.content;
      changed = true;
    }
  }

  return changed;
}

async function cleanupTrainingQueueStorage(trainingQueues = {}) {
  await ensureTrainingQueueStorageLayout();
  const items = Array.isArray(trainingQueues.items) ? trainingQueues.items : [];
  const expectedQueueDirs = new Map();

  items.forEach((queue) => {
    if (!queue?.id) {
      return;
    }

    const queueDir = getTrainingQueueDirectory(queue.id);
    const expectedFiles = new Set();
    (Array.isArray(queue.sources) ? queue.sources : []).forEach((source) => {
      if (!source?.id) {
        return;
      }
      expectedFiles.add(getTrainingQueueSourceFilePath(queue.id, source.id));
    });
    expectedQueueDirs.set(queueDir, expectedFiles);
  });

  const queueEntries = await fs.readdir(TRAINING_QUEUE_STORAGE_DIR, { withFileTypes: true });
  await Promise.all(queueEntries.map(async (entry) => {
    if (!entry.isDirectory()) {
      return;
    }

    const queueDir = path.join(TRAINING_QUEUE_STORAGE_DIR, entry.name);
    const expectedFiles = expectedQueueDirs.get(queueDir);
    if (!expectedFiles) {
      await fs.rm(queueDir, { recursive: true, force: true });
      return;
    }

    const files = await fs.readdir(queueDir, { withFileTypes: true });
    await Promise.all(files.map(async (fileEntry) => {
      if (!fileEntry.isFile()) {
        return;
      }

      const filePath = path.join(queueDir, fileEntry.name);
      if (!expectedFiles.has(filePath)) {
        await fs.rm(filePath, { force: true });
      }
    }));

    await removeDirectoryIfEmpty(queueDir);
  }));
}

async function createTrainingJobPayloadFile(payload) {
  await ensureTrainingQueueStorageLayout();
  const fileStamp = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const targetPath = path.join(TRAINING_JOB_STORAGE_DIR, `training-job-${fileStamp}.json`);
  await fs.writeFile(targetPath, JSON.stringify(payload), 'utf8');
  return targetPath;
}

async function removeTrainingJobPayloadFile(filePath) {
  if (!filePath) {
    return;
  }

  try {
    await fs.rm(filePath, { force: true });
  } catch (_error) {
    // Ignore missing files.
  }
}

module.exports = {
  TRAINING_QUEUE_STORAGE_DIR,
  TRAINING_JOB_STORAGE_DIR,
  cleanupTrainingQueueStorage,
  createTrainingJobPayloadFile,
  ensureTrainingQueueStorageLayout,
  externalizeTrainingQueueSources,
  getTrainingQueueDirectory,
  getTrainingQueueSourceFilePath,
  readTrainingQueueSourceContent,
  removeTrainingJobPayloadFile,
  removeTrainingQueueDirectory,
  removeTrainingQueueSourceContent,
  writeTrainingQueueSourceContent,
};
