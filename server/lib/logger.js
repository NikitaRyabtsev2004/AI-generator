const fs = require('fs/promises');
const path = require('path');
const { EventEmitter } = require('events');

const LOG_DIR = path.join(__dirname, '..', 'data');
const LOG_FILE = path.join(LOG_DIR, 'server.log');
const MAX_RECENT_LOGS = 400;

const emitter = new EventEmitter();
const recentLogs = [];
let appendQueue = Promise.resolve();

function serializeError(error) {
  if (!error) {
    return null;
  }

  return {
    name: String(error.name || 'Error'),
    message: String(error.message || ''),
    stack: typeof error.stack === 'string' ? error.stack : '',
    code: error.code || '',
  };
}

function normalizeMessage(message) {
  return String(message || '').trim() || 'Empty log message';
}

function normalizeDetails(details) {
  if (!details || typeof details !== 'object' || Array.isArray(details)) {
    return {};
  }

  return Object.entries(details).reduce((result, [key, value]) => {
    if (value === undefined) {
      return result;
    }

    if (value instanceof Error) {
      result[key] = serializeError(value);
      return result;
    }

    result[key] = value;
    return result;
  }, {});
}

function pushRecentLog(entry) {
  recentLogs.push(entry);
  if (recentLogs.length > MAX_RECENT_LOGS) {
    recentLogs.splice(0, recentLogs.length - MAX_RECENT_LOGS);
  }
}

function appendLogLine(entry) {
  appendQueue = appendQueue
    .catch(() => undefined)
    .then(() => fs.appendFile(LOG_FILE, `${JSON.stringify(entry)}\n`, 'utf8'))
    .catch(() => undefined);
}

async function initializeLogger() {
  await fs.mkdir(LOG_DIR, { recursive: true });
}

function log(level, message, details = {}) {
  const entry = {
    id: `log_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    createdAt: new Date().toISOString(),
    level: String(level || 'info').toLowerCase(),
    message: normalizeMessage(message),
    details: normalizeDetails(details),
  };

  pushRecentLog(entry);
  appendLogLine(entry);
  emitter.emit('log', entry);
  return entry;
}

function logError(message, error, details = {}) {
  return log('error', message, {
    ...details,
    error: serializeError(error),
  });
}

function getRecentLogs(limit = 120) {
  const safeLimit = Math.max(1, Number(limit) || 120);
  return recentLogs.slice(-safeLimit).reverse();
}

function subscribeLogs(listener) {
  emitter.on('log', listener);
  return () => emitter.off('log', listener);
}

module.exports = {
  LOG_FILE,
  getRecentLogs,
  initializeLogger,
  log,
  logError,
  subscribeLogs,
};
