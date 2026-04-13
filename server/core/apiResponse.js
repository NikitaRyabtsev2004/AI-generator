const { cleanText } = require('../lib/text');
const { logError } = require('../lib/logger');

function sendSuccess(response, payload = {}) {
  response.json({
    ok: true,
    ...payload,
  });
}

function resolveErrorMessage(error, limits = {}) {
  if (error?.code === 'LIMIT_FILE_SIZE') {
    return `Файл слишком большой. Максимальный размер: ${limits.maxUploadFileMb || 1} МБ на файл.`;
  }

  if (error?.code === 'LIMIT_FILE_COUNT') {
    return `Слишком много файлов в одном запросе. Максимум: ${limits.maxUploadFiles || 200}.`;
  }

  return cleanText(error?.message || 'Неизвестная ошибка сервера.');
}

function sendError(response, error, options = {}) {
  const statusCode = Math.max(400, Number(options.statusCode) || 400);
  const message = resolveErrorMessage(error, options.limits || {});

  logError(options.logMessage || 'API request failed.', error, {
    method: response.req?.method || '',
    path: response.req?.originalUrl || response.req?.url || '',
    statusCode,
  });

  response.status(statusCode).json({
    ok: false,
    error: message,
  });
}

function writeNdjsonChunk(response, payload) {
  response.write(`${JSON.stringify(payload)}\n`);
}

function writeSseEvent(response, eventName, payload) {
  response.write(`event: ${eventName}\n`);
  response.write(`data: ${JSON.stringify(payload)}\n\n`);
}

module.exports = {
  sendSuccess,
  sendError,
  writeNdjsonChunk,
  writeSseEvent,
};
