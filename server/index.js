const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs/promises');
const { fetchPageContent } = require('./content');
const { createModelEngine } = require('./modelEngine');
const { initializeRuntimeConfig } = require('./runtimeConfig');
const { getState, initializeStore, updateState } = require('./store');
const { cleanText, decodeBufferToText } = require('./text');

const PORT = Number(process.env.PORT || 4000);
const DEFAULT_MAX_UPLOAD_FILE_MB = 64;
const DEFAULT_MAX_UPLOAD_FILES = 10;

function readPositiveIntegerEnv(name, fallbackValue) {
  const value = Number(process.env[name]);
  if (!Number.isFinite(value) || value <= 0) {
    return fallbackValue;
  }
  return Math.floor(value);
}

const MAX_UPLOAD_FILE_MB = readPositiveIntegerEnv('MAX_UPLOAD_FILE_MB', DEFAULT_MAX_UPLOAD_FILE_MB);
const MAX_UPLOAD_FILES = readPositiveIntegerEnv('MAX_UPLOAD_FILES', DEFAULT_MAX_UPLOAD_FILES);

const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_UPLOAD_FILE_MB * 1024 * 1024,
    files: MAX_UPLOAD_FILES,
  },
});

app.use(cors());
app.use(express.json({ limit: '8mb' }));

function sendSuccess(response, payload) {
  response.json({
    ok: true,
    ...payload,
  });
}

function sendError(response, error) {
  const message = error?.code === 'LIMIT_FILE_SIZE'
    ? `Файл слишком большой. Максимальный размер: ${MAX_UPLOAD_FILE_MB} МБ на файл.`
    : error?.code === 'LIMIT_FILE_COUNT'
      ? `Слишком много файлов в одном запросе. Максимум: ${MAX_UPLOAD_FILES}.`
      : cleanText(error?.message || 'Неизвестная ошибка сервера.');

  response.status(400).json({
    ok: false,
    error: message,
  });
}

function writeNdjsonChunk(response, payload) {
  response.write(`${JSON.stringify(payload)}\n`);
}

async function bootstrap() {
  await initializeRuntimeConfig();
  await initializeStore();
  await updateState(async (state) => {
    if (!Array.isArray(state.chats) || !state.chats.length) {
      state.chats = [];
    }
  });

  const engine = createModelEngine({
    getState,
    updateState,
  });
  await engine.hydrateState();

  app.get('/api/health', (_request, response) => {
    sendSuccess(response, {
      serverTime: new Date().toISOString(),
      modelLifecycle: getState().model.lifecycle,
    });
  });

  app.get('/api/dashboard', async (request, response) => {
    try {
      const snapshot = await engine.getDashboard(request.query.chatId || null);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  app.post('/api/settings', async (request, response) => {
    try {
      const snapshot = await engine.updateSettings(request.body || {});
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  app.post('/api/sources/files', upload.array('files'), async (request, response) => {
    const files = request.files || [];
    if (!files.length) {
      sendError(response, new Error('Файлы не переданы.'));
      return;
    }

    response.status(200);
    response.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
    response.setHeader('Cache-Control', 'no-cache, no-transform');
    response.setHeader('X-Accel-Buffering', 'no');

    try {
      const totalFiles = files.length;
      let snapshot = null;
      writeNdjsonChunk(response, {
        type: 'processing_progress',
        processedFiles: 0,
        totalFiles,
        percent: 0,
      });

      for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        const content = decodeBufferToText(file.buffer);
        snapshot = await engine.addSource({
          type: 'file',
          label: file.originalname,
          content,
        });

        const processedFiles = index + 1;
        const percent = Number(((processedFiles / totalFiles) * 100).toFixed(1));
        writeNdjsonChunk(response, {
          type: 'processing_progress',
          processedFiles,
          totalFiles,
          percent,
          fileName: file.originalname,
        });
      }

      writeNdjsonChunk(response, {
        type: 'result',
        ok: true,
        snapshot,
      });
      response.end();
    } catch (error) {
      writeNdjsonChunk(response, {
        type: 'error',
        message: cleanText(error?.message || 'Ошибка загрузки файлов.'),
      });
      response.end();
    }
  });

  app.post('/api/sources/url', async (request, response) => {
    try {
      const inputUrl = request.body?.url;
      if (!inputUrl) {
        throw new Error('URL не передан.');
      }

      const page = await fetchPageContent(inputUrl);
      const snapshot = await engine.addSource({
        type: 'url',
        label: page.url,
        url: page.url,
        content: page.content,
      });
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  app.delete('/api/sources/:sourceId', async (request, response) => {
    try {
      const snapshot = await engine.removeSource(request.params.sourceId);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  app.post('/api/model/create', async (_request, response) => {
    try {
      const snapshot = await engine.createFreshModel();
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  app.post('/api/model/train', async (_request, response) => {
    try {
      const snapshot = await engine.trainModel();
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  app.post('/api/model/pause', async (_request, response) => {
    try {
      const snapshot = await engine.pauseTraining();
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  app.post('/api/model/reset', async (_request, response) => {
    try {
      const snapshot = await engine.resetModel();
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  app.post('/api/chats', async (_request, response) => {
    try {
      const snapshot = await engine.createChat();
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  app.delete('/api/chats/:chatId', async (request, response) => {
    try {
      const snapshot = await engine.deleteChat(request.params.chatId);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  app.post('/api/chats/:chatId/messages', async (request, response) => {
    try {
      const snapshot = await engine.sendMessage(request.params.chatId, request.body?.content);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  app.post('/api/messages/:messageId/rating', async (request, response) => {
    try {
      const score = Number(request.body?.score || 0);
      if (![1, -1].includes(score)) {
        throw new Error('Оценка должна быть 1 или -1.');
      }

      const snapshot = await engine.rateMessage(request.params.messageId, score);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  const buildDir = path.join(__dirname, '..', 'build');
  try {
    await fs.access(buildDir);
    app.use(express.static(buildDir));
    app.get('*', async (request, response, next) => {
      if (request.path.startsWith('/api/')) {
        next();
        return;
      }

      response.sendFile(path.join(buildDir, 'index.html'));
    });
  } catch (error) {
    app.get('/', (_request, response) => {
      response.json({
        ok: true,
        message: 'Сервер запущен. Папка build не найдена, поэтому сейчас активен только API-режим.',
      });
    });
  }

  app.use((error, _request, response, _next) => {
    if (response.headersSent) {
      return;
    }
    sendError(response, error);
  });

  app.listen(PORT, () => {
    console.log(`Сервер Liquid Glass AI запущен: http://localhost:${PORT}`);
  });
}

bootstrap().catch((error) => {
  console.error('Не удалось запустить сервер:', error);
  process.exit(1);
});
