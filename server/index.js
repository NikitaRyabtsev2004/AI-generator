const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs/promises');
const { fetchPageContent } = require('./content');
const { createModelEngine } = require('./modelEngine');
const { initializeRuntimeConfig } = require('./runtimeConfig');
const { getState, initializeStore, updateState } = require('./store');
const { decodeBufferToText } = require('./text');

const PORT = Number(process.env.PORT || 4000);
const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 12 * 1024 * 1024,
    files: 10,
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
  response.status(400).json({
    ok: false,
    error: error.message || 'Unknown server error',
  });
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
    try {
      const files = request.files || [];
      if (!files.length) {
        throw new Error('Не переданы файлы.');
      }

      let snapshot = null;
      for (const file of files) {
        const content = decodeBufferToText(file.buffer);
        snapshot = await engine.addSource({
          type: 'file',
          label: file.originalname,
          content,
        });
      }

      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
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
        message: 'Server is running. Build directory was not found, so only API mode is active.',
      });
    });
  }

  app.listen(PORT, () => {
    console.log(`Liquid Glass AI server listening on http://localhost:${PORT}`);
  });
}

bootstrap().catch((error) => {
  console.error('Failed to bootstrap server:', error);
  process.exit(1);
});
