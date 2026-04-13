const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs/promises');
const {
  LOG_FILE,
  getRecentLogs,
  initializeLogger,
  log,
  logError,
  subscribeLogs,
} = require('./lib/logger');
const { createModelEngine } = require('./engine/modelEngine');
const { initializeRuntimeConfig } = require('./storage/runtimeConfig');
const { getState, initializeStore, updateState } = require('./storage/store');
const { cleanText } = require('./lib/text');
const { sendError, writeSseEvent } = require('./core/apiResponse');
const { createOverloadGuard, readPositiveIntegerEnv } = require('./core/overloadGuard');
const { createServerStatusCollector } = require('./core/serverStatus');
const { createUploadMiddlewares } = require('./services/uploadSourceService');
const { createSystemRoutes } = require('./routes/systemRoutes');
const { createSourcesRoutes } = require('./routes/sourcesRoutes');
const { createTrainingRoutes } = require('./routes/trainingRoutes');
const { createModelsRoutes } = require('./routes/modelsRoutes');
const { createChatsRoutes } = require('./routes/chatsRoutes');

const PORT = Number(process.env.PORT || 4000);
const SLOW_REQUEST_THRESHOLD_MS = readPositiveIntegerEnv('SLOW_REQUEST_THRESHOLD_MS', 3000);
const SSE_HEARTBEAT_MS = readPositiveIntegerEnv('SSE_HEARTBEAT_MS', 20000);
const SSE_STATE_THROTTLE_MS = readPositiveIntegerEnv('SSE_STATE_THROTTLE_MS', 250);

async function bootstrap() {
  await initializeLogger();
  await initializeRuntimeConfig();
  await initializeStore();
  await updateState(async (state) => {
    if (!Array.isArray(state.chats) || !state.chats.length) {
      state.chats = [];
    }
  });

  const app = express();
  const overloadGuard = createOverloadGuard();
  const { sourceUpload, modelPackageUpload, limits } = createUploadMiddlewares();
  const realtimeClients = new Set();
  let pendingStateBroadcast = null;
  let scheduleStateBroadcast = () => {};

  app.use(cors());
  app.use(express.json({ limit: '8mb' }));

  const instrumentedUpdateState = async (mutator, options = {}) => {
    const nextState = await updateState(mutator, options);
    scheduleStateBroadcast();
    return nextState;
  };

  const engine = createModelEngine({
    getState,
    updateState: instrumentedUpdateState,
  });
  await engine.hydrateState();

  const statusCollector = createServerStatusCollector({
    getState,
    modelEngine: engine,
    getRealtimeClients: () => realtimeClients.size,
    getOverloadSnapshot: overloadGuard.snapshot,
  });

  const broadcastEvent = (eventName, payload) => {
    realtimeClients.forEach((client) => {
      try {
        writeSseEvent(client, eventName, payload);
      } catch (_error) {
        realtimeClients.delete(client);
      }
    });
  };

  scheduleStateBroadcast = () => {
    if (pendingStateBroadcast) {
      return;
    }

    pendingStateBroadcast = setTimeout(async () => {
      pendingStateBroadcast = null;
      try {
        const snapshot = await engine.getRealtimeSnapshot();
        broadcastEvent('snapshot', { snapshot });
      } catch (error) {
        logError('Failed to broadcast realtime snapshot.', error);
      }
    }, SSE_STATE_THROTTLE_MS);
  };

  const unsubscribeLogs = subscribeLogs((entry) => {
    broadcastEvent('log', { entry });
  });

  const handleSseEvents = async (request, response) => {
    response.status(200);
    response.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    response.setHeader('Cache-Control', 'no-cache, no-transform');
    response.setHeader('Connection', 'keep-alive');
    response.setHeader('X-Accel-Buffering', 'no');
    response.flushHeaders?.();
    response.write(': connected\n\n');

    realtimeClients.add(response);

    try {
      const snapshot = await engine.getRealtimeSnapshot();
      writeSseEvent(response, 'snapshot', { snapshot });
      writeSseEvent(response, 'logs', {
        logs: getRecentLogs(120),
        logFile: LOG_FILE,
      });
      writeSseEvent(response, 'status', {
        status: statusCollector.buildStatus(),
      });
    } catch (error) {
      logError('Failed to send initial realtime payload.', error);
    }

    const heartbeatId = setInterval(() => {
      try {
        response.write(': ping\n\n');
      } catch (_error) {
        clearInterval(heartbeatId);
      }
    }, SSE_HEARTBEAT_MS);

    request.on('close', () => {
      clearInterval(heartbeatId);
      realtimeClients.delete(response);
    });
  };

  process.on('unhandledRejection', (error) => {
    logError('Unhandled promise rejection.', error);
    broadcastEvent('server_error', {
      message: cleanText(error?.message || 'Unhandled promise rejection.'),
    });
  });

  process.on('uncaughtException', (error) => {
    logError('Uncaught exception.', error);
    broadcastEvent('server_error', {
      message: cleanText(error?.message || 'Uncaught exception.'),
    });
  });

  app.use((request, response, next) => {
    const startedAt = Date.now();

    response.on('finish', () => {
      const durationMs = Date.now() - startedAt;
      const shouldLogRequest =
        request.path.startsWith('/api/') &&
        request.path !== '/api/events' &&
        (
          response.statusCode >= 400 ||
          durationMs >= SLOW_REQUEST_THRESHOLD_MS ||
          request.path === '/api/model/train' ||
          request.path === '/api/model/pause' ||
          request.path === '/api/model/rollback' ||
          request.path === '/api/sources/files'
        );

      if (!shouldLogRequest) {
        return;
      }

      const level = response.statusCode >= 500
        ? 'error'
        : response.statusCode >= 400 || durationMs >= SLOW_REQUEST_THRESHOLD_MS
          ? 'warn'
          : 'info';
      const message = response.statusCode >= 400
        ? 'HTTP request completed with error status.'
        : 'HTTP request completed.';

      log(level, message, {
        method: request.method,
        path: request.originalUrl || request.url,
        statusCode: response.statusCode,
        durationMs,
      });
    });

    next();
  });

  app.use('/api', createSystemRoutes({
    engine,
    getState,
    getRecentLogs,
    logFile: LOG_FILE,
    statusCollector,
    handleSseEvents,
  }));
  app.use('/api', createSourcesRoutes({
    engine,
    sourceUpload,
    overloadMiddleware: overloadGuard.middleware,
    limits,
  }));
  app.use('/api', createTrainingRoutes({
    engine,
    sourceUpload,
    overloadMiddleware: overloadGuard.middleware,
    limits,
  }));
  app.use('/api', createModelsRoutes({
    engine,
    modelPackageUpload,
    overloadMiddleware: overloadGuard.middleware,
  }));
  app.use('/api', createChatsRoutes({
    engine,
    overloadMiddleware: overloadGuard.middleware,
  }));

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
  } catch (_error) {
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
    sendError(response, error, { limits });
  });

  const server = app.listen(PORT, () => {
    log('info', `Server started on http://localhost:${PORT}.`, {
      port: PORT,
      logFile: LOG_FILE,
    });
    console.log(`Сервер Liquid Glass AI запущен: http://localhost:${PORT}`);
  });

  process.on('exit', () => {
    if (pendingStateBroadcast) {
      clearTimeout(pendingStateBroadcast);
    }
    statusCollector.dispose();
    unsubscribeLogs();
    server.close();
  });
}

bootstrap().catch((error) => {
  initializeLogger()
    .then(() => {
      logError('Failed to bootstrap server.', error);
    })
    .finally(() => {
      console.error('Не удалось запустить сервер:', error);
      process.exit(1);
    });
});
