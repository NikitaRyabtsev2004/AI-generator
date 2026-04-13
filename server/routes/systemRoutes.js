const express = require('express');
const { sendError, sendSuccess } = require('../core/apiResponse');

function createSystemRoutes({
  engine,
  getState,
  getRecentLogs,
  logFile,
  statusCollector,
  handleSseEvents,
}) {
  const router = express.Router();

  router.get('/health', (_request, response) => {
    sendSuccess(response, {
      serverTime: new Date().toISOString(),
      modelLifecycle: getState().model.lifecycle,
      trainingStatus: getState().training.status,
    });
  });

  router.get('/status', (_request, response) => {
    sendSuccess(response, {
      status: statusCollector.buildStatus(),
    });
  });

  router.get('/logs/recent', (request, response) => {
    const limit = Math.max(1, Math.min(Number(request.query.limit) || 120, 400));
    sendSuccess(response, {
      logs: getRecentLogs(limit),
      logFile,
    });
  });

  router.get('/events', handleSseEvents);

  router.get('/dashboard', async (request, response) => {
    try {
      const snapshot = await engine.getDashboard(request.query.chatId || null);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  return router;
}

module.exports = {
  createSystemRoutes,
};
