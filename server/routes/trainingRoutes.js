const express = require('express');
const { sendError, sendSuccess, writeNdjsonChunk } = require('../core/apiResponse');
const { cleanText } = require('../lib/text');
const { prepareUploadedSources } = require('../services/uploadSourceService');

function createTrainingRoutes({
  engine,
  sourceUpload,
  overloadMiddleware,
  limits,
}) {
  const router = express.Router();
  const heavy = overloadMiddleware || ((_request, _response, next) => next());

  router.post('/training-queues', heavy, async (request, response) => {
    try {
      const snapshot = await engine.createTrainingQueue(request.body?.name || '');
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/training-queues/:queueId/files', heavy, sourceUpload.array('files'), async (request, response) => {
    const files = request.files || [];
    if (!files.length) {
      sendError(response, new Error('Файлы не переданы.'), { limits });
      return;
    }

    response.status(200);
    response.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
    response.setHeader('Cache-Control', 'no-cache, no-transform');
    response.setHeader('X-Accel-Buffering', 'no');

    try {
      const preparedSources = await prepareUploadedSources(files, response, {
        mode: 'training_queue',
      });
      const snapshot = await engine.addSourcesToTrainingQueue(request.params.queueId, preparedSources);

      writeNdjsonChunk(response, {
        type: 'result',
        ok: true,
        snapshot,
      });
      response.end();
    } catch (error) {
      writeNdjsonChunk(response, {
        type: 'error',
        message: cleanText(error?.message || 'Ошибка загрузки файлов в очередь обучения.'),
      });
      response.end();
    }
  });

  router.delete('/training-queues/:queueId/files/:sourceId', heavy, async (request, response) => {
    try {
      const snapshot = await engine.removeTrainingQueueSource(
        request.params.queueId,
        request.params.sourceId
      );
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.delete('/training-queues/:queueId', heavy, async (request, response) => {
    try {
      const snapshot = await engine.deleteTrainingQueue(request.params.queueId);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/model/train', heavy, async (_request, response) => {
    try {
      const snapshot = await engine.trainModel();
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/model/pause', heavy, async (_request, response) => {
    try {
      const snapshot = await engine.pauseTraining();
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/model/reset', heavy, async (_request, response) => {
    try {
      const snapshot = await engine.resetModel();
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/model/rollback', heavy, async (_request, response) => {
    try {
      const snapshot = await engine.rollbackTrainingToCheckpoint();
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  return router;
}

module.exports = {
  createTrainingRoutes,
};
