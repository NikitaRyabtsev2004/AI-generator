const express = require('express');
const { sendError, sendSuccess } = require('../core/apiResponse');

function createModelsRoutes({
  engine,
  modelPackageUpload,
  overloadMiddleware,
}) {
  const router = express.Router();
  const heavy = overloadMiddleware || ((_request, _response, next) => next());

  router.post('/settings', async (request, response) => {
    try {
      const snapshot = await engine.updateSettings(request.body || {});
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/runtime-config', async (request, response) => {
    try {
      const snapshot = await engine.updateRuntimeSettings(request.body || {});
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/model/create', heavy, async (request, response) => {
    try {
      const snapshot = await engine.createFreshModel(request.body?.name || '');
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/model/create-api', heavy, async (request, response) => {
    try {
      const snapshot = await engine.createApiModel(request.body || {});
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.patch('/models/:modelId/api', heavy, async (request, response) => {
    try {
      const snapshot = await engine.updateApiModel(request.params.modelId, request.body || {});
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/models/:modelId/select', heavy, async (request, response) => {
    try {
      const snapshot = await engine.selectModel(request.params.modelId);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.delete('/models/:modelId', heavy, async (request, response) => {
    try {
      const snapshot = await engine.deleteModelFromLibrary(request.params.modelId);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.get('/model/export', heavy, async (_request, response) => {
    try {
      const modelPackage = await engine.exportModelPackage();
      response.setHeader('Content-Type', modelPackage.contentType || 'application/octet-stream');
      response.setHeader('Content-Disposition', `attachment; filename="${modelPackage.fileName || 'model-export.aistudio.json'}"`);
      response.status(200).send(modelPackage.buffer);
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/model/import', heavy, modelPackageUpload.single('file'), async (request, response) => {
    try {
      if (!request.file?.buffer?.length) {
        throw new Error('Файл пакета модели не передан.');
      }

      let parsedPayload = null;
      try {
        parsedPayload = JSON.parse(request.file.buffer.toString('utf8'));
      } catch (_error) {
        throw new Error('Не удалось разобрать файл пакета модели. Ожидается корректный JSON.');
      }

      const snapshot = await engine.importModelPackage(parsedPayload);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  return router;
}

module.exports = {
  createModelsRoutes,
};
