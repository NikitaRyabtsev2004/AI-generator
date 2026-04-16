const express = require('express');
const { sendError, sendSuccess, writeNdjsonChunk } = require('../core/apiResponse');
const { cleanText } = require('../lib/text');
const { fetchPageContent } = require('../lib/content');
const { prepareUploadedSources } = require('../services/uploadSourceService');

function createSourcesRoutes({
  engine,
  sourceUpload,
  overloadMiddleware,
  limits,
}) {
  const router = express.Router();
  const heavy = overloadMiddleware || ((_request, _response, next) => next());

  router.post('/sources/files', heavy, sourceUpload.array('files'), async (request, response) => {
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
        mode: 'knowledge',
      });
      const snapshot = await engine.addSources(preparedSources);

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

  router.post('/sources/url', heavy, async (request, response) => {
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

  router.delete('/sources', heavy, async (_request, response) => {
    try {
      const snapshot = await engine.clearSources();
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.delete('/sources/:sourceId', heavy, async (request, response) => {
    try {
      const snapshot = await engine.removeSource(request.params.sourceId);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  return router;
}

module.exports = {
  createSourcesRoutes,
};
