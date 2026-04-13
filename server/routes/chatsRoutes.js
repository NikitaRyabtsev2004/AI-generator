const express = require('express');
const { sendError, sendSuccess } = require('../core/apiResponse');

function createChatsRoutes({
  engine,
  overloadMiddleware,
}) {
  const router = express.Router();
  const heavy = overloadMiddleware || ((_request, _response, next) => next());

  router.post('/chats', async (_request, response) => {
    try {
      const snapshot = await engine.createChat();
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.delete('/chats/:chatId', async (request, response) => {
    try {
      const snapshot = await engine.deleteChat(request.params.chatId);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/chats/:chatId/messages', heavy, async (request, response) => {
    try {
      const snapshot = await engine.sendMessage(request.params.chatId, request.body?.content);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/messages/:messageId/rating', async (request, response) => {
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

  return router;
}

module.exports = {
  createChatsRoutes,
};
