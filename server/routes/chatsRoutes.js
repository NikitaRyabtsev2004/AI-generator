const express = require('express');
const { sendError, sendSuccess } = require('../core/apiResponse');

function createChatsRoutes({
  engine,
  getState,
  chatShareService,
  overloadMiddleware,
}) {
  const router = express.Router();
  const heavy = overloadMiddleware || ((_request, _response, next) => next());

  function resolveSharedChatUrl(request, token) {
    const forwardedProto = String(request.headers['x-forwarded-proto'] || '').split(',')[0].trim();
    const forwardedHost = String(request.headers['x-forwarded-host'] || '').split(',')[0].trim();
    const protocol = forwardedProto || request.protocol || 'http';
    const host = forwardedHost || request.get('host') || 'localhost:4000';
    return `${protocol}://${host}/shared/chat/${encodeURIComponent(token)}`;
  }

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

  router.post('/chats/:chatId/stop', async (request, response) => {
    try {
      const snapshot = await engine.stopChatReply(request.params.chatId);
      sendSuccess(response, { snapshot });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/chats/:chatId/share', async (request, response) => {
    try {
      const chatId = String(request.params.chatId || '').trim();
      const state = getState();
      const chat = (state.chats || []).find((entry) => entry.id === chatId) || null;
      if (!chat) {
        const error = new Error('Чат не найден или недоступен.');
        error.statusCode = 404;
        throw error;
      }

      const share = chatShareService.createShare({
        ownerUserId: request.auth?.user?.id || '',
        chatId: chat.id,
        chatTitle: chat.title,
      });
      sendSuccess(response, {
        share: {
          token: share.token,
          path: `/shared/chat/${encodeURIComponent(share.token)}`,
          url: resolveSharedChatUrl(request, share.token),
          createdAt: share.createdAt,
          chatId: share.chatId,
        },
      });
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

  router.patch('/messages/:messageId', heavy, async (request, response) => {
    try {
      const snapshot = await engine.editMessage(request.params.messageId, request.body?.content);
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
