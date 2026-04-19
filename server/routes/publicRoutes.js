const express = require('express');
const { sendError, sendSuccess } = require('../core/apiResponse');

function createPublicRoutes({
  chatShareService,
  readWorkspaceStateForUser,
}) {
  const router = express.Router();

  router.get('/chat-shares/:token', async (request, response) => {
    try {
      const share = chatShareService.getShare(request.params.token);
      if (!share) {
        const error = new Error('Ссылка-приглашение не найдена или больше недоступна.');
        error.statusCode = 404;
        throw error;
      }

      const ownerState = readWorkspaceStateForUser(share.owner_user_id);
      const sharedChat = ownerState?.chats?.find((chat) => chat.id === share.chat_id) || null;
      if (!sharedChat) {
        const error = new Error('Чат по этой ссылке больше недоступен.');
        error.statusCode = 404;
        throw error;
      }

      sendSuccess(response, {
        share: {
          token: share.token,
          createdAt: share.created_at,
        },
        chat: {
          id: sharedChat.id,
          title: sharedChat.title,
          createdAt: sharedChat.createdAt,
          updatedAt: sharedChat.updatedAt,
          messageCount: Array.isArray(sharedChat.messages) ? sharedChat.messages.length : 0,
          messages: Array.isArray(sharedChat.messages) ? sharedChat.messages : [],
        },
      });
    } catch (error) {
      sendError(response, error);
    }
  });

  return router;
}

module.exports = {
  createPublicRoutes,
};
