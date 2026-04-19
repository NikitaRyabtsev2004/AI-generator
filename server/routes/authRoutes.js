const express = require('express');
const { sendError, sendSuccess } = require('../core/apiResponse');

function createAuthRoutes({ authService }) {
  const router = express.Router();

  router.get('/session', (request, response) => {
    try {
      const auth = authService.resolveAuthFromRequest(request);
      sendSuccess(response, {
        authenticated: Boolean(auth?.user),
        user: auth?.user || null,
      });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/register/start', async (request, response) => {
    try {
      await authService.startRegistration({
        email: request.body?.email,
        password: request.body?.password,
      });
      sendSuccess(response, {
        message: 'Код подтверждения отправлен на email.',
      });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/register/verify', async (request, response) => {
    try {
      const result = authService.verifyRegistration({
        email: request.body?.email,
        code: request.body?.code,
      });
      authService.setSessionCookie(response, result.session.token, result.session.expiresAt);
      sendSuccess(response, {
        user: result.user,
      });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/login', async (request, response) => {
    try {
      const result = authService.login({
        email: request.body?.email,
        password: request.body?.password,
        requestIp: request.ip || request.socket?.remoteAddress || '',
      });
      authService.setSessionCookie(response, result.session.token, result.session.expiresAt);
      sendSuccess(response, {
        user: result.user,
      });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/logout', (request, response) => {
    try {
      authService.logout(request, response);
      sendSuccess(response, {
        loggedOut: true,
      });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/password-reset/start', async (request, response) => {
    try {
      await authService.startPasswordReset({
        email: request.body?.email,
      });
      sendSuccess(response, {
        message: 'Если такой email существует, код отправлен.',
      });
    } catch (error) {
      sendError(response, error);
    }
  });

  router.post('/password-reset/confirm', async (request, response) => {
    try {
      const result = authService.confirmPasswordReset({
        email: request.body?.email,
        code: request.body?.code,
        newPassword: request.body?.newPassword,
      });
      authService.setSessionCookie(response, result.session.token, result.session.expiresAt);
      sendSuccess(response, {
        user: result.user,
      });
    } catch (error) {
      sendError(response, error);
    }
  });

  return router;
}

module.exports = {
  createAuthRoutes,
};

