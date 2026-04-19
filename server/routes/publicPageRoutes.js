const express = require('express');

function createSharedChatViewerHtml(token) {
  const safeToken = String(token || '').replace(/"/g, '&quot;');

  return `<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Поделившийся чат</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #071019;
      --panel: rgba(13, 23, 36, 0.88);
      --panel-border: rgba(255, 255, 255, 0.14);
      --muted: rgba(210, 223, 238, 0.7);
      --text: rgba(248, 251, 255, 0.97);
      --accent: #8ddcc7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      overflow-x: hidden;
      font-family: "Manrope", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top, rgba(27, 115, 178, 0.24), transparent 46%),
        radial-gradient(circle at bottom, rgba(111, 61, 170, 0.22), transparent 40%),
        var(--bg);
      color: var(--text);
    }
    .page {
      width: min(1100px, calc(100vw - 32px));
      max-width: 100%;
      margin: 0 auto;
      padding: 20px 0 40px;
      overflow-x: hidden;
    }
    .hero, .message {
      border: 1px solid var(--panel-border);
      background: var(--panel);
      backdrop-filter: blur(18px);
      border-radius: 24px;
      box-shadow: 0 18px 42px rgba(0,0,0,0.26);
      max-width: 100%;
    }
    .hero {
      padding: 18px 20px;
      margin-bottom: 18px;
    }
    .hero__title {
      margin: 0 0 6px;
      font-size: 1.45rem;
    }
    .hero__meta {
      margin: 0;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .message-list {
      display: grid;
      gap: 14px;
    }
    .message {
      padding: 16px 18px;
    }
    .message--user {
      border-color: rgba(141, 220, 199, 0.26);
      background: rgba(18, 39, 39, 0.82);
    }
    .message__meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 0.82rem;
    }
    .message__content {
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      line-height: 1.62;
      max-width: 100%;
    }
    .message__content pre {
      margin: 12px 0 0;
      padding: 14px;
      overflow: auto;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(7, 14, 26, 0.96);
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      max-width: 100%;
    }
    .message__content code {
      font-family: ui-monospace, Consolas, monospace;
      word-break: break-word;
      overflow-wrap: anywhere;
    }
    .empty, .loading, .error {
      padding: 18px;
      border-radius: 20px;
      border: 1px solid var(--panel-border);
      background: var(--panel);
      color: var(--muted);
    }
    .error {
      color: #ffd2d2;
      border-color: rgba(255, 144, 144, 0.28);
    }
    .hero__actions {
      display: flex;
      gap: 10px;
      margin-top: 14px;
      flex-wrap: wrap;
    }
    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      padding: 0 14px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.18);
      color: var(--text);
      background: rgba(255,255,255,0.08);
      text-decoration: none;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1 class="hero__title">Поделившийся чат</h1>
      <p class="hero__meta">Режим только для чтения по пригласительной ссылке.</p>
      <div class="hero__actions">
        <a class="button" href="/">Открыть студию</a>
      </div>
    </section>
    <div id="app" class="loading">Загрузка чата...</div>
  </div>
  <script>
    (function () {
      const token = "${safeToken}";
      const root = document.getElementById('app');

      function escapeHtml(value) {
        return String(value || '')
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;');
      }

      function formatContent(raw) {
        const text = String(raw || '');
        const codeFencePattern = /\`\`\`([a-zA-Z0-9_-]+)?\\n([\\s\\S]*?)\`\`\`/g;
        let lastIndex = 0;
        let html = '';
        let match;

        while ((match = codeFencePattern.exec(text))) {
          const before = text.slice(lastIndex, match.index);
          if (before) {
            html += '<div>' + escapeHtml(before) + '</div>';
          }

          const language = escapeHtml(match[1] || 'code');
          const code = escapeHtml(match[2] || '');
          html += '<pre><code data-language="' + language + '">' + code + '</code></pre>';
          lastIndex = codeFencePattern.lastIndex;
        }

        const tail = text.slice(lastIndex);
        if (tail) {
          html += '<div>' + escapeHtml(tail) + '</div>';
        }

        return html || '<div></div>';
      }

      function renderChat(payload) {
        const chat = payload && payload.chat ? payload.chat : null;
        if (!chat) {
          root.className = 'error';
          root.textContent = 'Чат по этой ссылке недоступен.';
          return;
        }

        const messages = Array.isArray(chat.messages) ? chat.messages : [];
        const heroTitle = document.querySelector('.hero__title');
        const heroMeta = document.querySelector('.hero__meta');
        heroTitle.textContent = chat.title || 'Поделившийся чат';
        heroMeta.textContent = 'Сообщений: ' + messages.length;

        if (!messages.length) {
          root.className = 'empty';
          root.textContent = 'В этом чате пока нет сообщений.';
          return;
        }

        root.className = 'message-list';
        root.innerHTML = messages.map((message) => {
          const roleLabel = message.role === 'user' ? 'Автор чата' : 'Модель';
          const createdAt = message.createdAt ? new Date(message.createdAt).toLocaleString('ru-RU') : '';
          return [
            '<article class="message ' + (message.role === 'user' ? 'message--user' : 'message--assistant') + '">',
            '<div class="message__meta"><span>' + escapeHtml(roleLabel) + '</span><span>' + escapeHtml(createdAt) + '</span></div>',
            '<div class="message__content">' + formatContent(message.content) + '</div>',
            '</article>',
          ].join('');
        }).join('');
      }

      fetch('/api/public/chat-shares/' + encodeURIComponent(token), { credentials: 'include' })
        .then(async (response) => {
          const payload = await response.json().catch(() => ({}));
          if (!response.ok || payload.ok === false) {
            throw new Error(payload.error || 'Не удалось открыть поделившийся чат.');
          }
          renderChat(payload);
        })
        .catch((error) => {
          root.className = 'error';
          root.textContent = error && error.message ? error.message : 'Не удалось открыть поделившийся чат.';
        });
    })();
  </script>
</body>
</html>`;
}

function createPublicPageRoutes() {
  const router = express.Router();

  router.get('/shared/chat/:token', (request, response) => {
    response.setHeader('Content-Type', 'text/html; charset=utf-8');
    response.status(200).send(createSharedChatViewerHtml(request.params.token));
  });

  return router;
}

module.exports = {
  createPublicPageRoutes,
};
