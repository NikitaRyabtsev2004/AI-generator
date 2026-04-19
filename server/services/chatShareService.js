const crypto = require('crypto');
const Database = require('better-sqlite3');
const { cleanText } = require('../lib/text');

function normalizeToken(token) {
  return cleanText(String(token || ''));
}

function createChatShareService({ dbPath }) {
  const db = new Database(dbPath);
  db.pragma('journal_mode = PERSIST');
  db.pragma('busy_timeout = 5000');

  db.exec(`
    CREATE TABLE IF NOT EXISTS chat_share_links (
      token TEXT PRIMARY KEY,
      owner_user_id TEXT NOT NULL,
      chat_id TEXT NOT NULL,
      chat_title TEXT NOT NULL,
      created_at TEXT NOT NULL,
      revoked_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_chat_share_links_owner_chat
    ON chat_share_links(owner_user_id, chat_id, revoked_at);
  `);

  function createShare({ ownerUserId, chatId, chatTitle = '' }) {
    const normalizedOwnerUserId = cleanText(ownerUserId);
    const normalizedChatId = cleanText(chatId);
    if (!normalizedOwnerUserId || !normalizedChatId) {
      throw new Error('Недостаточно данных для создания ссылки-приглашения.');
    }

    const createdAt = new Date().toISOString();
    const token = crypto.randomBytes(24).toString('base64url');
    db.prepare(`
      INSERT INTO chat_share_links(token, owner_user_id, chat_id, chat_title, created_at, revoked_at)
      VALUES (?, ?, ?, ?, ?, NULL)
    `).run(
      token,
      normalizedOwnerUserId,
      normalizedChatId,
      cleanText(chatTitle || '') || 'Чат',
      createdAt
    );

    return {
      token,
      ownerUserId: normalizedOwnerUserId,
      chatId: normalizedChatId,
      chatTitle: cleanText(chatTitle || '') || 'Чат',
      createdAt,
    };
  }

  function getShare(token) {
    const normalizedToken = normalizeToken(token);
    if (!normalizedToken) {
      return null;
    }

    return db.prepare(`
      SELECT token, owner_user_id, chat_id, chat_title, created_at
      FROM chat_share_links
      WHERE token = ? AND revoked_at IS NULL
      LIMIT 1
    `).get(normalizedToken) || null;
  }

  return {
    createShare,
    getShare,
  };
}

module.exports = {
  createChatShareService,
};
