const crypto = require('crypto');
const Database = require('better-sqlite3');
const nodemailer = require('nodemailer');
const { cleanText, createId } = require('../lib/text');

const SESSION_COOKIE_NAME = 'ag_session';
const DEFAULT_SESSION_TTL_DAYS = Math.max(1, Number(process.env.AUTH_SESSION_TTL_DAYS) || 14);
const DEFAULT_CODE_TTL_MINUTES = Math.max(3, Number(process.env.AUTH_CODE_TTL_MINUTES) || 15);
const MAX_CODE_ATTEMPTS = Math.max(3, Number(process.env.AUTH_MAX_CODE_ATTEMPTS) || 5);
const MAX_LOGIN_ATTEMPTS = Math.max(3, Number(process.env.AUTH_MAX_LOGIN_ATTEMPTS) || 8);
const LOGIN_WINDOW_MS = Math.max(60_000, Number(process.env.AUTH_LOGIN_WINDOW_MS) || 15 * 60 * 1000);
const LOCKOUT_MS = Math.max(60_000, Number(process.env.AUTH_LOCKOUT_MS) || 15 * 60 * 1000);
const PASSWORD_PEPPER = String(process.env.AUTH_PASSWORD_PEPPER || '');

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/u;
const PASSWORD_ALLOWED_PATTERN = /^[A-Za-z0-9!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]{12,}$/u;

function createHttpError(message, statusCode = 400) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

function nowIso() {
  return new Date().toISOString();
}

function normalizeEmail(email) {
  return cleanText(String(email || '')).toLowerCase();
}

function assertValidEmail(email) {
  if (!EMAIL_PATTERN.test(email)) {
    throw createHttpError('Некорректный email.', 400);
  }
}

function assertStrongPassword(password) {
  const value = String(password || '');
  if (!PASSWORD_ALLOWED_PATTERN.test(value)) {
    throw createHttpError(
      'Пароль должен быть не короче 12 символов, содержать только английские буквы, цифры и спецсимволы.',
      400
    );
  }
  if (!/[A-Z]/u.test(value)) {
    throw createHttpError('Пароль должен содержать хотя бы одну заглавную букву.', 400);
  }
  if (!/[a-z]/u.test(value)) {
    throw createHttpError('Пароль должен содержать хотя бы одну строчную букву.', 400);
  }
  if (!/\d/u.test(value)) {
    throw createHttpError('Пароль должен содержать хотя бы одну цифру.', 400);
  }
  if (!/[^A-Za-z0-9]/u.test(value)) {
    throw createHttpError('Пароль должен содержать хотя бы один спецсимвол.', 400);
  }
}

function hashCode(code) {
  return crypto.createHash('sha256').update(String(code || ''), 'utf8').digest('hex');
}

function randomCode(length = 6) {
  const maxValue = 10 ** length;
  const value = crypto.randomInt(0, maxValue);
  return value.toString().padStart(length, '0');
}

function hashPassword(password) {
  const salt = crypto.randomBytes(16);
  const derived = crypto.scryptSync(`${password}${PASSWORD_PEPPER}`, salt, 64);
  return `scrypt$${salt.toString('base64')}$${derived.toString('base64')}`;
}

function verifyPassword(password, storedHash) {
  const normalized = String(storedHash || '');
  const parts = normalized.split('$');
  if (parts.length !== 3 || parts[0] !== 'scrypt') {
    return false;
  }

  try {
    const salt = Buffer.from(parts[1], 'base64');
    const expected = Buffer.from(parts[2], 'base64');
    const actual = crypto.scryptSync(`${password}${PASSWORD_PEPPER}`, salt, expected.length);
    return actual.length === expected.length && crypto.timingSafeEqual(actual, expected);
  } catch (_error) {
    return false;
  }
}

function hashSessionToken(token) {
  return crypto.createHash('sha256').update(String(token || ''), 'utf8').digest('hex');
}

function parseCookies(request) {
  const raw = String(request?.headers?.cookie || '');
  if (!raw) {
    return {};
  }

  return raw.split(';').reduce((acc, part) => {
    const index = part.indexOf('=');
    if (index <= 0) {
      return acc;
    }
    const key = part.slice(0, index).trim();
    const value = part.slice(index + 1).trim();
    if (!key) {
      return acc;
    }
    try {
      acc[key] = decodeURIComponent(value);
    } catch (_error) {
      acc[key] = value;
    }
    return acc;
  }, {});
}

function serializeCookie(name, value, options = {}) {
  const segments = [`${name}=${encodeURIComponent(value)}`];
  if (options.maxAge !== undefined) {
    segments.push(`Max-Age=${Math.max(0, Math.floor(Number(options.maxAge) || 0))}`);
  }
  segments.push(`Path=${options.path || '/'}`);
  if (options.httpOnly !== false) {
    segments.push('HttpOnly');
  }
  if (options.sameSite) {
    segments.push(`SameSite=${options.sameSite}`);
  }
  if (options.secure) {
    segments.push('Secure');
  }
  return segments.join('; ');
}

function buildMailTransport() {
  const host = cleanText(process.env.SMTP_HOST || '');
  const port = Number(process.env.SMTP_PORT || 0);
  const secure = String(process.env.SMTP_SECURE || '').toLowerCase() === 'true' || port === 465;
  const user = cleanText(process.env.SMTP_USER || '');
  const pass = String(process.env.SMTP_PASS || '');
  const from = cleanText(process.env.SMTP_FROM || '');

  if (!host || !port || !user || !pass || !from) {
    throw createHttpError(
      'SMTP не настроен: заполните SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS и SMTP_FROM в .env.',
      500
    );
  }

  return nodemailer.createTransport({
    host,
    port,
    secure,
    auth: {
      user,
      pass,
    },
  });
}

function normalizePublicUser(userRow) {
  return {
    id: userRow.id,
    email: userRow.email,
    verified: Boolean(userRow.is_verified),
    createdAt: userRow.created_at,
    lastLoginAt: userRow.last_login_at || null,
  };
}

function createAuthService({ dbPath }) {
  const db = new Database(dbPath);
  db.pragma('journal_mode = PERSIST');
  db.pragma('synchronous = NORMAL');
  db.pragma('busy_timeout = 5000');
  db.pragma('foreign_keys = ON');

  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      email TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      is_verified INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      last_login_at TEXT
    );

    CREATE TABLE IF NOT EXISTS pending_registrations (
      email TEXT PRIMARY KEY,
      password_hash TEXT NOT NULL,
      code_hash TEXT NOT NULL,
      expires_at TEXT NOT NULL,
      attempts INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS pending_password_resets (
      email TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      code_hash TEXT NOT NULL,
      expires_at TEXT NOT NULL,
      attempts INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS auth_sessions (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      token_hash TEXT NOT NULL UNIQUE,
      created_at TEXT NOT NULL,
      expires_at TEXT NOT NULL,
      last_seen_at TEXT,
      revoked_at TEXT,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
  `);

  const loginFailures = new Map();
  let mailTransport = null;

  const readUserByEmail = db.prepare(`
    SELECT id, email, password_hash, is_verified, created_at, updated_at, last_login_at
    FROM users
    WHERE email = ?
    LIMIT 1
  `);

  const readPendingRegistration = db.prepare(`
    SELECT email, password_hash, code_hash, expires_at, attempts
    FROM pending_registrations
    WHERE email = ?
    LIMIT 1
  `);

  const readPendingPasswordReset = db.prepare(`
    SELECT email, user_id, code_hash, expires_at, attempts
    FROM pending_password_resets
    WHERE email = ?
    LIMIT 1
  `);

  function getTransport() {
    if (!mailTransport) {
      mailTransport = buildMailTransport();
    }
    return mailTransport;
  }

  async function sendCodeEmail({ email, code, type }) {
    const transport = getTransport();
    const from = cleanText(process.env.SMTP_FROM || '');
    const subject = type === 'reset'
      ? 'AI Generator Studio | код сброса пароля'
      : 'AI Generator Studio | код подтверждения регистрации';
    const actionText = type === 'reset'
      ? 'сброса пароля'
      : 'подтверждения регистрации';

    await transport.sendMail({
      from,
      to: email,
      subject,
      text:
        `Код для ${actionText}: ${code}\n\n` +
        `Код действует ${DEFAULT_CODE_TTL_MINUTES} минут.\n` +
        'Если это были не вы, просто проигнорируйте письмо.',
      html:
        `<div style="font-family:Arial,sans-serif;line-height:1.5;color:#111;">` +
        `<h2 style="margin:0 0 12px;">AI Generator Studio</h2>` +
        `<p style="margin:0 0 8px;">Код для ${actionText}:</p>` +
        `<p style="font-size:24px;font-weight:700;letter-spacing:2px;margin:0 0 12px;">${code}</p>` +
        `<p style="margin:0;color:#555;">Код действует ${DEFAULT_CODE_TTL_MINUTES} минут.</p>` +
        `</div>`,
    });
  }

  function clearOldSessionCookies(response) {
    response.setHeader('Set-Cookie', serializeCookie(SESSION_COOKIE_NAME, '', {
      maxAge: 0,
      path: '/',
      sameSite: 'Lax',
      secure: process.env.NODE_ENV === 'production',
    }));
  }

  function issueSession(userId) {
    const token = crypto.randomBytes(48).toString('base64url');
    const tokenHash = hashSessionToken(token);
    const createdAt = nowIso();
    const expiresAt = new Date(Date.now() + (DEFAULT_SESSION_TTL_DAYS * 24 * 60 * 60 * 1000)).toISOString();
    db.prepare(`
      INSERT INTO auth_sessions(id, user_id, token_hash, created_at, expires_at, last_seen_at, revoked_at)
      VALUES (?, ?, ?, ?, ?, ?, NULL)
    `).run(createId('session'), userId, tokenHash, createdAt, expiresAt, createdAt);
    return {
      token,
      expiresAt,
    };
  }

  function setSessionCookie(response, token, expiresAt) {
    const maxAgeSeconds = Math.max(0, Math.floor((Date.parse(expiresAt) - Date.now()) / 1000));
    response.setHeader('Set-Cookie', serializeCookie(SESSION_COOKIE_NAME, token, {
      maxAge: maxAgeSeconds,
      path: '/',
      sameSite: 'Lax',
      secure: process.env.NODE_ENV === 'production',
      httpOnly: true,
    }));
  }

  function revokeSessionByToken(token) {
    if (!token) {
      return;
    }
    db.prepare(`
      UPDATE auth_sessions
      SET revoked_at = ?, last_seen_at = ?
      WHERE token_hash = ? AND revoked_at IS NULL
    `).run(nowIso(), nowIso(), hashSessionToken(token));
  }

  function resolveAuthFromRequest(request) {
    const cookies = parseCookies(request);
    const sessionToken = cleanText(cookies[SESSION_COOKIE_NAME] || '');
    if (!sessionToken) {
      return null;
    }

    const now = nowIso();
    const row = db.prepare(`
      SELECT
        s.id AS session_id,
        s.user_id,
        s.expires_at,
        u.email,
        u.is_verified,
        u.created_at,
        u.last_login_at
      FROM auth_sessions s
      INNER JOIN users u ON u.id = s.user_id
      WHERE s.token_hash = ?
        AND s.revoked_at IS NULL
        AND s.expires_at > ?
      LIMIT 1
    `).get(hashSessionToken(sessionToken), now);

    if (!row) {
      return null;
    }

    db.prepare(`
      UPDATE auth_sessions
      SET last_seen_at = ?
      WHERE id = ?
    `).run(now, row.session_id);

    return {
      sessionId: row.session_id,
      user: {
        id: row.user_id,
        email: row.email,
        verified: Boolean(row.is_verified),
        createdAt: row.created_at,
        lastLoginAt: row.last_login_at || null,
      },
    };
  }

  function assertLoginAllowed(email, requestIp = '') {
    const key = `${email}:${requestIp}`;
    const nowTs = Date.now();
    const entry = loginFailures.get(key);
    if (!entry) {
      return;
    }
    if (entry.lockedUntil && entry.lockedUntil > nowTs) {
      throw createHttpError('Слишком много неудачных попыток входа. Подождите немного.', 429);
    }
    if (entry.windowStart + LOGIN_WINDOW_MS < nowTs) {
      loginFailures.delete(key);
    }
  }

  function registerFailedLogin(email, requestIp = '') {
    const key = `${email}:${requestIp}`;
    const nowTs = Date.now();
    const entry = loginFailures.get(key) || {
      count: 0,
      windowStart: nowTs,
      lockedUntil: 0,
    };
    if (entry.windowStart + LOGIN_WINDOW_MS < nowTs) {
      entry.count = 0;
      entry.windowStart = nowTs;
      entry.lockedUntil = 0;
    }
    entry.count += 1;
    if (entry.count >= MAX_LOGIN_ATTEMPTS) {
      entry.lockedUntil = nowTs + LOCKOUT_MS;
    }
    loginFailures.set(key, entry);
  }

  function clearFailedLogins(email, requestIp = '') {
    loginFailures.delete(`${email}:${requestIp}`);
  }

  async function startRegistration({ email, password }) {
    const normalizedEmail = normalizeEmail(email);
    const rawPassword = String(password || '');
    assertValidEmail(normalizedEmail);
    assertStrongPassword(rawPassword);

    const existingUser = readUserByEmail.get(normalizedEmail);
    if (existingUser && Number(existingUser.is_verified) === 1) {
      throw createHttpError('Пользователь с таким email уже существует. Выполните вход.', 409);
    }

    const passwordHash = hashPassword(rawPassword);
    const code = randomCode(6);
    const codeHash = hashCode(code);
    const createdAt = nowIso();
    const expiresAt = new Date(Date.now() + DEFAULT_CODE_TTL_MINUTES * 60 * 1000).toISOString();

    db.prepare(`
      INSERT INTO pending_registrations(email, password_hash, code_hash, expires_at, attempts, created_at, updated_at)
      VALUES (?, ?, ?, ?, 0, ?, ?)
      ON CONFLICT(email) DO UPDATE SET
        password_hash = excluded.password_hash,
        code_hash = excluded.code_hash,
        expires_at = excluded.expires_at,
        attempts = 0,
        updated_at = excluded.updated_at
    `).run(normalizedEmail, passwordHash, codeHash, expiresAt, createdAt, createdAt);

    await sendCodeEmail({
      email: normalizedEmail,
      code,
      type: 'register',
    });
  }

  function verifyRegistration({ email, code }) {
    const normalizedEmail = normalizeEmail(email);
    const normalizedCode = cleanText(String(code || ''));
    assertValidEmail(normalizedEmail);
    if (!/^\d{6}$/u.test(normalizedCode)) {
      throw createHttpError('Код подтверждения должен состоять из 6 цифр.', 400);
    }

    const pending = readPendingRegistration.get(normalizedEmail);
    if (!pending) {
      throw createHttpError('Код подтверждения не найден или уже использован.', 400);
    }
    if (Date.parse(pending.expires_at) <= Date.now()) {
      db.prepare('DELETE FROM pending_registrations WHERE email = ?').run(normalizedEmail);
      throw createHttpError('Срок действия кода истек. Запросите новый код.', 400);
    }

    if (pending.code_hash !== hashCode(normalizedCode)) {
      const nextAttempts = Number(pending.attempts || 0) + 1;
      if (nextAttempts >= MAX_CODE_ATTEMPTS) {
        db.prepare('DELETE FROM pending_registrations WHERE email = ?').run(normalizedEmail);
        throw createHttpError('Код подтверждения введен слишком много раз. Запросите новый код.', 429);
      }
      db.prepare(`
        UPDATE pending_registrations
        SET attempts = ?, updated_at = ?
        WHERE email = ?
      `).run(nextAttempts, nowIso(), normalizedEmail);
      throw createHttpError('Неверный код подтверждения.', 400);
    }

    const createdAt = nowIso();
    const existingUser = readUserByEmail.get(normalizedEmail);
    let userId = existingUser?.id || createId('user');

    db.exec('BEGIN');
    try {
      if (existingUser) {
        db.prepare(`
          UPDATE users
          SET password_hash = ?, is_verified = 1, updated_at = ?
          WHERE id = ?
        `).run(pending.password_hash, createdAt, userId);
      } else {
        db.prepare(`
          INSERT INTO users(id, email, password_hash, is_verified, created_at, updated_at, last_login_at)
          VALUES (?, ?, ?, 1, ?, ?, NULL)
        `).run(userId, normalizedEmail, pending.password_hash, createdAt, createdAt);
      }
      db.prepare('DELETE FROM pending_registrations WHERE email = ?').run(normalizedEmail);
      db.exec('COMMIT');
    } catch (error) {
      try {
        db.exec('ROLLBACK');
      } catch (_rollbackError) {
        // Ignore rollback errors.
      }
      throw error;
    }

    const session = issueSession(userId);
    db.prepare(`
      UPDATE users
      SET last_login_at = ?, updated_at = ?
      WHERE id = ?
    `).run(createdAt, createdAt, userId);

    const user = readUserByEmail.get(normalizedEmail);
    return {
      user: normalizePublicUser(user),
      session,
    };
  }

  function login({ email, password, requestIp = '' }) {
    const normalizedEmail = normalizeEmail(email);
    const rawPassword = String(password || '');
    assertValidEmail(normalizedEmail);
    if (!rawPassword) {
      throw createHttpError('Введите пароль.', 400);
    }

    assertLoginAllowed(normalizedEmail, requestIp);
    const user = readUserByEmail.get(normalizedEmail);
    if (!user || !verifyPassword(rawPassword, user.password_hash)) {
      registerFailedLogin(normalizedEmail, requestIp);
      throw createHttpError('Неверный email или пароль.', 401);
    }
    if (Number(user.is_verified) !== 1) {
      throw createHttpError('Аккаунт не подтвержден. Завершите подтверждение кода из письма.', 403);
    }

    clearFailedLogins(normalizedEmail, requestIp);
    const session = issueSession(user.id);
    db.prepare(`
      UPDATE users
      SET last_login_at = ?, updated_at = ?
      WHERE id = ?
    `).run(nowIso(), nowIso(), user.id);

    return {
      user: normalizePublicUser(user),
      session,
    };
  }

  async function startPasswordReset({ email }) {
    const normalizedEmail = normalizeEmail(email);
    assertValidEmail(normalizedEmail);
    const user = readUserByEmail.get(normalizedEmail);
    if (!user || Number(user.is_verified) !== 1) {
      return;
    }

    const code = randomCode(6);
    const codeHash = hashCode(code);
    const createdAt = nowIso();
    const expiresAt = new Date(Date.now() + DEFAULT_CODE_TTL_MINUTES * 60 * 1000).toISOString();
    db.prepare(`
      INSERT INTO pending_password_resets(email, user_id, code_hash, expires_at, attempts, created_at, updated_at)
      VALUES (?, ?, ?, ?, 0, ?, ?)
      ON CONFLICT(email) DO UPDATE SET
        user_id = excluded.user_id,
        code_hash = excluded.code_hash,
        expires_at = excluded.expires_at,
        attempts = 0,
        updated_at = excluded.updated_at
    `).run(normalizedEmail, user.id, codeHash, expiresAt, createdAt, createdAt);

    await sendCodeEmail({
      email: normalizedEmail,
      code,
      type: 'reset',
    });
  }

  function confirmPasswordReset({ email, code, newPassword }) {
    const normalizedEmail = normalizeEmail(email);
    const normalizedCode = cleanText(String(code || ''));
    const rawPassword = String(newPassword || '');
    assertValidEmail(normalizedEmail);
    if (!/^\d{6}$/u.test(normalizedCode)) {
      throw createHttpError('Код подтверждения должен состоять из 6 цифр.', 400);
    }
    assertStrongPassword(rawPassword);

    const pending = readPendingPasswordReset.get(normalizedEmail);
    if (!pending) {
      throw createHttpError('Код подтверждения не найден или уже использован.', 400);
    }
    if (Date.parse(pending.expires_at) <= Date.now()) {
      db.prepare('DELETE FROM pending_password_resets WHERE email = ?').run(normalizedEmail);
      throw createHttpError('Срок действия кода истек. Запросите новый код.', 400);
    }
    if (pending.code_hash !== hashCode(normalizedCode)) {
      const nextAttempts = Number(pending.attempts || 0) + 1;
      if (nextAttempts >= MAX_CODE_ATTEMPTS) {
        db.prepare('DELETE FROM pending_password_resets WHERE email = ?').run(normalizedEmail);
        throw createHttpError('Код подтверждения введен слишком много раз. Запросите новый код.', 429);
      }
      db.prepare(`
        UPDATE pending_password_resets
        SET attempts = ?, updated_at = ?
        WHERE email = ?
      `).run(nextAttempts, nowIso(), normalizedEmail);
      throw createHttpError('Неверный код подтверждения.', 400);
    }

    const user = readUserByEmail.get(normalizedEmail);
    if (!user) {
      db.prepare('DELETE FROM pending_password_resets WHERE email = ?').run(normalizedEmail);
      throw createHttpError('Пользователь не найден.', 404);
    }

    const updatedAt = nowIso();
    const nextHash = hashPassword(rawPassword);
    db.exec('BEGIN');
    try {
      db.prepare(`
        UPDATE users
        SET password_hash = ?, updated_at = ?
        WHERE id = ?
      `).run(nextHash, updatedAt, user.id);
      db.prepare('DELETE FROM pending_password_resets WHERE email = ?').run(normalizedEmail);
      db.prepare('UPDATE auth_sessions SET revoked_at = ?, last_seen_at = ? WHERE user_id = ? AND revoked_at IS NULL')
        .run(updatedAt, updatedAt, user.id);
      db.exec('COMMIT');
    } catch (error) {
      try {
        db.exec('ROLLBACK');
      } catch (_rollbackError) {
        // Ignore rollback errors.
      }
      throw error;
    }

    const session = issueSession(user.id);
    db.prepare(`
      UPDATE users
      SET last_login_at = ?, updated_at = ?
      WHERE id = ?
    `).run(updatedAt, updatedAt, user.id);

    return {
      user: normalizePublicUser(readUserByEmail.get(normalizedEmail)),
      session,
    };
  }

  function logout(request, response) {
    const cookies = parseCookies(request);
    const token = cleanText(cookies[SESSION_COOKIE_NAME] || '');
    if (token) {
      revokeSessionByToken(token);
    }
    clearOldSessionCookies(response);
  }

  function clearSessionCookie(response) {
    clearOldSessionCookies(response);
  }

  return {
    login,
    logout,
    startRegistration,
    verifyRegistration,
    startPasswordReset,
    confirmPasswordReset,
    resolveAuthFromRequest,
    setSessionCookie,
    clearSessionCookie,
  };
}

module.exports = {
  SESSION_COOKIE_NAME,
  createAuthService,
  createHttpError,
};
