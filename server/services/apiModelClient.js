const { cleanText } = require('../lib/text');

function normalizeApiHeaders(headers = {}) {
  if (!headers || typeof headers !== 'object' || Array.isArray(headers)) {
    return {};
  }

  return Object.entries(headers).reduce((accumulator, [rawKey, rawValue]) => {
    const key = cleanText(String(rawKey || ''));
    if (!key) {
      return accumulator;
    }

    const value = typeof rawValue === 'string'
      ? rawValue.trim()
      : (rawValue === null || rawValue === undefined ? '' : String(rawValue));

    if (!value) {
      return accumulator;
    }

    accumulator[key] = value;
    return accumulator;
  }, {});
}

function normalizeApiModelConfig(input = {}) {
  const source = input && typeof input === 'object' && !Array.isArray(input)
    ? input
    : {};
  const timeoutMs = Math.max(5_000, Number(source.timeoutMs) || 60_000);

  return {
    provider: cleanText(source.provider || ''),
    endpoint: cleanText(source.endpoint || ''),
    model: cleanText(source.model || ''),
    apiKey: typeof source.apiKey === 'string' ? source.apiKey.trim() : '',
    headers: normalizeApiHeaders(source.headers),
    timeoutMs,
  };
}

function assertApiModelConfig(apiConfig) {
  if (!apiConfig.endpoint) {
    throw new Error('Для API-модели нужен URL endpoint.');
  }

  if (!/^https?:\/\//i.test(apiConfig.endpoint)) {
    throw new Error('Endpoint API-модели должен начинаться с http:// или https://.');
  }

  if (!apiConfig.model) {
    throw new Error('Для API-модели нужен идентификатор модели.');
  }

  if (!apiConfig.apiKey) {
    throw new Error('Для API-модели нужен API key.');
  }
}

function redactApiModelConfig(input = {}) {
  const apiConfig = normalizeApiModelConfig(input);
  return {
    provider: apiConfig.provider,
    endpoint: apiConfig.endpoint,
    model: apiConfig.model,
    hasKey: Boolean(apiConfig.apiKey),
    timeoutMs: apiConfig.timeoutMs,
    headerCount: Object.keys(apiConfig.headers || {}).length,
  };
}

function extractTextFromApiPayload(payload = {}) {
  if (typeof payload.output_text === 'string' && payload.output_text.trim()) {
    return payload.output_text.trim();
  }

  if (Array.isArray(payload.choices)) {
    for (const choice of payload.choices) {
      const messageContent = choice?.message?.content;
      if (typeof messageContent === 'string' && messageContent.trim()) {
        return messageContent.trim();
      }

      if (Array.isArray(messageContent)) {
        const parts = messageContent
          .map((item) => {
            if (typeof item === 'string') {
              return item;
            }
            if (typeof item?.text === 'string') {
              return item.text;
            }
            if (typeof item?.content === 'string') {
              return item.content;
            }
            return '';
          })
          .filter(Boolean)
          .join('\n')
          .trim();

        if (parts) {
          return parts;
        }
      }

      if (typeof choice?.text === 'string' && choice.text.trim()) {
        return choice.text.trim();
      }
    }
  }

  if (Array.isArray(payload.content)) {
    const content = payload.content
      .map((item) => {
        if (typeof item === 'string') {
          return item;
        }
        if (typeof item?.text === 'string') {
          return item.text;
        }
        return '';
      })
      .filter(Boolean)
      .join('\n')
      .trim();

    if (content) {
      return content;
    }
  }

  return '';
}

function extractApiErrorMessage(payload, response) {
  const payloadMessage = cleanText(
    payload?.error?.message ||
    payload?.message ||
    payload?.detail ||
    ''
  );

  if (payloadMessage) {
    return payloadMessage;
  }

  return `API-модель вернула ошибку ${response.status}${response.statusText ? ` ${response.statusText}` : ''}.`;
}

async function generateTextWithApiModel({ apiConfig, promptText, settings, systemPrompt = '', signal }) {
  const normalizedConfig = normalizeApiModelConfig(apiConfig);
  assertApiModelConfig(normalizedConfig);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), normalizedConfig.timeoutMs);
  const abortFromCaller = () => controller.abort();

  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener('abort', abortFromCaller, { once: true });
    }
  }

  try {
    const headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${normalizedConfig.apiKey}`,
      ...normalizedConfig.headers,
    };

    const response = await fetch(normalizedConfig.endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: normalizedConfig.model,
        stream: false,
        temperature: Number(settings?.generation?.responseTemperature) || 0.4,
        max_tokens: Math.max(16, Number(settings?.generation?.maxGeneratedTokens) || 256),
        messages: [
          ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
          { role: 'user', content: promptText },
        ],
      }),
      signal: controller.signal,
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(extractApiErrorMessage(payload, response));
    }

    const text = extractTextFromApiPayload(payload);
    if (!text) {
      throw new Error('API-модель не прислала текст в ожидаемом формате.');
    }

    return {
      text,
      usage: payload?.usage || null,
      provider: normalizedConfig.provider || 'openai-compatible',
      model: normalizedConfig.model,
      finishReason: cleanText(payload?.choices?.[0]?.finish_reason || payload?.finish_reason || ''),
    };
  } catch (error) {
    if (error?.name === 'AbortError') {
      if (signal?.aborted) {
        const abortError = new Error('Генерация ответа остановлена пользователем.');
        abortError.name = 'AbortError';
        throw abortError;
      }

      throw new Error(`API-модель не ответила за ${Math.round(normalizedConfig.timeoutMs / 1000)} сек.`);
    }

    throw error;
  } finally {
    clearTimeout(timeoutId);
    signal?.removeEventListener?.('abort', abortFromCaller);
  }
}

module.exports = {
  assertApiModelConfig,
  generateTextWithApiModel,
  normalizeApiModelConfig,
  redactApiModelConfig,
};
