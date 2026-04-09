const cheerio = require('cheerio');
const { cleanText } = require('./text');

function normalizeUrl(input) {
  try {
    const prepared = /^https?:\/\//i.test(input) ? input : `https://${input}`;
    return new URL(prepared).toString();
  } catch (error) {
    throw new Error('Укажите корректный URL.');
  }
}

function extractTextFromHtml(html) {
  const $ = cheerio.load(html);
  $('script, style, noscript, iframe, svg, canvas, form').remove();

  const title = cleanText($('title').first().text());
  const body = cleanText($('body').text() || $.root().text());

  return cleanText([title, body].filter(Boolean).join('\n\n'));
}

function looksLikeHtml(contentType, bodyText) {
  return contentType.includes('text/html') || /^\s*</.test(bodyText);
}

async function requestContent(url, strategy) {
  const response = await fetch(strategy.build(url), {
    headers: {
      'User-Agent': 'LiquidGlassAIStudio/1.0',
      Accept: 'text/html, text/plain, */*',
    },
  });

  if (!response.ok) {
    throw new Error(`${strategy.label}: HTTP ${response.status}`);
  }

  const rawText = await response.text();
  const contentType = (response.headers.get('content-type') || '').toLowerCase();
  const parsedText = strategy.forcePlainText
    ? cleanText(rawText)
    : looksLikeHtml(contentType, rawText)
      ? extractTextFromHtml(rawText)
      : cleanText(rawText);

  if (parsedText.length < 80) {
    throw new Error(`${strategy.label}: страница вернула слишком мало текста.`);
  }

  return parsedText;
}

async function fetchPageContent(input) {
  const url = normalizeUrl(input);
  const strippedUrl = url.replace(/^https?:\/\//i, '');
  const strategies = [
    {
      label: 'direct fetch',
      build: (value) => value,
      forcePlainText: false,
    },
    {
      label: 'jina reader',
      build: () => `https://r.jina.ai/http://${strippedUrl}`,
      forcePlainText: true,
    },
    {
      label: 'allorigins',
      build: (value) => `https://api.allorigins.win/raw?url=${encodeURIComponent(value)}`,
      forcePlainText: false,
    },
  ];

  let lastError = null;

  for (const strategy of strategies) {
    try {
      return {
        url,
        content: await requestContent(url, strategy),
      };
    } catch (error) {
      lastError = error;
    }
  }

  throw new Error(
    lastError?.message ||
      'Не удалось получить содержимое страницы. Возможно, сайт блокирует автоматическое чтение.'
  );
}

module.exports = {
  fetchPageContent,
  normalizeUrl,
};
