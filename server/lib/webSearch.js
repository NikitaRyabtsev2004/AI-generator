const cheerio = require('cheerio');
const { fetchPageContent } = require('./content');
const { cleanText } = require('./text');

function decodeSearchResultUrl(rawUrl) {
  const value = String(rawUrl || '').trim();
  if (!value) {
    return '';
  }

  try {
    const parsed = new URL(value, 'https://duckduckgo.com');
    const uddg = parsed.searchParams.get('uddg');
    if (uddg) {
      return decodeURIComponent(uddg);
    }
    if (/^https?:\/\//iu.test(parsed.toString())) {
      return parsed.toString();
    }
  } catch (_error) {
    // Ignore invalid URL parsing and fall through.
  }

  return value;
}

function filterPreferredDomains(results, preferredDomainsRaw) {
  const preferredDomains = String(preferredDomainsRaw || '')
    .split(',')
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);

  if (!preferredDomains.length) {
    return results;
  }

  const preferred = [];
  const other = [];

  results.forEach((result) => {
    const host = String(result.host || '').toLowerCase();
    if (preferredDomains.some((domain) => host === domain || host.endsWith(`.${domain}`))) {
      preferred.push(result);
    } else {
      other.push(result);
    }
  });

  return [...preferred, ...other];
}

async function searchWeb(query, options = {}) {
  const timeoutMs = Math.max(2000, Number(options.timeoutMs) || 12000);
  const maxResults = Math.max(1, Number(options.maxResults) || 5);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(
      `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`,
      {
        headers: {
          'User-Agent': 'AI-Generator/1.0',
          Accept: 'text/html,application/xhtml+xml',
        },
        signal: controller.signal,
      }
    );

    if (!response.ok) {
      throw new Error(`Web search failed with HTTP ${response.status}.`);
    }

    const html = await response.text();
    const $ = cheerio.load(html);
    const results = [];

    $('a.result__a').each((_, element) => {
      const anchor = $(element);
      const title = cleanText(anchor.text());
      const href = decodeSearchResultUrl(anchor.attr('href'));
      if (!title || !href) {
        return;
      }

      let host = '';
      try {
        host = new URL(href).host;
      } catch (_error) {
        host = '';
      }

      const container = anchor.closest('.result');
      const snippet = cleanText(container.find('.result__snippet').first().text());

      results.push({
        title,
        url: href,
        host,
        snippet,
      });
    });

    return filterPreferredDomains(results, options.preferredDomains).slice(0, maxResults);
  } finally {
    clearTimeout(timeoutId);
  }
}

async function fetchSearchDocuments(query, options = {}) {
  const searchResults = await searchWeb(query, options);
  const fetchCount = Math.max(
    1,
    Math.min(
      searchResults.length,
      Number(options.fetchPages) || Math.min(2, searchResults.length)
    )
  );

  const fetchedDocuments = [];
  for (let index = 0; index < fetchCount; index += 1) {
    const result = searchResults[index];
    try {
      const page = await fetchPageContent(result.url);
      fetchedDocuments.push({
        ...result,
        content: page.content,
      });
    } catch (_error) {
      fetchedDocuments.push({
        ...result,
        content: cleanText(result.snippet || ''),
      });
    }
  }

  return fetchedDocuments.filter((entry) => cleanText(entry.content));
}

module.exports = {
  fetchSearchDocuments,
  searchWeb,
};
