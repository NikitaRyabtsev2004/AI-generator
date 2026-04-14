const cheerio = require('cheerio');
const { fetchPageContent } = require('./content');
const { cleanText } = require('./text');

const WORD_PATTERN = /[\p{L}\p{N}]+/gu;

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
    .map((item) => {
      if (!item) {
        return '';
      }
      try {
        const host = new URL(item.includes('://') ? item : `https://${item}`).host.toLowerCase();
        return host.replace(/^www\./u, '');
      } catch (_error) {
        return item.replace(/^https?:\/\//u, '').replace(/\/.*$/u, '').replace(/^www\./u, '');
      }
    })
    .filter(Boolean);

  if (!preferredDomains.length) {
    return results;
  }

  const preferred = [];
  const other = [];

  results.forEach((result) => {
    const host = String(result.host || '').toLowerCase().replace(/^www\./u, '');
    if (preferredDomains.some((domain) => host === domain || host.endsWith(`.${domain}`))) {
      preferred.push(result);
    } else {
      other.push(result);
    }
  });

  return [...preferred, ...other];
}

function tokenizeSearchText(value = '') {
  return cleanText(value)
    .toLowerCase()
    .match(WORD_PATTERN) || [];
}

function scoreSearchResult(queryTokens, result, preferredDomainsRaw = '') {
  const titleTokens = tokenizeSearchText(result.title || '');
  const snippetTokens = tokenizeSearchText(result.snippet || '');
  const hostTokens = tokenizeSearchText(result.host || '');
  const allTokens = [...titleTokens, ...snippetTokens, ...hostTokens];
  if (!allTokens.length) {
    return 0;
  }

  const uniqueQueryTokens = Array.from(new Set(queryTokens));
  const tokenPool = new Set(allTokens);
  let score = 0;

  uniqueQueryTokens.forEach((token) => {
    if (titleTokens.includes(token)) {
      score += 2.4;
      return;
    }

    if (snippetTokens.includes(token)) {
      score += 1.35;
      return;
    }

    if (tokenPool.has(token)) {
      score += 0.6;
    }
  });

  const normalizedPreferredDomains = String(preferredDomainsRaw || '')
    .split(',')
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  const host = String(result.host || '').toLowerCase();
  if (normalizedPreferredDomains.some((domain) => host === domain || host.endsWith(`.${domain}`))) {
    score += 2.2;
  }

  if (titleTokens.length > 0) {
    score += Math.min(titleTokens.length / 24, 0.4);
  }

  return score;
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
  const queryTokens = tokenizeSearchText(query);
  const rankedSearchResults = [...searchResults]
    .map((result, index) => ({
      ...result,
      searchScore: scoreSearchResult(queryTokens, result, options.preferredDomains) + Math.max(0, (searchResults.length - index) * 0.02),
    }))
    .sort((left, right) => right.searchScore - left.searchScore);
  const fetchCount = Math.max(
    1,
    Math.min(
      rankedSearchResults.length,
      Number(options.fetchPages) || Math.min(2, rankedSearchResults.length)
    )
  );
  const selectedResults = rankedSearchResults.slice(0, fetchCount);

  const fetchedDocuments = await Promise.all(selectedResults.map(async (result) => {
    try {
      const page = await fetchPageContent(result.url);
      return {
        ...result,
        content: page.content,
      };
    } catch (_error) {
      return {
        ...result,
        content: cleanText(result.snippet || ''),
      };
    }
  }));

  return fetchedDocuments
    .filter((entry) => cleanText(entry.content))
    .sort((left, right) => (Number(right.searchScore) || 0) - (Number(left.searchScore) || 0));
}

module.exports = {
  fetchSearchDocuments,
  searchWeb,
};
