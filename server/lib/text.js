const crypto = require('crypto');
const iconv = require('iconv-lite');

function createId(prefix = 'id') {
  return `${prefix}-${crypto.randomUUID()}`;
}

function normalizeWhitespace(input = '') {
  return String(input)
    .replace(/\r\n/g, '\n')
    .replace(/\t/g, ' ')
    .replace(/[ ]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function stripControlChars(input = '') {
  return String(input)
    .split('\u0000')
    .join('')
    .replace(/[\u0001-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '');
}

function normalizeTextShape(input = '') {
  return normalizeWhitespace(stripControlChars(input));
}

function countMatches(input, pattern) {
  return (input.match(pattern) || []).length;
}

function scoreTextQuality(text = '') {
  const readable = countMatches(text, /[A-Za-zА-Яа-яЁё0-9]/g);
  const cyrillic = countMatches(text, /[А-Яа-яЁё]/g);
  const mojibakeClusters = countMatches(
    text,
    /(?:[РС][^\s]){2,}|(?:[ÐÑ][^\s]){2,}/g
  );
  const replacements = countMatches(text, /�/g);
  const weirdControls = countMatches(text, /[\u0080-\u009F]/g);

  return readable + cyrillic * 2 - mojibakeClusters * 8 - replacements * 40 - weirdControls * 10;
}

function repairMojibakeText(input = '') {
  const normalized = normalizeTextShape(input);
  if (!normalized) {
    return '';
  }

  const decodeUtf8FromWin1251Mojibake = (value) => {
    const bytes = [];
    for (const character of value) {
      const codePoint = character.charCodeAt(0);
      if (codePoint <= 0xFF) {
        bytes.push(codePoint);
        continue;
      }

      const encoded = iconv.encode(character, 'win1251');
      if (!encoded.length) {
        return value;
      }
      bytes.push(...encoded);
    }

    return normalizeTextShape(Buffer.from(bytes).toString('utf8'));
  };

  const candidates = [normalized];

  try {
    candidates.push(
      normalizeTextShape(iconv.decode(iconv.encode(normalized, 'win1251'), 'utf8'))
    );
  } catch (error) {
    // Ignore conversion failures and keep the original text.
  }

  try {
    candidates.push(decodeUtf8FromWin1251Mojibake(normalized));
  } catch (error) {
    // Ignore conversion failures and keep the original text.
  }

  try {
    candidates.push(normalizeTextShape(Buffer.from(normalized, 'latin1').toString('utf8')));
  } catch (error) {
    // Ignore conversion failures and keep the original text.
  }

  let bestCandidate = normalized;
  let bestScore = scoreTextQuality(normalized);

  candidates.forEach((candidate) => {
    const candidateScore = scoreTextQuality(candidate);
    if (candidateScore > bestScore + 2) {
      bestCandidate = candidate;
      bestScore = candidateScore;
    }
  });

  return bestCandidate;
}

function cleanText(input = '') {
  return repairMojibakeText(input);
}

function tokenizeWords(input = '') {
  const matches = cleanText(input)
    .toLowerCase()
    .match(/[\p{L}\p{N}]+/gu);

  return matches ? matches.filter(Boolean) : [];
}

function splitIntoSentences(input = '') {
  return cleanText(input)
    .split(/(?<=[.!?])\s+|\n+/g)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function previewText(input = '', maxLength = 220) {
  const value = cleanText(input);
  if (value.length <= maxLength) {
    return value;
  }

  return `${value.slice(0, maxLength).trimEnd()}...`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function computeStats(input = '') {
  const text = cleanText(input);
  const tokens = tokenizeWords(text);
  const sentences = splitIntoSentences(text);

  return {
    charCount: text.length,
    tokenCount: tokens.length,
    sentenceCount: sentences.length,
  };
}

function inferChatTitle(messages = []) {
  const firstUserMessage = messages.find((message) => message.role === 'user');
  if (!firstUserMessage) {
    return 'Новый чат';
  }

  return previewText(firstUserMessage.content, 42);
}

function scoreDecodedText(text) {
  const readableLetters = (text.match(/[A-Za-zА-Яа-яЁё0-9]/g) || []).length;
  const replacements = (text.match(/�/g) || []).length;
  const questions = (text.match(/\?/g) || []).length;
  const exoticLetters = (text.match(/[^\x00-\x7FА-Яа-яЁё]/g) || []).length;
  return readableLetters * 4 - exoticLetters * 6 - replacements * 30 - questions * 3;
}

function decodeBufferToText(buffer) {
  const candidates = [
    iconv.decode(buffer, 'utf8'),
    iconv.decode(buffer, 'utf16le'),
    iconv.decode(buffer, 'win1251'),
  ]
    .map((text) => cleanText(text))
    .filter(Boolean);

  if (!candidates.length) {
    return '';
  }

  return candidates.sort((left, right) => scoreDecodedText(right) - scoreDecodedText(left))[0];
}

module.exports = {
  clamp,
  cleanText,
  computeStats,
  createId,
  decodeBufferToText,
  inferChatTitle,
  normalizeWhitespace,
  previewText,
  repairMojibakeText,
  splitIntoSentences,
  tokenizeWords,
};
