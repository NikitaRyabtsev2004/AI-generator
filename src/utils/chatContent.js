const CODE_BLOCK_PATTERN = /```([\w#+.-]*)\n([\s\S]*?)```/gu;
const WORD_PATTERN = /[\p{L}\p{N}_$]+/u;
const STRUCTURED_TEXT_FIELDS = new Set([
  'text',
  'comment',
  'description',
  'explanation',
  'summary',
  'content',
]);
const STRUCTURED_CODE_FIELDS = new Set([
  'code',
  'snippet',
  'example',
  'new_contents',
  'old_contents',
]);
const STRUCTURED_LANGUAGE_FIELDS = new Set(['language', 'lang', 'syntax']);
const STRUCTURED_FIELD_PATTERN =
  /["']?(text|comment|description|explanation|summary|content|code|snippet|example|new_contents|old_contents|language|lang|syntax)["']?\s*:\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/giu;

const LANGUAGE_KEYWORDS = {
  javascript: new Set([
    'async', 'await', 'break', 'case', 'catch', 'class', 'const', 'continue', 'default',
    'delete', 'else', 'export', 'extends', 'finally', 'for', 'function', 'if', 'import',
    'in', 'instanceof', 'let', 'new', 'return', 'super', 'switch', 'this', 'throw', 'try',
    'typeof', 'var', 'void', 'while', 'yield',
  ]),
  typescript: new Set([
    'abstract', 'any', 'as', 'async', 'await', 'boolean', 'break', 'case', 'catch', 'class',
    'const', 'continue', 'declare', 'default', 'delete', 'else', 'enum', 'export', 'extends',
    'finally', 'for', 'from', 'function', 'if', 'implements', 'import', 'in', 'infer',
    'instanceof', 'interface', 'keyof', 'let', 'module', 'namespace', 'never', 'new', 'null',
    'number', 'private', 'protected', 'public', 'readonly', 'return', 'satisfies', 'static',
    'string', 'super', 'switch', 'this', 'throw', 'try', 'type', 'typeof', 'undefined',
    'unknown', 'var', 'void', 'while',
  ]),
  python: new Set([
    'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif',
    'else', 'except', 'False', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'None', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'True', 'try', 'while',
    'with', 'yield',
  ]),
  json: new Set(['true', 'false', 'null']),
  html: new Set(['doctype']),
  css: new Set([
    '@media', '@keyframes', 'display', 'position', 'absolute', 'relative', 'fixed', 'grid', 'flex',
    'color', 'background', 'border', 'padding', 'margin', 'font', 'width', 'height',
  ]),
  bash: new Set([
    'if', 'then', 'else', 'fi', 'for', 'do', 'done', 'case', 'esac', 'function', 'in', 'while',
    'until', 'select',
  ]),
  sql: new Set([
    'select', 'from', 'where', 'insert', 'into', 'values', 'update', 'delete', 'join', 'left',
    'right', 'inner', 'outer', 'group', 'by', 'order', 'having', 'limit', 'create', 'table',
    'alter', 'drop', 'as', 'and', 'or', 'on', 'set',
  ]),
};

function normalizeLanguage(language = '') {
  const value = String(language || '').trim().toLowerCase();
  if (value === 'js') {
    return 'javascript';
  }
  if (value === 'ts') {
    return 'typescript';
  }
  if (value === 'py') {
    return 'python';
  }
  if (value === 'sh' || value === 'shell') {
    return 'bash';
  }
  return value || 'text';
}

function renderMarkdownCodeBlock(language, code) {
  const normalizedCode = String(code || '').replace(/\r\n/g, '\n').trim();
  if (!normalizedCode) {
    return '';
  }

  return `\`\`\`${normalizeLanguage(language)}\n${normalizedCode}\n\`\`\``;
}

function decodeStructuredValue(value = '') {
  const normalized = String(value || '').trim();
  if (!normalized) {
    return '';
  }

  if (normalized.startsWith('"') && normalized.endsWith('"')) {
    try {
      return JSON.parse(normalized);
    } catch (_error) {
      // Fallback handled below.
    }
  }

  let body = normalized;
  if (
    (normalized.startsWith('"') && normalized.endsWith('"')) ||
    (normalized.startsWith('\'') && normalized.endsWith('\''))
  ) {
    body = normalized.slice(1, -1);
  }

  return body
    .replace(/\\\\/gu, '\\')
    .replace(/\\n/gu, '\n')
    .replace(/\\r/gu, '')
    .replace(/\\t/gu, '  ')
    .replace(/\\"/gu, '"')
    .replace(/\\'/gu, '\'')
    .trim();
}

function extractTopLevelJsonFragments(content = '') {
  const value = String(content || '').trim();
  if (!value) {
    return [];
  }

  const fragments = [];
  let startIndex = -1;
  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let index = 0; index < value.length; index += 1) {
    const symbol = value[index];

    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (symbol === '\\') {
        escaped = true;
        continue;
      }
      if (symbol === '"') {
        inString = false;
      }
      continue;
    }

    if (symbol === '"') {
      inString = true;
      continue;
    }

    if (symbol === '{' || symbol === '[') {
      if (depth === 0) {
        startIndex = index;
      }
      depth += 1;
      continue;
    }

    if (symbol === '}' || symbol === ']') {
      depth = Math.max(0, depth - 1);
      if (depth === 0 && startIndex >= 0) {
        fragments.push(value.slice(startIndex, index + 1));
        startIndex = -1;
      }
    }
  }

  return fragments;
}

function collectStructuredObjectParts(payload, target = []) {
  if (Array.isArray(payload)) {
    payload.forEach((entry) => collectStructuredObjectParts(entry, target));
    return target;
  }

  if (!payload || typeof payload !== 'object') {
    return target;
  }

  const title = String(payload.title || payload.heading || payload.name || '').trim();
  if (title) {
    target.push({
      type: 'text',
      content: title,
    });
  }

  const textContent = String(
    payload.text ||
    payload.comment ||
    payload.description ||
    payload.explanation ||
    payload.summary ||
    payload.content ||
    ''
  ).trim();
  if (textContent) {
    target.push({
      type: 'text',
      content: textContent,
    });
  }

  const codeContent = String(payload.code || payload.snippet || payload.example || '')
    .replace(/\r\n/g, '\n')
    .trim();
  if (codeContent) {
    target.push({
      type: 'code',
      language: detectCodeLanguage(codeContent, payload.language || payload.lang || payload.syntax || ''),
      content: codeContent,
    });
  }

  return target;
}

function collectLooseStructuredParts(raw, target = []) {
  let pendingLanguage = '';
  let match = STRUCTURED_FIELD_PATTERN.exec(raw);
  while (match) {
    const fieldName = String(match[1] || '').trim().toLowerCase();
    const value = decodeStructuredValue(match[2] || '');

    if (STRUCTURED_LANGUAGE_FIELDS.has(fieldName)) {
      pendingLanguage = normalizeLanguage(value);
    } else if (STRUCTURED_CODE_FIELDS.has(fieldName)) {
      const codeContent = String(value || '').replace(/\r\n/g, '\n').trim();
      if (codeContent) {
        target.push({
          type: 'code',
          language: detectCodeLanguage(codeContent, pendingLanguage),
          content: codeContent,
        });
      }
      pendingLanguage = '';
    } else if (STRUCTURED_TEXT_FIELDS.has(fieldName)) {
      const textContent = String(value || '').trim();
      if (textContent) {
        target.push({
          type: 'text',
          content: textContent,
        });
      }
    }

    match = STRUCTURED_FIELD_PATTERN.exec(raw);
  }
  STRUCTURED_FIELD_PATTERN.lastIndex = 0;
  return target;
}

function normalizeStructuredPayloadToMarkdown(content = '') {
  const raw = String(content || '').trim();
  if (!raw || /```/u.test(raw)) {
    return raw;
  }

  if (!/["']?(text|code|snippet|example|language|lang|new_contents|old_contents)["']?\s*:/iu.test(raw)) {
    return raw;
  }

  const parts = [];
  const candidates = [];
  if (/^[[{]/u.test(raw)) {
    candidates.push(raw);
  }
  candidates.push(...extractTopLevelJsonFragments(raw));

  candidates.forEach((candidate) => {
    try {
      const parsed = JSON.parse(candidate);
      collectStructuredObjectParts(parsed, parts);
    } catch (_error) {
      // Ignore malformed JSON candidates.
    }
  });

  if (!parts.length) {
    collectLooseStructuredParts(raw, parts);
  }

  if (!parts.length) {
    return raw;
  }

  const seen = new Set();
  return parts
    .map((part) => (
      part.type === 'code'
        ? renderMarkdownCodeBlock(part.language, part.content)
        : String(part.content || '').trim()
    ))
    .filter((part) => {
      if (!part) {
        return false;
      }
      const key = `${part.length}:${part.slice(0, 120)}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    })
    .join('\n\n')
    .trim();
}

function isLikelyStandaloneCode(content = '') {
  const value = String(content || '').trim();
  if (!value) {
    return false;
  }

  const lines = value.split('\n').map((line) => line.trim()).filter(Boolean);
  if (!lines.length || lines.length > 40) {
    return false;
  }

  const codeSignals = lines.filter((line) => (
    /^(const|let|var|function|class|if|for|while|switch|case|return|import|export|def|print\(|SELECT|INSERT|UPDATE|DELETE|console\.log)/iu.test(line) ||
    /[[\]{}();=<>]/u.test(line)
  )).length;

  const sentenceSignals = lines.filter((line) => /[.!?]$/u.test(line)).length;
  return codeSignals >= Math.max(2, Math.ceil(lines.length * 0.45)) && sentenceSignals <= 1;
}

export function detectCodeLanguage(code = '', hint = '') {
  const normalizedHint = normalizeLanguage(hint);
  if (normalizedHint !== 'text') {
    return normalizedHint;
  }

  const value = String(code || '');
  if (!value.trim()) {
    return 'text';
  }

  if (/^\s*(?:\{|\[)/u.test(value) && /"\s*:\s*/u.test(value)) {
    return 'json';
  }
  if (/<[a-z][\s\S]*>/iu.test(value)) {
    return 'html';
  }
  if (/\b(function|const|let|var|=>|document\.|console\.)\b/u.test(value)) {
    return 'javascript';
  }
  if (/\b(def |import |print\(|self\b|None\b|True\b|False\b)\b/u.test(value)) {
    return 'python';
  }
  if (/\b(SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|JOIN)\b/i.test(value)) {
    return 'sql';
  }
  if (/\b(display\s*:|position\s*:|color\s*:|background\s*:)\b/i.test(value)) {
    return 'css';
  }
  if (/^\s*(npm|yarn|pnpm|git|cd|ls|mkdir|curl|wget)\b/mu.test(value)) {
    return 'bash';
  }

  return 'text';
}

export function parseMessageContent(content = '') {
  const normalized = normalizeStructuredPayloadToMarkdown(content);
  const value = String(normalized || '').replace(/\r\n/g, '\n').trim();
  if (!value) {
    return [];
  }

  if (isLikelyStandaloneCode(value)) {
    return [{
      type: 'code',
      language: detectCodeLanguage(value),
      content: value,
    }];
  }

  const parts = [];
  let lastIndex = 0;
  CODE_BLOCK_PATTERN.lastIndex = 0;
  let match = CODE_BLOCK_PATTERN.exec(value);

  while (match) {
    if (match.index > lastIndex) {
      const textBlock = value.slice(lastIndex, match.index).trim();
      if (textBlock) {
        parts.push({
          type: 'text',
          content: textBlock,
        });
      }
    }

    parts.push({
      type: 'code',
      language: detectCodeLanguage(match[2], match[1]),
      content: String(match[2] || '').replace(/\n+$/u, ''),
    });

    lastIndex = CODE_BLOCK_PATTERN.lastIndex;
    match = CODE_BLOCK_PATTERN.exec(value);
  }

  if (lastIndex < value.length) {
    const textBlock = value.slice(lastIndex).trim();
    if (textBlock) {
      parts.push({
        type: 'text',
        content: textBlock,
      });
    }
  }

  return parts.length ? parts : [{ type: 'text', content: value }];
}

function buildTokenPatterns(language) {
  const normalizedLanguage = normalizeLanguage(language);
  const keywordSet = LANGUAGE_KEYWORDS[normalizedLanguage] || new Set();
  const commentPattern = /^(\/\/.*|\/\*.*?\*\/|#.*)/u;
  const stringPattern = /^("(?:\\.|[^"])*"|'(?:\\.|[^'])*'|`(?:\\.|[^`])*`)/u;
  const numberPattern = /^-?\b\d+(?:\.\d+)?\b/u;
  const functionPattern = /^([A-Za-z_$][\w$]*)(?=\s*\()/u;
  const constantPattern = /^([A-Z][A-Z0-9_]{2,})\b/u;
  const punctuationPattern = /^([{}()[\].,;:+\-/*%=<>!&|^~?]+)/u;
  const tagPattern = /^(<\/?[A-Za-z][^>\s/]*)/u;
  const attributePattern = /^([:@A-Za-z-]+)(?==)/u;

  return {
    normalizedLanguage,
    keywordSet,
    commentPattern,
    stringPattern,
    numberPattern,
    functionPattern,
    constantPattern,
    punctuationPattern,
    tagPattern,
    attributePattern,
  };
}

function tokenizeCodeLine(line, language) {
  const patterns = buildTokenPatterns(language);
  const tokens = [];
  let rest = line;

  while (rest.length > 0) {
    const whitespaceMatch = rest.match(/^\s+/u);
    if (whitespaceMatch) {
      tokens.push({ type: 'plain', text: whitespaceMatch[0] });
      rest = rest.slice(whitespaceMatch[0].length);
      continue;
    }

    const commentMatch = rest.match(patterns.commentPattern);
    if (commentMatch) {
      tokens.push({ type: 'comment', text: commentMatch[0] });
      break;
    }

    const stringMatch = rest.match(patterns.stringPattern);
    if (stringMatch) {
      tokens.push({ type: 'string', text: stringMatch[0] });
      rest = rest.slice(stringMatch[0].length);
      continue;
    }

    if (patterns.normalizedLanguage === 'html') {
      const tagMatch = rest.match(patterns.tagPattern);
      if (tagMatch) {
        tokens.push({ type: 'keyword', text: tagMatch[0] });
        rest = rest.slice(tagMatch[0].length);
        continue;
      }

      const attributeMatch = rest.match(patterns.attributePattern);
      if (attributeMatch) {
        tokens.push({ type: 'property', text: attributeMatch[0] });
        rest = rest.slice(attributeMatch[0].length);
        continue;
      }
    }

    const numberMatch = rest.match(patterns.numberPattern);
    if (numberMatch) {
      tokens.push({ type: 'number', text: numberMatch[0] });
      rest = rest.slice(numberMatch[0].length);
      continue;
    }

    const functionMatch = rest.match(patterns.functionPattern);
    if (functionMatch) {
      tokens.push({ type: 'function', text: functionMatch[0] });
      rest = rest.slice(functionMatch[0].length);
      continue;
    }

    const constantMatch = rest.match(patterns.constantPattern);
    if (constantMatch) {
      tokens.push({ type: 'constant', text: constantMatch[0] });
      rest = rest.slice(constantMatch[0].length);
      continue;
    }

    const wordMatch = rest.match(WORD_PATTERN);
    if (wordMatch && rest.startsWith(wordMatch[0])) {
      const word = wordMatch[0];
      if (
        patterns.keywordSet.has(word) ||
        patterns.keywordSet.has(word.toLowerCase()) ||
        patterns.keywordSet.has(word.toUpperCase())
      ) {
        tokens.push({ type: 'keyword', text: word });
      } else if (word === 'true' || word === 'false' || word === 'null' || word === 'None') {
        tokens.push({ type: 'value', text: word });
      } else {
        tokens.push({ type: 'plain', text: word });
      }
      rest = rest.slice(word.length);
      continue;
    }

    const punctuationMatch = rest.match(patterns.punctuationPattern);
    if (punctuationMatch) {
      tokens.push({ type: 'operator', text: punctuationMatch[0] });
      rest = rest.slice(punctuationMatch[0].length);
      continue;
    }

    tokens.push({ type: 'plain', text: rest[0] });
    rest = rest.slice(1);
  }

  return tokens;
}

export function highlightCode(code = '', language = 'text') {
  return String(code || '')
    .replace(/\r\n/g, '\n')
    .split('\n')
    .map((line) => tokenizeCodeLine(line, language));
}

export function splitTextParagraphs(text = '') {
  return String(text || '')
    .split(/\n{2,}/u)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
}
