const multer = require('multer');
const path = require('path');
const { cleanText, decodeBufferToText } = require('../lib/text');
const { readPositiveIntegerEnv } = require('../core/overloadGuard');
const { writeNdjsonChunk } = require('../core/apiResponse');

const DEFAULT_MAX_UPLOAD_FILE_MB = 1;
const DEFAULT_MAX_UPLOAD_FILES = 200;
const DEFAULT_MAX_MODEL_PACKAGE_MB = 512;
const DEFAULT_MAX_SOURCE_TEXT_CHARS = 180000;
const DEFAULT_MAX_STRUCTURED_RECORDS_PER_FILE = 500;
const DEFAULT_UPLOAD_YIELD_EVERY_FILES = 8;

const MAX_UPLOAD_FILE_MB = readPositiveIntegerEnv('MAX_UPLOAD_FILE_MB', DEFAULT_MAX_UPLOAD_FILE_MB);
const MAX_UPLOAD_FILES = readPositiveIntegerEnv('MAX_UPLOAD_FILES', DEFAULT_MAX_UPLOAD_FILES);
const MAX_MODEL_PACKAGE_MB = readPositiveIntegerEnv('MAX_MODEL_PACKAGE_MB', DEFAULT_MAX_MODEL_PACKAGE_MB);
const MAX_SOURCE_TEXT_CHARS = readPositiveIntegerEnv('MAX_SOURCE_TEXT_CHARS', DEFAULT_MAX_SOURCE_TEXT_CHARS);
const MAX_STRUCTURED_RECORDS_PER_FILE = readPositiveIntegerEnv(
  'MAX_STRUCTURED_RECORDS_PER_FILE',
  DEFAULT_MAX_STRUCTURED_RECORDS_PER_FILE
);
const UPLOAD_YIELD_EVERY_FILES = readPositiveIntegerEnv(
  'UPLOAD_YIELD_EVERY_FILES',
  DEFAULT_UPLOAD_YIELD_EVERY_FILES
);

function yieldToEventLoop() {
  return new Promise((resolve) => setImmediate(resolve));
}

function trimSourceText(value) {
  const normalized = cleanText(value || '');
  if (!normalized) {
    return '';
  }
  if (normalized.length <= MAX_SOURCE_TEXT_CHARS) {
    return normalized;
  }

  return `${normalized.slice(0, MAX_SOURCE_TEXT_CHARS)}\n\n[TRUNCATED:${normalized.length - MAX_SOURCE_TEXT_CHARS}]`;
}

function createUploadMiddlewares() {
  const sourceUpload = multer({
    storage: multer.memoryStorage(),
    limits: {
      fileSize: MAX_UPLOAD_FILE_MB * 1024 * 1024,
      files: MAX_UPLOAD_FILES,
    },
  });

  const modelPackageUpload = multer({
    storage: multer.memoryStorage(),
    limits: {
      fileSize: MAX_MODEL_PACKAGE_MB * 1024 * 1024,
      files: 1,
    },
  });

  return {
    sourceUpload,
    modelPackageUpload,
    limits: {
      maxUploadFileMb: MAX_UPLOAD_FILE_MB,
      maxUploadFiles: MAX_UPLOAD_FILES,
      maxModelPackageMb: MAX_MODEL_PACKAGE_MB,
      maxSourceTextChars: MAX_SOURCE_TEXT_CHARS,
      maxStructuredRecordsPerFile: MAX_STRUCTURED_RECORDS_PER_FILE,
    },
  };
}

function detectStructuredFileKind(file = {}) {
  const extension = path.extname(file.originalname || '').toLowerCase();
  const mimeType = String(file.mimetype || '').toLowerCase();

  if (extension === '.json' || mimeType.includes('json')) {
    return 'json';
  }
  if (extension === '.csv' || mimeType.includes('csv')) {
    return 'csv';
  }
  return 'txt';
}

function splitCsvRow(row = '', delimiter = ',') {
  const cells = [];
  let currentCell = '';
  let insideQuotes = false;

  for (let index = 0; index < row.length; index += 1) {
    const character = row[index];
    const nextCharacter = row[index + 1];

    if (character === '"') {
      if (insideQuotes && nextCharacter === '"') {
        currentCell += '"';
        index += 1;
        continue;
      }

      insideQuotes = !insideQuotes;
      continue;
    }

    if (character === delimiter && !insideQuotes) {
      cells.push(cleanText(currentCell));
      currentCell = '';
      continue;
    }

    currentCell += character;
  }

  cells.push(cleanText(currentCell));
  return cells;
}

function detectCsvDelimiter(headerLine = '') {
  const delimiters = [',', ';', '\t'];
  return delimiters
    .map((delimiter) => ({
      delimiter,
      score: (headerLine.match(new RegExp(`\\${delimiter}`, 'g')) || []).length,
    }))
    .sort((left, right) => right.score - left.score)[0]?.delimiter || ',';
}

function flattenStructuredValue(value, prefix = '') {
  if (value === null || value === undefined) {
    return [];
  }

  if (Array.isArray(value)) {
    return value.flatMap((item, index) => flattenStructuredValue(
      item,
      prefix ? `${prefix}[${index}]` : `[${index}]`
    ));
  }

  if (typeof value === 'object') {
    return Object.entries(value).flatMap(([key, nestedValue]) => (
      flattenStructuredValue(nestedValue, prefix ? `${prefix}.${key}` : key)
    ));
  }

  const normalized = cleanText(String(value));
  if (!normalized) {
    return [];
  }

  return [prefix ? `${prefix}: ${normalized}` : normalized];
}

function buildSourcesFromJsonFile(file, rawText) {
  const parsed = JSON.parse(rawText);
  const entries = Array.isArray(parsed) ? parsed : [parsed];
  const limitedEntries = entries.slice(0, MAX_STRUCTURED_RECORDS_PER_FILE);

  return limitedEntries
    .map((entry, index) => {
      const lines = flattenStructuredValue(entry);
      const content = trimSourceText(lines.join('\n'));
      if (!content) {
        return null;
      }

      return {
        type: 'file',
        label: limitedEntries.length > 1 ? `${file.originalname} #${index + 1}` : file.originalname,
        content,
      };
    })
    .filter(Boolean);
}

function buildSourcesFromCsvFile(file, rawText) {
  const lines = rawText
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean);

  if (!lines.length) {
    return [];
  }

  const delimiter = detectCsvDelimiter(lines[0]);
  const headerCells = splitCsvRow(lines[0], delimiter);
  const dataRows = lines.slice(1, MAX_STRUCTURED_RECORDS_PER_FILE + 1);

  if (!dataRows.length) {
    const headerOnlyContent = trimSourceText(headerCells.join(' '));
    return headerOnlyContent ? [{
      type: 'file',
      label: file.originalname,
      content: headerOnlyContent,
    }] : [];
  }

  return dataRows
    .map((row, index) => {
      const values = splitCsvRow(row, delimiter);
      const content = trimSourceText(values.map((value, cellIndex) => {
        const header = cleanText(headerCells[cellIndex] || `column_${cellIndex + 1}`);
        return `${header}: ${value}`;
      }).join('\n'));

      if (!content) {
        return null;
      }

      return {
        type: 'file',
        label: `${file.originalname} #${index + 1}`,
        content,
      };
    })
    .filter(Boolean);
}

function buildSourcesFromUploadedFile(file) {
  const rawText = decodeBufferToText(file.buffer);
  if (!rawText) {
    return [];
  }

  const fileKind = detectStructuredFileKind(file);
  if (fileKind === 'json') {
    return buildSourcesFromJsonFile(file, rawText);
  }
  if (fileKind === 'csv') {
    return buildSourcesFromCsvFile(file, rawText);
  }

  const content = trimSourceText(rawText);
  return content ? [{
    type: 'file',
    label: file.originalname,
    content,
  }] : [];
}

async function prepareUploadedSources(files, response = null) {
  const totalFiles = files.length;
  const preparedSources = [];

  if (response) {
    writeNdjsonChunk(response, {
      type: 'processing_progress',
      processedFiles: 0,
      totalFiles,
      percent: 0,
    });
  }

  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    preparedSources.push(...buildSourcesFromUploadedFile(file));

    if (response) {
      const processedFiles = index + 1;
      const percent = Number(((processedFiles / totalFiles) * 100).toFixed(1));
      writeNdjsonChunk(response, {
        type: 'processing_progress',
        processedFiles,
        totalFiles,
        percent,
        fileName: file.originalname,
      });
    }

    if ((index + 1) % UPLOAD_YIELD_EVERY_FILES === 0) {
      await yieldToEventLoop();
    }
  }

  return preparedSources;
}

module.exports = {
  createUploadMiddlewares,
  prepareUploadedSources,
};
