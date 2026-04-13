const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');
const readline = require('readline');
const crypto = require('crypto');
const multer = require('multer');
const { cleanText, computeStats, decodeBufferToText } = require('../lib/text');
const { writeNdjsonChunk } = require('../core/apiResponse');

const UPLOADS_DIR = path.join(__dirname, '..', 'data', 'uploaded-datasets');
const DEFAULT_MAX_UPLOAD_FILE_MB = 2048;
const DEFAULT_MAX_UPLOAD_FILES = 5000;
const DEFAULT_MAX_MODEL_PACKAGE_MB = 1024;
const DEFAULT_MAX_SOURCE_TEXT_CHARS = 210000;
const DEFAULT_MAX_STRUCTURED_RECORDS_PER_FILE = 500;
const DEFAULT_UPLOAD_YIELD_EVERY_FILES = 4;
const DEFAULT_HEAD_SAMPLE_BYTES = 6 * 1024 * 1024;
const DEFAULT_DATASET_REFERENCE_MIN_MB = 8;

function readOptionalLimitEnv(name, fallback) {
  const rawValue = process.env[name];
  if (rawValue === undefined || rawValue === null || rawValue === '') {
    return fallback;
  }

  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return fallback;
  }

  return Math.floor(parsed);
}

const MAX_UPLOAD_FILE_MB = readOptionalLimitEnv('MAX_UPLOAD_FILE_MB', DEFAULT_MAX_UPLOAD_FILE_MB);
const MAX_UPLOAD_FILES = readOptionalLimitEnv('MAX_UPLOAD_FILES', DEFAULT_MAX_UPLOAD_FILES);
const MAX_MODEL_PACKAGE_MB = readOptionalLimitEnv('MAX_MODEL_PACKAGE_MB', DEFAULT_MAX_MODEL_PACKAGE_MB);
const MAX_SOURCE_TEXT_CHARS = readOptionalLimitEnv('MAX_SOURCE_TEXT_CHARS', DEFAULT_MAX_SOURCE_TEXT_CHARS);
const MAX_STRUCTURED_RECORDS_PER_FILE = readOptionalLimitEnv(
  'MAX_STRUCTURED_RECORDS_PER_FILE',
  DEFAULT_MAX_STRUCTURED_RECORDS_PER_FILE
);
const UPLOAD_YIELD_EVERY_FILES = readOptionalLimitEnv(
  'UPLOAD_YIELD_EVERY_FILES',
  DEFAULT_UPLOAD_YIELD_EVERY_FILES
);
const HEAD_SAMPLE_BYTES = readOptionalLimitEnv('UPLOAD_HEAD_SAMPLE_BYTES', DEFAULT_HEAD_SAMPLE_BYTES);
const DATASET_REFERENCE_MIN_MB = readOptionalLimitEnv(
  'DATASET_REFERENCE_MIN_MB',
  DEFAULT_DATASET_REFERENCE_MIN_MB
);

const DATASET_REFERENCE_MIN_BYTES = Math.max(1, DATASET_REFERENCE_MIN_MB) * 1024 * 1024;

function ensureUploadsDir() {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

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

function buildMulterLimits(maxFileMb, maxFiles) {
  const limits = {};
  if (maxFileMb > 0) {
    limits.fileSize = maxFileMb * 1024 * 1024;
  }
  if (maxFiles > 0) {
    limits.files = maxFiles;
  }
  return limits;
}

function createSourceStorage() {
  ensureUploadsDir();
  return multer.diskStorage({
    destination: (_request, _file, callback) => {
      try {
        ensureUploadsDir();
        callback(null, UPLOADS_DIR);
      } catch (error) {
        callback(error);
      }
    },
    filename: (_request, file, callback) => {
      const extension = path.extname(file.originalname || '').slice(0, 16);
      const uniquePart = `${Date.now()}-${crypto.randomUUID()}`;
      callback(null, `${uniquePart}${extension}`);
    },
  });
}

function createUploadMiddlewares() {
  const sourceUpload = multer({
    storage: createSourceStorage(),
    limits: buildMulterLimits(MAX_UPLOAD_FILE_MB, MAX_UPLOAD_FILES),
  });

  const modelPackageUpload = multer({
    storage: multer.memoryStorage(),
    limits: buildMulterLimits(MAX_MODEL_PACKAGE_MB, 1),
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
      uploadMode: 'disk',
    },
  };
}

function detectStructuredFileKind(file = {}) {
  const extension = path.extname(file.originalname || '').toLowerCase();
  const mimeType = String(file.mimetype || '').toLowerCase();

  if (extension === '.parquet' || mimeType.includes('parquet')) {
    return 'parquet';
  }
  if (extension === '.jsonl' || extension === '.ndjson') {
    return 'jsonl';
  }
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

async function readFileHeadBuffer(file, maxBytes = HEAD_SAMPLE_BYTES) {
  if (file?.buffer && Buffer.isBuffer(file.buffer)) {
    return file.buffer.slice(0, maxBytes);
  }

  const sourcePath = cleanText(file?.path || '');
  if (!sourcePath) {
    return Buffer.from('');
  }

  const handle = await fsPromises.open(sourcePath, 'r');
  try {
    const buffer = Buffer.alloc(Math.max(1, maxBytes));
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    return buffer.slice(0, bytesRead);
  } finally {
    await handle.close();
  }
}

async function decodeTextFromFileHead(file, maxBytes = HEAD_SAMPLE_BYTES) {
  const headBuffer = await readFileHeadBuffer(file, maxBytes);
  const text = decodeBufferToText(headBuffer);
  const fileSize = Math.max(Number(file?.size) || headBuffer.length, headBuffer.length);
  if (fileSize <= headBuffer.length) {
    return cleanText(text);
  }

  return cleanText(`${text}\n\n[TRUNCATED_BYTES:${fileSize - headBuffer.length}]`);
}

async function readLimitedLines(filePath, maxLines) {
  const lines = [];
  const stream = fs.createReadStream(filePath, { encoding: 'utf8' });
  const lineReader = readline.createInterface({
    input: stream,
    crlfDelay: Infinity,
  });

  try {
    for await (const line of lineReader) {
      if (lines.length >= maxLines) {
        break;
      }
      const normalized = String(line || '');
      if (normalized) {
        lines.push(normalized);
      }
    }
  } finally {
    lineReader.close();
    stream.destroy();
  }

  return lines;
}

async function buildSourcesFromJsonFile(file, mode = 'knowledge') {
  const fileText = await decodeTextFromFileHead(file);
  let parsed;
  try {
    parsed = JSON.parse(fileText);
  } catch (_error) {
    const content = trimSourceText(fileText);
    return content ? [{
      type: 'file',
      label: file.originalname,
      content,
    }] : [];
  }

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
        type: mode === 'training_queue' ? 'file' : 'file',
        label: limitedEntries.length > 1 ? `${file.originalname} #${index + 1}` : file.originalname,
        content,
      };
    })
    .filter(Boolean);
}

async function buildSourcesFromCsvFile(file) {
  const filePath = cleanText(file.path || '');
  if (!filePath) {
    return [];
  }

  const lines = await readLimitedLines(filePath, MAX_STRUCTURED_RECORDS_PER_FILE + 1);
  const nonEmptyLines = lines
    .map((line) => line.trim())
    .filter(Boolean);

  if (!nonEmptyLines.length) {
    return [];
  }

  const delimiter = detectCsvDelimiter(nonEmptyLines[0]);
  const headerCells = splitCsvRow(nonEmptyLines[0], delimiter);
  const dataRows = nonEmptyLines.slice(1, MAX_STRUCTURED_RECORDS_PER_FILE + 1);
  if (!dataRows.length) {
    const headerOnly = trimSourceText(headerCells.join(' '));
    return headerOnly ? [{
      type: 'file',
      label: file.originalname,
      content: headerOnly,
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

async function buildSourcesFromJsonlFile(file) {
  const filePath = cleanText(file.path || '');
  if (!filePath) {
    return [];
  }

  const lines = await readLimitedLines(filePath, MAX_STRUCTURED_RECORDS_PER_FILE);
  return lines
    .map((line, index) => {
      const normalizedLine = cleanText(line);
      if (!normalizedLine) {
        return null;
      }

      try {
        const parsed = JSON.parse(normalizedLine);
        const content = trimSourceText(flattenStructuredValue(parsed).join('\n'));
        if (!content) {
          return null;
        }
        return {
          type: 'file',
          label: `${file.originalname} #${index + 1}`,
          content,
        };
      } catch (_error) {
        return {
          type: 'file',
          label: `${file.originalname} #${index + 1}`,
          content: trimSourceText(normalizedLine),
        };
      }
    })
    .filter(Boolean);
}

async function buildSourcesFromTxtFile(file) {
  const content = trimSourceText(await decodeTextFromFileHead(file));
  return content ? [{
    type: 'file',
    label: file.originalname,
    content,
  }] : [];
}

async function buildKnowledgeSourcesFromUploadedFile(file, fileKind) {
  if (fileKind === 'json') {
    return buildSourcesFromJsonFile(file, 'knowledge');
  }
  if (fileKind === 'csv') {
    return buildSourcesFromCsvFile(file);
  }
  if (fileKind === 'jsonl') {
    return buildSourcesFromJsonlFile(file);
  }
  if (fileKind === 'parquet') {
    const metaText = trimSourceText(
      `Файл ${file.originalname} (${Math.max(Number(file.size) || 0, 0)} bytes) в формате parquet загружен. ` +
      'Для обучения добавляйте parquet в очередь обучения: сервер передаст его напрямую в Python-обработчик.'
    );
    return [{
      type: 'file',
      label: file.originalname,
      content: metaText,
    }];
  }

  return buildSourcesFromTxtFile(file);
}

function shouldUseDatasetFileReference(file, fileKind, mode = 'knowledge') {
  if (mode !== 'training_queue') {
    return false;
  }

  if (fileKind === 'parquet' || fileKind === 'jsonl') {
    return true;
  }

  return Math.max(Number(file.size) || 0, 0) >= DATASET_REFERENCE_MIN_BYTES;
}

function createDatasetFileReference(file) {
  const filePath = cleanText(file.path || '');
  return {
    type: 'dataset_file',
    label: cleanText(file.originalname || '') || path.basename(filePath) || 'dataset_file',
    datasetFilePath: filePath,
    contentSize: Math.max(Number(file.size) || 0, 0),
    stats: computeStats(''),
  };
}

async function cleanupUploadedFile(file) {
  const filePath = cleanText(file?.path || '');
  if (!filePath) {
    return;
  }
  try {
    await fsPromises.rm(filePath, { force: true });
  } catch (_error) {
    // Ignore cleanup failures.
  }
}

async function prepareUploadedSources(files, response = null, options = {}) {
  const mode = options.mode === 'training_queue' ? 'training_queue' : 'knowledge';
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
    const fileKind = detectStructuredFileKind(file);
    const keepAsDatasetReference = shouldUseDatasetFileReference(file, fileKind, mode);

    try {
      if (keepAsDatasetReference) {
        const datasetPath = cleanText(file.path || '');
        if (!datasetPath) {
          throw new Error(`Не удалось сохранить файл "${cleanText(file.originalname || '') || 'без имени'}" во временное хранилище.`);
        }
        await fsPromises.access(datasetPath);
        preparedSources.push(createDatasetFileReference(file));
      } else {
        preparedSources.push(...(await buildKnowledgeSourcesFromUploadedFile(file, fileKind)));
      }
    } catch (error) {
      if (error?.code === 'ENOENT') {
        throw new Error(
          `Файл "${cleanText(file.originalname || '') || 'без имени'}" не найден во временном хранилище. ` +
          'Повторите загрузку: файл мог быть удален до завершения обработки.'
        );
      }
      throw error;
    } finally {
      if (!keepAsDatasetReference) {
        await cleanupUploadedFile(file);
      }
    }

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
