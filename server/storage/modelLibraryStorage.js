const fs = require('fs/promises');
const path = require('path');

const MODEL_LIBRARY_ROOT = path.join(__dirname, '..', 'data', 'model-library', 'users');
let activeModelLibraryUserId = 'legacy-local';

function normalizeModelLibraryUserId(userId) {
  const normalized = String(userId || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]/gu, '-')
    .replace(/-+/gu, '-')
    .slice(0, 80);
  return normalized || 'legacy-local';
}

function setActiveModelLibraryUser(userId) {
  activeModelLibraryUserId = normalizeModelLibraryUserId(userId);
  return getModelLibraryDir();
}

function getModelLibraryDir(userId = activeModelLibraryUserId) {
  return path.join(MODEL_LIBRARY_ROOT, normalizeModelLibraryUserId(userId));
}

function normalizeModelId(modelId) {
  const normalized = String(modelId || '').trim();
  if (!normalized) {
    throw new Error('Model id is required for model library operation.');
  }
  return normalized;
}

function getModelLibraryItemDir(modelId, userId = activeModelLibraryUserId) {
  return path.join(getModelLibraryDir(userId), normalizeModelId(modelId));
}

function getModelLibraryPackagePath(modelId, userId = activeModelLibraryUserId) {
  return path.join(getModelLibraryItemDir(modelId, userId), 'model-package.json');
}

async function ensureModelLibraryLayout(userId = activeModelLibraryUserId) {
  await fs.mkdir(getModelLibraryDir(userId), { recursive: true });
}

async function writeModelLibraryPackage(modelId, payload, userId = activeModelLibraryUserId) {
  const targetDir = getModelLibraryItemDir(modelId, userId);
  await fs.mkdir(targetDir, { recursive: true });
  const targetPath = getModelLibraryPackagePath(modelId, userId);
  await fs.writeFile(targetPath, JSON.stringify(payload), 'utf8');
  return targetPath;
}

async function readModelLibraryPackage(modelId, userId = activeModelLibraryUserId) {
  const targetPath = getModelLibraryPackagePath(modelId, userId);
  const raw = await fs.readFile(targetPath, 'utf8');
  return JSON.parse(raw);
}

async function removeModelLibraryPackage(modelId, userId = activeModelLibraryUserId) {
  await fs.rm(getModelLibraryItemDir(modelId, userId), {
    recursive: true,
    force: true,
  });
}

async function modelLibraryPackageExists(modelId, userId = activeModelLibraryUserId) {
  try {
    await fs.access(getModelLibraryPackagePath(modelId, userId));
    return true;
  } catch (_error) {
    return false;
  }
}

module.exports = {
  MODEL_LIBRARY_ROOT,
  ensureModelLibraryLayout,
  getModelLibraryDir,
  getModelLibraryItemDir,
  getModelLibraryPackagePath,
  modelLibraryPackageExists,
  readModelLibraryPackage,
  removeModelLibraryPackage,
  setActiveModelLibraryUser,
  writeModelLibraryPackage,
};
