const fs = require('fs/promises');
const path = require('path');

const MODEL_LIBRARY_DIR = path.join(__dirname, '..', 'data', 'model-library');

function getModelLibraryItemDir(modelId) {
  return path.join(MODEL_LIBRARY_DIR, modelId);
}

function getModelLibraryPackagePath(modelId) {
  return path.join(getModelLibraryItemDir(modelId), 'model-package.json');
}

async function ensureModelLibraryLayout() {
  await fs.mkdir(MODEL_LIBRARY_DIR, { recursive: true });
}

async function writeModelLibraryPackage(modelId, payload) {
  const targetDir = getModelLibraryItemDir(modelId);
  await fs.mkdir(targetDir, { recursive: true });
  const targetPath = getModelLibraryPackagePath(modelId);
  await fs.writeFile(targetPath, JSON.stringify(payload, null, 2), 'utf8');
  return targetPath;
}

async function readModelLibraryPackage(modelId) {
  const targetPath = getModelLibraryPackagePath(modelId);
  const raw = await fs.readFile(targetPath, 'utf8');
  return JSON.parse(raw);
}

async function removeModelLibraryPackage(modelId) {
  await fs.rm(getModelLibraryItemDir(modelId), {
    recursive: true,
    force: true,
  });
}

async function modelLibraryPackageExists(modelId) {
  try {
    await fs.access(getModelLibraryPackagePath(modelId));
    return true;
  } catch (_error) {
    return false;
  }
}

module.exports = {
  MODEL_LIBRARY_DIR,
  ensureModelLibraryLayout,
  getModelLibraryItemDir,
  getModelLibraryPackagePath,
  modelLibraryPackageExists,
  readModelLibraryPackage,
  removeModelLibraryPackage,
  writeModelLibraryPackage,
};
