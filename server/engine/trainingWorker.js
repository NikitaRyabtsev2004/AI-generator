const fs = require('fs/promises');
const path = require('path');
const readline = require('readline');
const { parentPort, workerData } = require('node:worker_threads');
const {
  cleanupTempConfig,
  createTempConfigFile,
  startPythonBackendProcess,
} = require('./pythonBridge');

let stopRequested = false;
let activeChild = null;
let stopSignalPath = '';

function postMessage(type, payload = {}) {
  if (!parentPort) {
    return;
  }
  parentPort.postMessage({
    type,
    ...payload,
  });
}

function isStopRequested(stopSignalView) {
  return stopRequested || (stopSignalView ? Atomics.load(stopSignalView, 0) === 1 : false);
}

async function writeStopSignalFile() {
  if (!stopSignalPath) {
    return;
  }
  try {
    await fs.writeFile(stopSignalPath, '1', 'utf8');
  } catch (_error) {
    // Ignore stop-signal write failures.
  }
}

if (parentPort) {
  parentPort.on('message', (message) => {
    if (message?.type !== 'stop') {
      return;
    }

    stopRequested = true;
    writeStopSignalFile().catch(() => {});
  });
}

process.on('exit', () => {
  if (activeChild && !activeChild.killed) {
    activeChild.kill('SIGTERM');
  }
});

async function main() {
  const {
    settings,
    trainingPayloadPath = '',
    storage,
    resumeFromCheckpoint,
    resumeEpochOffset = 0,
    manifest,
    positiveFeedbackCount,
    stopSignalBuffer = null,
  } = workerData || {};
  const stopSignalView = stopSignalBuffer ? new Int32Array(stopSignalBuffer) : null;

  if (isStopRequested(stopSignalView)) {
    return;
  }

  const temp = await createTempConfigFile('ai-generator-train-worker', {});
  stopSignalPath = path.join(temp.dir, 'stop.signal');

  const pythonConfig = {
    settings,
    trainingPayloadPath,
    storage,
    resumeFromCheckpoint: Boolean(resumeFromCheckpoint),
    resumeEpochOffset: Math.max(Number(resumeEpochOffset) || 0, 0),
    manifest: manifest || {},
    positiveFeedbackCount: Number(positiveFeedbackCount) || 0,
    stopSignalPath,
  };
  await fs.writeFile(temp.filePath, JSON.stringify(pythonConfig, null, 2), 'utf8');

  let finished = false;
  let stderrOutput = '';
  let forwardedError = false;

  try {
    activeChild = await startPythonBackendProcess('train', temp.filePath);

    const stdoutReader = readline.createInterface({
      input: activeChild.stdout,
      crlfDelay: Infinity,
    });

    stdoutReader.on('line', (line) => {
      const trimmed = String(line || '').trim();
      if (!trimmed || finished) {
        return;
      }

      let payload = null;
      try {
        payload = JSON.parse(trimmed);
      } catch (_error) {
        return;
      }

      if (!payload || typeof payload !== 'object') {
        return;
      }

      if (payload.type === 'error') {
        forwardedError = true;
        postMessage('error', {
          error: {
            message: payload.error?.message || payload.error || 'Python training backend failed.',
            stack: '',
          },
        });
        return;
      }

      if (payload.type) {
        postMessage(payload.type, payload);
        if (payload.type === 'done') {
          finished = true;
        }
      }
    });

    activeChild.stderr.on('data', (chunk) => {
      stderrOutput += chunk.toString();
    });

    const exitCode = await new Promise((resolve, reject) => {
      activeChild.on('error', reject);
      activeChild.on('exit', (code) => resolve(Number(code ?? 0)));
    });

    if (!finished && !forwardedError && exitCode !== 0) {
      const message = String(stderrOutput || '').trim() || `Python training backend exited with code ${exitCode}.`;
      postMessage('error', {
        error: {
          message,
          stack: '',
        },
      });
    }
  } finally {
    activeChild = null;
    await cleanupTempConfig(temp.dir);
  }
}

main().catch((error) => {
  postMessage('error', {
    error: {
      message: error?.message || 'Training worker crashed.',
      stack: error?.stack || '',
    },
  });
  process.exitCode = 1;
});
