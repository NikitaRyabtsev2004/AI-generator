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
const STDERR_RING_LIMIT_CHARS = Math.max(16_384, Number(process.env.TRAINING_STDERR_RING_LIMIT_CHARS) || 262_144);
const STDOUT_RING_LIMIT_CHARS = Math.max(16_384, Number(process.env.TRAINING_STDOUT_RING_LIMIT_CHARS) || 262_144);
const TRAINING_NO_OUTPUT_TIMEOUT_MS = Math.max(60_000, Number(process.env.TRAINING_NO_OUTPUT_TIMEOUT_MS) || 240_000);
const PREPARATION_NO_OUTPUT_TIMEOUT_MS = Math.max(120_000, Number(process.env.PREPARATION_NO_OUTPUT_TIMEOUT_MS) || 1_800_000);
const CHECKPOINT_NO_OUTPUT_TIMEOUT_MS = Math.max(120_000, Number(process.env.CHECKPOINT_NO_OUTPUT_TIMEOUT_MS) || 3_600_000);

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

function resolveNoOutputTimeoutMs(lastPayloadType) {
  if (lastPayloadType === 'checkpointing' || lastPayloadType === 'recovery_checkpoint') {
    return CHECKPOINT_NO_OUTPUT_TIMEOUT_MS;
  }
  if (lastPayloadType === 'preparing' || lastPayloadType === 'prepared') {
    return PREPARATION_NO_OUTPUT_TIMEOUT_MS;
  }
  return TRAINING_NO_OUTPUT_TIMEOUT_MS;
}

async function main() {
  const {
    settings,
    trainingPayloadPath = '',
    storage,
    resumeFromCheckpoint,
    resumeEpochOffset = 0,
    resumeBatchOffset = 0,
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
    resumeBatchOffset: Math.max(Number(resumeBatchOffset) || 0, 0),
    manifest: manifest || {},
    positiveFeedbackCount: Number(positiveFeedbackCount) || 0,
    stopSignalPath,
  };
  await fs.writeFile(temp.filePath, JSON.stringify(pythonConfig, null, 2), 'utf8');

  let finished = false;
  let stderrOutput = '';
  let stdoutOutput = '';
  let forwardedError = false;
  let lastPayloadAtMs = Date.now();
  let lastPayloadType = 'startup';
  let stallWatchdogId = null;

  try {
    activeChild = await startPythonBackendProcess('train', temp.filePath);

    const updatePayloadHeartbeat = (payloadType = '') => {
      lastPayloadAtMs = Date.now();
      if (payloadType) {
        lastPayloadType = String(payloadType);
      }
    };

    updatePayloadHeartbeat('startup');
    stallWatchdogId = setInterval(() => {
      if (finished || !activeChild || activeChild.killed) {
        return;
      }
      const now = Date.now();
      const timeoutMs = resolveNoOutputTimeoutMs(lastPayloadType);
      const idleMs = now - lastPayloadAtMs;
      if (idleMs < timeoutMs) {
        return;
      }

      forwardedError = true;
      finished = true;
      const idleSec = Math.max(1, Math.round(idleMs / 1000));
      postMessage('error', {
        error: {
          message:
            `Воркер обучения не присылал телеметрию ${idleSec} с (последний тип: ${lastPayloadType}). ` +
            'Процесс автоматически остановлен как зависший.',
          stack: '',
        },
      });
      try {
        activeChild.kill('SIGTERM');
      } catch (_error) {
        // Ignore kill failures.
      }
    }, 5000);

    const stdoutReader = readline.createInterface({
      input: activeChild.stdout,
      crlfDelay: Infinity,
    });

    stdoutReader.on('line', (line) => {
      const trimmed = String(line || '').trim();
      if (!trimmed || finished) {
        return;
      }
      stdoutOutput += `${trimmed}\n`;
      if (stdoutOutput.length > STDOUT_RING_LIMIT_CHARS) {
        stdoutOutput = stdoutOutput.slice(-STDOUT_RING_LIMIT_CHARS);
      }

      let payload = null;
      try {
        payload = JSON.parse(trimmed);
      } catch (_error) {
        const sanitizedLine = trimmed
          .replace(/-Infinity/g, 'null')
          .replace(/\bInfinity\b/g, 'null')
          .replace(/\bNaN\b/g, 'null');
        if (sanitizedLine === trimmed) {
          return;
        }
        try {
          payload = JSON.parse(sanitizedLine);
        } catch (_error2) {
          return;
        }
      }

      if (!payload || typeof payload !== 'object') {
        return;
      }

      updatePayloadHeartbeat(payload.type);

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
      if (stderrOutput.length > STDERR_RING_LIMIT_CHARS) {
        stderrOutput = stderrOutput.slice(-STDERR_RING_LIMIT_CHARS);
      }
    });

    const exitCode = await new Promise((resolve, reject) => {
      activeChild.on('error', reject);
      activeChild.on('exit', (code) => resolve(Number(code ?? 0)));
    });

    if (!finished && !forwardedError) {
      const stderrText = String(stderrOutput || '').trim();
      const stdoutText = String(stdoutOutput || '').trim();
      const stdoutTail = stdoutText
        ? stdoutText
          .split(/\r?\n/u)
          .filter(Boolean)
          .slice(-8)
          .join('\n')
        : '';
      const message = exitCode === 0
        ? (
          stderrText ||
          (stdoutTail ? `Python stdout before exit:\n${stdoutTail}` : '') ||
          'Python training backend exited before sending final training result.'
        )
        : (
          stderrText ||
          (stdoutTail ? `Python stdout before exit:\n${stdoutTail}` : '') ||
          `Python training backend exited with code ${exitCode}.`
        );
      postMessage('error', {
        error: {
          message,
          stack: '',
        },
      });
    }
  } finally {
    if (stallWatchdogId) {
      clearInterval(stallWatchdogId);
      stallWatchdogId = null;
    }
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
