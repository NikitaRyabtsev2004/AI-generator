function readPositiveIntegerEnv(name, fallbackValue) {
  const value = Number(process.env[name]);
  if (!Number.isFinite(value) || value <= 0) {
    return fallbackValue;
  }
  return Math.floor(value);
}

function createOverloadGuard() {
  const maxActive = readPositiveIntegerEnv('MAX_HEAVY_ACTIVE_REQUESTS', 4);
  const maxQueued = readPositiveIntegerEnv('MAX_HEAVY_QUEUED_REQUESTS', 24);
  const queueTimeoutMs = readPositiveIntegerEnv('MAX_HEAVY_QUEUE_WAIT_MS', 25000);
  const queue = [];
  let activeCount = 0;
  let droppedRequests = 0;

  function release() {
    activeCount = Math.max(activeCount - 1, 0);
    if (!queue.length) {
      return;
    }

    const waiter = queue.shift();
    clearTimeout(waiter.timeoutId);
    activeCount += 1;
    waiter.resolve();
  }

  function acquire() {
    if (activeCount < maxActive) {
      activeCount += 1;
      return Promise.resolve();
    }

    if (queue.length >= maxQueued) {
      droppedRequests += 1;
      return Promise.reject(new Error('Server is overloaded. Too many heavy requests.'));
    }

    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        const index = queue.findIndex((entry) => entry.timeoutId === timeoutId);
        if (index >= 0) {
          queue.splice(index, 1);
        }
        droppedRequests += 1;
        reject(new Error('Server is overloaded. Heavy request queue timed out.'));
      }, queueTimeoutMs);

      queue.push({ resolve, reject, timeoutId });
    });
  }

  async function middleware(_request, response, next) {
    try {
      await acquire();
    } catch (error) {
      response.status(503).json({
        ok: false,
        error: 'Сервер сейчас перегружен тяжелыми задачами. Попробуйте через несколько секунд.',
        details: {
          reason: error.message,
          retryAfterMs: queueTimeoutMs,
        },
      });
      return;
    }

    let released = false;
    const finalize = () => {
      if (released) {
        return;
      }
      released = true;
      release();
    };

    response.once('finish', finalize);
    response.once('close', finalize);
    next();
  }

  function snapshot() {
    return {
      maxActive,
      maxQueued,
      queueTimeoutMs,
      activeCount,
      queuedCount: queue.length,
      droppedRequests,
    };
  }

  return {
    middleware,
    snapshot,
  };
}

module.exports = {
  createOverloadGuard,
  readPositiveIntegerEnv,
};
