async function requestJson(url, options = {}) {
  const {
    timeoutMs = 25000,
    signal,
    ...fetchOptions
  } = options;
  const controller = new AbortController();
  const timeoutEnabled = Number(timeoutMs) > 0;
  let timeoutId = null;

  const handleAbort = () => {
    controller.abort();
  };

  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener('abort', handleAbort, { once: true });
    }
  }

  if (timeoutEnabled) {
    timeoutId = window.setTimeout(() => {
      controller.abort();
    }, Number(timeoutMs));
  }

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({}));

    if (!response.ok || payload.ok === false) {
      throw new Error(payload.error || 'Запрос к серверу завершился ошибкой.');
    }

    return payload;
  } catch (error) {
    if (error?.name === 'AbortError' && !signal?.aborted && timeoutEnabled) {
      throw new Error('Сервер слишком долго не отвечает. Проверьте живой журнал и попробуйте снова.');
    }
    throw error;
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
    signal?.removeEventListener?.('abort', handleAbort);
  }
}

function uploadFormDataWithProgress(url, formData, handlers = {}) {
  const {
    onUploadProgress,
    onProcessingProgress,
  } = handlers;

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url, true);
    xhr.setRequestHeader('Accept', 'application/x-ndjson');
    xhr.timeout = 180000;

    let readOffset = 0;
    let pendingLine = '';
    let settled = false;

    const processMessage = (message) => {
      if (!message || settled) {
        return;
      }

      if (message.type === 'processing_progress' && typeof onProcessingProgress === 'function') {
        onProcessingProgress({
          percent: Number(message.percent) || 0,
          processedFiles: Number(message.processedFiles) || 0,
          totalFiles: Number(message.totalFiles) || 0,
          fileName: message.fileName || '',
        });
        return;
      }

      if (message.type === 'result') {
        settled = true;
        resolve(message);
        return;
      }

      if (message.type === 'error') {
        settled = true;
        reject(new Error(message.message || 'Запрос к серверу завершился ошибкой.'));
      }
    };

    const parseResponseChunks = (isFinal = false) => {
      const fullText = xhr.responseText || '';
      if (readOffset < fullText.length) {
        pendingLine += fullText.slice(readOffset);
        readOffset = fullText.length;
      }

      const parts = pendingLine.split('\n');
      pendingLine = parts.pop() || '';

      parts
        .map((line) => line.trim())
        .filter(Boolean)
        .forEach((line) => {
          try {
            processMessage(JSON.parse(line));
          } catch (_error) {
            // Ignore malformed incremental chunks and continue parsing.
          }
        });

      if (isFinal && pendingLine.trim()) {
        try {
          processMessage(JSON.parse(pendingLine.trim()));
        } catch (_error) {
          // Ignore trailing malformed payload.
        }
        pendingLine = '';
      }
    };

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable || typeof onUploadProgress !== 'function') {
        return;
      }

      const percent = Math.min(100, Math.max(0, Number(((event.loaded / event.total) * 100).toFixed(1))));
      onUploadProgress(percent);
    };

    xhr.onprogress = () => {
      parseResponseChunks(false);
    };

    xhr.onerror = () => {
      if (settled) {
        return;
      }
      settled = true;
      reject(new Error('Ошибка сети при загрузке файла.'));
    };

    xhr.ontimeout = () => {
      if (settled) {
        return;
      }
      settled = true;
      reject(new Error('Сервер слишком долго обрабатывает загрузку файлов.'));
    };

    xhr.onload = () => {
      parseResponseChunks(true);
      if (settled) {
        return;
      }

      let payload = {};
      try {
        payload = JSON.parse(xhr.responseText || '{}');
      } catch (_error) {
        payload = {};
      }

      if (xhr.status < 200 || xhr.status >= 300 || payload.ok === false) {
        settled = true;
        reject(new Error(payload.error || 'Запрос к серверу завершился ошибкой.'));
        return;
      }

      settled = true;
      resolve(payload);
    };

    xhr.send(formData);
  });
}

export async function fetchDashboard(chatId = null, options = {}) {
  const query = chatId ? `?chatId=${encodeURIComponent(chatId)}` : '';
  const payload = await requestJson(`/api/dashboard${query}`, options);
  return payload.snapshot;
}

export async function fetchRecentLogs(limit = 120) {
  const payload = await requestJson(`/api/logs/recent?limit=${encodeURIComponent(limit)}`, {
    timeoutMs: 10000,
  });
  return payload.logs || [];
}

export async function fetchServerStatus() {
  const payload = await requestJson('/api/status', {
    timeoutMs: 12000,
  });
  return payload.status || null;
}

export function subscribeToServerEvents(handlers = {}) {
  const eventSource = new EventSource('/api/events');

  const parseEvent = (event, callback) => {
    if (typeof callback !== 'function') {
      return;
    }

    try {
      callback(JSON.parse(event.data || '{}'));
    } catch (_error) {
      // Ignore malformed realtime payloads.
    }
  };

  eventSource.onopen = () => {
    handlers.onOpen?.();
  };

  eventSource.onerror = () => {
    handlers.onConnectionError?.();
  };

  eventSource.addEventListener('snapshot', (event) => {
    parseEvent(event, handlers.onSnapshot);
  });
  eventSource.addEventListener('training_progress', (event) => {
    parseEvent(event, handlers.onTrainingProgress);
  });
  eventSource.addEventListener('log', (event) => {
    parseEvent(event, handlers.onLog);
  });
  eventSource.addEventListener('logs', (event) => {
    parseEvent(event, handlers.onLogs);
  });
  eventSource.addEventListener('server_error', (event) => {
    parseEvent(event, handlers.onServerError);
  });
  eventSource.addEventListener('status', (event) => {
    parseEvent(event, handlers.onStatus);
  });

  return () => {
    eventSource.close();
  };
}

export async function saveSettings(settings) {
  const payload = await requestJson('/api/settings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(settings),
  });
  return payload.snapshot;
}

export async function saveRuntimeConfig(runtimeConfig) {
  const payload = await requestJson('/api/runtime-config', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(runtimeConfig),
  });
  return payload.snapshot;
}

export async function uploadFiles(files, options = {}) {
  const { onUploadProgress, onProcessingProgress } = options;
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));

  const payload = await uploadFormDataWithProgress('/api/sources/files', formData, {
    onUploadProgress,
    onProcessingProgress,
  });

  return payload.snapshot;
}

export async function addUrlSource(url) {
  const payload = await requestJson('/api/sources/url', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ url }),
  });
  return payload.snapshot;
}

export async function removeSource(sourceId) {
  const payload = await requestJson(`/api/sources/${sourceId}`, {
    method: 'DELETE',
  });
  return payload.snapshot;
}

export async function clearSources() {
  const payload = await requestJson('/api/sources', {
    method: 'DELETE',
  });
  return payload.snapshot;
}

export async function createTrainingQueue(name = '') {
  const payload = await requestJson('/api/training-queues', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name }),
  });
  return payload.snapshot;
}

export async function uploadQueueFiles(queueId, files, options = {}) {
  const { onUploadProgress, onProcessingProgress } = options;
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));

  const payload = await uploadFormDataWithProgress(`/api/training-queues/${encodeURIComponent(queueId)}/files`, formData, {
    onUploadProgress,
    onProcessingProgress,
  });

  return payload.snapshot;
}

export async function removeTrainingQueueSource(queueId, sourceId) {
  const payload = await requestJson(
    `/api/training-queues/${encodeURIComponent(queueId)}/files/${encodeURIComponent(sourceId)}`,
    {
      method: 'DELETE',
    }
  );
  return payload.snapshot;
}

export async function deleteTrainingQueue(queueId) {
  const payload = await requestJson(`/api/training-queues/${encodeURIComponent(queueId)}`, {
    method: 'DELETE',
  });
  return payload.snapshot;
}

export async function createModel() {
  const payload = await requestJson('/api/model/create', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({}),
    timeoutMs: 15000,
  });
  return payload.snapshot;
}

export async function createNamedModel(name) {
  const payload = await requestJson('/api/model/create', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name }),
    timeoutMs: 15000,
  });
  return payload.snapshot;
}

export async function selectModel(modelId) {
  const payload = await requestJson(`/api/models/${encodeURIComponent(modelId)}/select`, {
    method: 'POST',
    timeoutMs: 25000,
  });
  return payload.snapshot;
}

export async function deleteLibraryModel(modelId) {
  const payload = await requestJson(`/api/models/${encodeURIComponent(modelId)}`, {
    method: 'DELETE',
    timeoutMs: 15000,
  });
  return payload.snapshot;
}

export async function trainModel() {
  const payload = await requestJson('/api/model/train', {
    method: 'POST',
    timeoutMs: 15000,
  });
  return payload.snapshot;
}

export async function pauseModel() {
  const payload = await requestJson('/api/model/pause', {
    method: 'POST',
    timeoutMs: 15000,
  });
  return payload.snapshot;
}

export async function resetModel() {
  const payload = await requestJson('/api/model/reset', {
    method: 'POST',
    timeoutMs: 15000,
  });
  return payload.snapshot;
}

export async function rollbackTrainingToCheckpoint() {
  const payload = await requestJson('/api/model/rollback', {
    method: 'POST',
    timeoutMs: 15000,
  });
  return payload.snapshot;
}

export async function exportModelPackage() {
  const response = await fetch('/api/model/export', {
    method: 'GET',
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || 'Не удалось экспортировать модель.');
  }

  const blob = await response.blob();
  const contentDisposition = response.headers.get('content-disposition') || '';
  const match = contentDisposition.match(/filename="?([^"]+)"?/i);
  const fileName = match?.[1] || 'model-export.aistudio.json';

  return { blob, fileName };
}

export async function importModelPackage(file) {
  if (!file) {
    throw new Error('Файл пакета модели не выбран.');
  }

  const formData = new FormData();
  formData.append('file', file);

  const payload = await requestJson('/api/model/import', {
    method: 'POST',
    body: formData,
    timeoutMs: 30000,
  });
  return payload.snapshot;
}

export async function createChat() {
  const payload = await requestJson('/api/chats', {
    method: 'POST',
  });
  return payload.snapshot;
}

export async function deleteChat(chatId) {
  const payload = await requestJson(`/api/chats/${chatId}`, {
    method: 'DELETE',
  });
  return payload.snapshot;
}

export async function sendChatMessage(chatId, content) {
  const payload = await requestJson(`/api/chats/${chatId}/messages`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ content }),
    timeoutMs: 120000,
  });
  return payload.snapshot;
}

export async function rateMessage(messageId, score) {
  const payload = await requestJson(`/api/messages/${messageId}/rating`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ score }),
  });
  return payload.snapshot;
}
