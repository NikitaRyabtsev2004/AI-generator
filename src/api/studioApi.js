async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));

  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || 'Запрос к серверу завершился ошибкой.');
  }

  return payload;
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
          } catch (error) {
            // Ignore malformed incremental chunks and continue parsing.
          }
        });

      if (isFinal && pendingLine.trim()) {
        try {
          processMessage(JSON.parse(pendingLine.trim()));
        } catch (error) {
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

    xhr.onload = () => {
      parseResponseChunks(true);
      if (settled) {
        return;
      }

      let payload = {};
      try {
        payload = JSON.parse(xhr.responseText || '{}');
      } catch (error) {
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

export async function createModel() {
  const payload = await requestJson('/api/model/create', {
    method: 'POST',
  });
  return payload.snapshot;
}

export async function trainModel() {
  const payload = await requestJson('/api/model/train', {
    method: 'POST',
  });
  return payload.snapshot;
}

export async function pauseModel() {
  const payload = await requestJson('/api/model/pause', {
    method: 'POST',
  });
  return payload.snapshot;
}

export async function resetModel() {
  const payload = await requestJson('/api/model/reset', {
    method: 'POST',
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
