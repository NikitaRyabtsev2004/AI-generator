async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));

  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || 'Server request failed.');
  }

  return payload;
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

export async function uploadFiles(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));

  const payload = await requestJson('/api/sources/files', {
    method: 'POST',
    body: formData,
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
