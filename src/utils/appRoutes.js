export const APP_TAB_KEYS = {
  MODELS: 'models',
  CHATS: 'chats',
  PROJECT: 'project',
  ACCOUNT: 'account',
};

export const APP_TABS = [
  APP_TAB_KEYS.MODELS,
  APP_TAB_KEYS.CHATS,
  APP_TAB_KEYS.PROJECT,
  APP_TAB_KEYS.ACCOUNT,
];

export function getTabIndexByKey(tabKey) {
  const index = APP_TABS.indexOf(tabKey);
  return index >= 0 ? index : 0;
}

export function getTabKeyByIndex(index) {
  return APP_TABS[index] || APP_TAB_KEYS.MODELS;
}

export function buildTabPath(tabKey, options = {}) {
  const normalizedTabKey = APP_TABS.includes(tabKey) ? tabKey : APP_TAB_KEYS.MODELS;
  if (normalizedTabKey === APP_TAB_KEYS.CHATS && options.chatId) {
    return `/chats/${encodeURIComponent(options.chatId)}`;
  }
  return `/${normalizedTabKey}`;
}

export function buildSharedChatPath(token) {
  return `/shared/chat/${encodeURIComponent(String(token || '').trim())}`;
}

export function resolveAppRoute(pathname = '/') {
  const normalizedPath = String(pathname || '/').split('?')[0].split('#')[0] || '/';
  const segments = normalizedPath.replace(/^\/+/u, '').split('/').filter(Boolean);

  if (!segments.length) {
    return {
      type: 'tab',
      tabKey: APP_TAB_KEYS.MODELS,
      chatId: null,
      token: null,
      path: buildTabPath(APP_TAB_KEYS.MODELS),
    };
  }

  if (segments[0] === 'shared' && segments[1] === 'chat' && segments[2]) {
    return {
      type: 'shared-chat',
      tabKey: null,
      chatId: null,
      token: decodeURIComponent(segments[2]),
      path: normalizedPath,
    };
  }

  if (segments[0] === 'chats' && segments[1]) {
    return {
      type: 'chat',
      tabKey: APP_TAB_KEYS.CHATS,
      chatId: decodeURIComponent(segments[1]),
      token: null,
      path: normalizedPath,
    };
  }

  if (APP_TABS.includes(segments[0])) {
    return {
      type: 'tab',
      tabKey: segments[0],
      chatId: null,
      token: null,
      path: normalizedPath,
    };
  }

  return {
    type: 'tab',
    tabKey: APP_TAB_KEYS.MODELS,
    chatId: null,
    token: null,
    path: buildTabPath(APP_TAB_KEYS.MODELS),
  };
}
