import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  addUrlSource,
  clearSources,
  createChat,
  createChatShareLink,
  createApiModel,
  createModel,
  createNamedModel,
  createTrainingQueue,
  deleteChat,
  deleteLibraryModel,
  deleteTrainingQueue,
  exportModelPackage,
  fetchDashboard,
  fetchRecentLogs,
  fetchServerStatus,
  importModelPackage,
  pauseModel,
  rateMessage,
  removeSource,
  removeTrainingQueueSource,
  rollbackTrainingToCheckpoint,
  resetModel,
  saveRuntimeConfig,
  saveSettings,
  sendChatMessage,
  stopChatReply,
  selectModel,
  subscribeToServerEvents,
  trainModel,
  updateChatMessage,
  updateApiModel,
  uploadFiles,
  uploadQueueFiles,
} from '../api/studioApi';

const ACTIVE_POLL_STATUSES = new Set([
  'training',
  'generating_reply',
  'syncing_knowledge',
  'learning_from_feedback',
]);

const ACTIVE_POLL_LIFECYCLES = new Set([
  'training',
  'generating_reply',
  'syncing_knowledge',
]);

const FAST_REFRESH_ACTIONS = new Set([
  'trainModel',
  'pauseModel',
  'rollbackTraining',
  'importModel',
  'resetModel',
]);

const MAX_SERVER_LOGS = 160;

function mergeLogEntries(currentLogs, incomingLogs) {
  const nextLogs = [];
  const seenIds = new Set();

  [...incomingLogs, ...currentLogs].forEach((entry) => {
    if (!entry?.id || seenIds.has(entry.id)) {
      return;
    }

    seenIds.add(entry.id);
    nextLogs.push(entry);
  });

  return nextLogs.slice(0, MAX_SERVER_LOGS);
}

export function useStudioApp(options = {}) {
  const enabled = options.enabled !== false;
  const [snapshot, setSnapshot] = useState(null);
  const [selectedChatId, setSelectedChatId] = useState(null);
  const [requestState, setRequestState] = useState({
    loading: enabled,
    busy: false,
    error: '',
    action: '',
  });
  const [pendingReply, setPendingReply] = useState(null);
  const [pendingRatings, setPendingRatings] = useState({});
  const [uploadProgress, setUploadProgress] = useState(null);
  const [processingProgress, setProcessingProgress] = useState(null);
  const [serverLogs, setServerLogs] = useState([]);
  const [serverStatus, setServerStatus] = useState(null);
  const [realtimeConnected, setRealtimeConnected] = useState(false);

  const snapshotRef = useRef(null);
  const selectedChatIdRef = useRef(null);
  const dashboardAbortRef = useRef(null);
  const dashboardRequestIdRef = useRef(0);
  const followUpRefreshRef = useRef([]);

  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);

  useEffect(() => {
    selectedChatIdRef.current = selectedChatId;
  }, [selectedChatId]);

  const mergeRealtimeSnapshot = useCallback((nextRealtimeSnapshot) => {
    if (!nextRealtimeSnapshot) {
      return;
    }

    setSnapshot((current) => {
      if (!current) {
        return {
          ...nextRealtimeSnapshot,
          activeChat: null,
        };
      }

      return {
        ...current,
        ...nextRealtimeSnapshot,
        activeChat: current.activeChat,
      };
    });

    setSelectedChatId((currentSelectedChatId) => {
      if (!Array.isArray(nextRealtimeSnapshot.chats) || !nextRealtimeSnapshot.chats.length) {
        return currentSelectedChatId;
      }

      if (currentSelectedChatId && nextRealtimeSnapshot.chats.some((chat) => chat.id === currentSelectedChatId)) {
        return currentSelectedChatId;
      }

      return nextRealtimeSnapshot.chats[0]?.id || null;
    });

    setRequestState((current) => ({
      ...current,
      loading: false,
    }));
  }, []);

  const mergeTrainingProgressSnapshot = useCallback((nextTrainingSnapshot) => {
    if (!nextTrainingSnapshot) {
      return;
    }

    setSnapshot((current) => {
      if (!current) {
        return current;
      }

      return {
        ...current,
        model: {
          ...current.model,
          ...(nextTrainingSnapshot.model || {}),
        },
        training: {
          ...current.training,
          ...(nextTrainingSnapshot.training || {}),
        },
        knowledge: {
          ...current.knowledge,
          ...(nextTrainingSnapshot.knowledge || {}),
        },
        runtime: {
          ...current.runtime,
          ...(nextTrainingSnapshot.runtime || {}),
        },
      };
    });
  }, []);

  const loadDashboard = useCallback(async (chatId = selectedChatIdRef.current) => {
    if (!enabled) {
      return null;
    }

    const controller = new AbortController();
    const requestId = dashboardRequestIdRef.current + 1;

    dashboardRequestIdRef.current = requestId;
    if (dashboardAbortRef.current) {
      dashboardAbortRef.current.abort();
    }
    dashboardAbortRef.current = controller;

    try {
      const nextSnapshot = await fetchDashboard(chatId, {
        signal: controller.signal,
        timeoutMs: 15000,
      });

      if (dashboardRequestIdRef.current !== requestId) {
        return null;
      }

      setSnapshot(nextSnapshot);
      setSelectedChatId(nextSnapshot.activeChat?.id || nextSnapshot.chats?.[0]?.id || null);
      setRequestState((current) => ({
        ...current,
        loading: false,
        error: '',
      }));
      return nextSnapshot;
    } catch (error) {
      if (error?.name === 'AbortError') {
        return null;
      }

      setRequestState((current) => ({
        ...current,
        loading: false,
        error: error.message,
      }));
      return null;
    } finally {
      if (dashboardAbortRef.current === controller) {
        dashboardAbortRef.current = null;
      }
    }
  }, [enabled]);

  const scheduleDashboardRefresh = useCallback((delays, chatId = selectedChatIdRef.current) => {
    followUpRefreshRef.current.forEach((timeoutId) => clearTimeout(timeoutId));
    followUpRefreshRef.current = delays.map((delay) => setTimeout(() => {
      loadDashboard(chatId);
    }, delay));
  }, [loadDashboard]);

  useEffect(() => {
    if (!enabled) {
      setSnapshot(null);
      setSelectedChatId(null);
      setRequestState((current) => ({
        ...current,
        loading: false,
        busy: false,
        error: '',
        action: '',
      }));
      return () => {};
    }

    loadDashboard(null);

    return () => {
      dashboardAbortRef.current?.abort();
      followUpRefreshRef.current.forEach((timeoutId) => clearTimeout(timeoutId));
    };
  }, [enabled, loadDashboard]);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    if (!selectedChatId || !snapshotRef.current) {
      return;
    }

    if (snapshotRef.current.activeChat?.id === selectedChatId) {
      return;
    }

    loadDashboard(selectedChatId);
  }, [enabled, loadDashboard, selectedChatId]);

  useEffect(() => {
    if (!enabled) {
      return () => {};
    }

    let isDisposed = false;

    fetchRecentLogs()
      .then((logs) => {
        if (!isDisposed) {
          setServerLogs(logs);
        }
      })
      .catch(() => {
        // Ignore initial log bootstrap failures.
      });

    fetchServerStatus()
      .then((status) => {
        if (!isDisposed) {
          setServerStatus(status);
        }
      })
      .catch(() => {
        // Ignore initial status bootstrap failures.
      });

    const unsubscribe = subscribeToServerEvents({
      onOpen: () => {
        if (!isDisposed) {
          setRealtimeConnected(true);
        }
      },
      onConnectionError: () => {
        if (!isDisposed) {
          setRealtimeConnected(false);
        }
      },
      onSnapshot: ({ snapshot: nextRealtimeSnapshot }) => {
        if (!isDisposed) {
          mergeRealtimeSnapshot(nextRealtimeSnapshot);
        }
      },
      onTrainingProgress: ({ snapshot: nextTrainingSnapshot }) => {
        if (!isDisposed) {
          mergeTrainingProgressSnapshot(nextTrainingSnapshot);
        }
      },
      onLog: ({ entry }) => {
        if (!isDisposed && entry) {
          setServerLogs((current) => mergeLogEntries(current, [entry]));
        }
      },
      onLogs: ({ logs }) => {
        if (!isDisposed && Array.isArray(logs)) {
          setServerLogs(logs.slice(0, MAX_SERVER_LOGS));
        }
      },
      onServerError: ({ message }) => {
        if (!isDisposed && message) {
          setRequestState((current) => ({
            ...current,
            error: message,
          }));
        }
      },
      onStatus: ({ status }) => {
        if (!isDisposed && status) {
          setServerStatus(status);
        }
      },
    });

    return () => {
      isDisposed = true;
      unsubscribe();
    };
  }, [enabled, mergeRealtimeSnapshot, mergeTrainingProgressSnapshot]);

  useEffect(() => {
    if (!enabled) {
      return () => {};
    }

    const interval = setInterval(() => {
      fetchServerStatus()
        .then((status) => {
          setServerStatus(status);
        })
        .catch(() => {
          // Ignore intermittent status fetch failures.
        });
    }, realtimeConnected ? 14000 : 7000);

    return () => clearInterval(interval);
  }, [enabled, realtimeConnected]);

  useEffect(() => {
    if (!enabled) {
      return () => {};
    }

    if (!snapshotRef.current) {
      return undefined;
    }

    const shouldUseFastPolling =
      ACTIVE_POLL_STATUSES.has(snapshotRef.current?.training?.status) ||
      ACTIVE_POLL_LIFECYCLES.has(snapshotRef.current?.model?.lifecycle) ||
      FAST_REFRESH_ACTIONS.has(requestState.action);
    const intervalMs = shouldUseFastPolling
      ? (realtimeConnected ? 4000 : 1200)
      : (realtimeConnected ? 12000 : 6000);

    const interval = setInterval(() => {
      loadDashboard(selectedChatIdRef.current);
    }, intervalMs);

    return () => clearInterval(interval);
  }, [enabled, loadDashboard, realtimeConnected, requestState.action, snapshot?.model?.lifecycle, snapshot?.training?.status]);

  const runAction = useCallback(async (action, options = {}) => {
    if (!enabled) {
      return null;
    }

    const {
      actionName = '',
      onStart,
      onSuccess,
      onFinish,
    } = options;

    onStart?.();
    setRequestState((current) => ({
      ...current,
      busy: true,
      error: '',
      action: actionName,
    }));

    try {
      const nextSnapshot = await action();
      if (nextSnapshot) {
        setSnapshot(nextSnapshot);
        setSelectedChatId(nextSnapshot.activeChat?.id || nextSnapshot.chats?.[0]?.id || null);
      }
      await onSuccess?.(nextSnapshot);
      return nextSnapshot;
    } catch (error) {
      setRequestState((current) => ({
        ...current,
        error: error.message,
      }));
      return null;
    } finally {
      onFinish?.();
      setRequestState((current) => ({
        ...current,
        busy: false,
        action: '',
      }));
    }
  }, [enabled]);

  const actions = useMemo(() => ({
    refresh: () => loadDashboard(selectedChatIdRef.current),
    saveSettings: (nextSettings) => runAction(() => saveSettings(nextSettings), { actionName: 'saveSettings' }),
    saveRuntimeConfig: (nextRuntimeConfig) => runAction(
      () => saveRuntimeConfig(nextRuntimeConfig),
      { actionName: 'saveRuntimeConfig' }
    ),
    uploadFiles: (files) => runAction(
      () => uploadFiles(files, {
        onUploadProgress: (percent) => setUploadProgress(percent),
        onProcessingProgress: ({ percent }) => setProcessingProgress(percent),
      }),
      {
        actionName: 'uploadFiles',
        onStart: () => {
          setUploadProgress(0);
          setProcessingProgress(0);
        },
        onFinish: () => {
          setUploadProgress(null);
          setProcessingProgress(null);
        },
      }
    ),
    createTrainingQueue: (name) => runAction(
      () => createTrainingQueue(name),
      { actionName: 'createTrainingQueue' }
    ),
    uploadQueueFiles: (queueId, files) => runAction(
      () => uploadQueueFiles(queueId, files, {
        onUploadProgress: (percent) => setUploadProgress(percent),
        onProcessingProgress: ({ percent }) => setProcessingProgress(percent),
      }),
      {
        actionName: 'uploadQueueFiles',
        onStart: () => {
          setUploadProgress(0);
          setProcessingProgress(0);
        },
        onFinish: () => {
          setUploadProgress(null);
          setProcessingProgress(null);
        },
      }
    ),
    removeTrainingQueueSource: (queueId, sourceId) => runAction(
      () => removeTrainingQueueSource(queueId, sourceId),
      { actionName: 'removeTrainingQueueSource' }
    ),
    deleteTrainingQueue: (queueId) => runAction(
      () => deleteTrainingQueue(queueId),
      { actionName: 'deleteTrainingQueue' }
    ),
    addUrlSource: (url) => runAction(() => addUrlSource(url), { actionName: 'addUrlSource' }),
    clearSources: () => runAction(() => clearSources(), { actionName: 'clearSources' }),
    removeSource: (sourceId) => runAction(() => removeSource(sourceId), { actionName: 'removeSource' }),
    createModel: () => runAction(() => createModel(), { actionName: 'createModel' }),
    createNamedModel: (name) => runAction(() => createNamedModel(name), { actionName: 'createNamedModel' }),
    createApiModel: (apiModel) => runAction(() => createApiModel(apiModel), { actionName: 'createApiModel' }),
    updateApiModel: (modelId, apiModel) => runAction(() => updateApiModel(modelId, apiModel), { actionName: 'updateApiModel' }),
    selectModel: (modelId) => runAction(() => selectModel(modelId), { actionName: 'selectModel' }),
    deleteLibraryModel: (modelId) => runAction(() => deleteLibraryModel(modelId), { actionName: 'deleteLibraryModel' }),
    trainModel: () => runAction(() => trainModel(), {
      actionName: 'trainModel',
      onSuccess: async (nextSnapshot) => {
        const chatId = nextSnapshot?.activeChat?.id || selectedChatIdRef.current;
        await loadDashboard(chatId);
        scheduleDashboardRefresh([500, 1500, 3500], chatId);
      },
    }),
    pauseModel: () => runAction(() => pauseModel(), {
      actionName: 'pauseModel',
      onSuccess: async (nextSnapshot) => {
        const chatId = nextSnapshot?.activeChat?.id || selectedChatIdRef.current;
        await loadDashboard(chatId);
        scheduleDashboardRefresh([500, 1500], chatId);
      },
    }),
    resetModel: () => runAction(() => resetModel(), {
      actionName: 'resetModel',
      onSuccess: async (nextSnapshot) => {
        const chatId = nextSnapshot?.activeChat?.id || selectedChatIdRef.current;
        await loadDashboard(chatId);
      },
    }),
    rollbackTraining: () => runAction(() => rollbackTrainingToCheckpoint(), {
      actionName: 'rollbackTraining',
      onSuccess: async (nextSnapshot) => {
        const chatId = nextSnapshot?.activeChat?.id || selectedChatIdRef.current;
        await loadDashboard(chatId);
      },
    }),
    exportModel: async () => {
      if (!enabled) {
        return;
      }

      setRequestState((current) => ({
        ...current,
        busy: true,
        error: '',
        action: 'exportModel',
      }));

      try {
        if (snapshotRef.current?.model?.kind === 'api') {
          throw new Error('Экспорт доступен только для локальной модели. Выберите локальную модель в библиотеке.');
        }
        const { blob, fileName } = await exportModelPackage();
        const objectUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = objectUrl;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(objectUrl);
      } catch (error) {
        setRequestState((current) => ({
          ...current,
          error: error.message,
        }));
      } finally {
        setRequestState((current) => ({
          ...current,
          busy: false,
          action: '',
        }));
      }
    },
    importModel: (file) => runAction(() => importModelPackage(file), {
      actionName: 'importModel',
      onSuccess: async (nextSnapshot) => {
        const chatId = nextSnapshot?.activeChat?.id || selectedChatIdRef.current;
        await loadDashboard(chatId);
      },
    }),
    createChat: () => runAction(() => createChat(), { actionName: 'createChat' }),
    stopChatReply: async (chatId) => {
      if (!enabled) {
        return null;
      }

      const nextSnapshot = await stopChatReply(chatId);
      if (nextSnapshot) {
        setSnapshot(nextSnapshot);
        setSelectedChatId(nextSnapshot.activeChat?.id || nextSnapshot.chats?.[0]?.id || null);
      }
      return nextSnapshot;
    },
    createChatShareLink: async (chatId) => createChatShareLink(chatId),
    updateChatMessage: (messageId, content) => runAction(
      () => updateChatMessage(messageId, content),
      { actionName: 'updateChatMessage' }
    ),
    deleteChat: (chatId) => runAction(() => deleteChat(chatId), { actionName: 'deleteChat' }),
    sendMessage: (chatId, content) => runAction(
      () => sendChatMessage(chatId, content),
      {
        actionName: 'sendMessage',
        onStart: () => {
          setPendingReply({
            chatId,
            content,
            createdAt: new Date().toISOString(),
          });
        },
        onFinish: () => {
          setPendingReply(null);
        },
      }
    ),
    rateMessage: (messageId, score) => runAction(
      () => rateMessage(messageId, score),
      {
        actionName: 'rateMessage',
        onStart: () => {
          setPendingRatings((current) => ({
            ...current,
            [messageId]: score,
          }));
        },
        onFinish: () => {
          setPendingRatings((current) => {
            const nextState = { ...current };
            delete nextState[messageId];
            return nextState;
          });
        },
      }
    ),
  }), [enabled, loadDashboard, runAction, scheduleDashboardRefresh]);

  return {
    snapshot,
    selectedChatId,
    setSelectedChatId,
    loading: requestState.loading,
    busy: requestState.busy,
    error: requestState.error,
    pendingAction: requestState.action,
    uploadProgress,
    processingProgress,
    pendingReply,
    pendingRatings,
    realtimeConnected,
    serverLogs,
    serverStatus,
    actions,
  };
}
