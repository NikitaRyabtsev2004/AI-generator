import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  addUrlSource,
  createChat,
  createModel,
  deleteChat,
  fetchDashboard,
  pauseModel,
  rateMessage,
  removeSource,
  resetModel,
  saveSettings,
  sendChatMessage,
  trainModel,
  uploadFiles,
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
  'resetModel',
]);

export function useStudioApp() {
  const [snapshot, setSnapshot] = useState(null);
  const [selectedChatId, setSelectedChatId] = useState(null);
  const [requestState, setRequestState] = useState({
    loading: true,
    busy: false,
    error: '',
    action: '',
  });
  const [pendingReply, setPendingReply] = useState(null);
  const [pendingRatings, setPendingRatings] = useState({});
  const [uploadProgress, setUploadProgress] = useState(null);
  const [processingProgress, setProcessingProgress] = useState(null);

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

  const loadDashboard = useCallback(async (chatId = selectedChatIdRef.current) => {
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
  }, []);

  const scheduleDashboardRefresh = useCallback((delays, chatId = selectedChatIdRef.current) => {
    followUpRefreshRef.current.forEach((timeoutId) => clearTimeout(timeoutId));
    followUpRefreshRef.current = delays.map((delay) => setTimeout(() => {
      loadDashboard(chatId);
    }, delay));
  }, [loadDashboard]);

  useEffect(() => {
    loadDashboard(null);

    return () => {
      dashboardAbortRef.current?.abort();
      followUpRefreshRef.current.forEach((timeoutId) => clearTimeout(timeoutId));
    };
  }, [loadDashboard]);

  useEffect(() => {
    if (!selectedChatId || !snapshotRef.current) {
      return;
    }

    if (snapshotRef.current.activeChat?.id === selectedChatId) {
      return;
    }

    loadDashboard(selectedChatId);
  }, [loadDashboard, selectedChatId]);

  useEffect(() => {
    if (!snapshotRef.current) {
      return undefined;
    }

    const shouldUseFastPolling =
      ACTIVE_POLL_STATUSES.has(snapshotRef.current?.training?.status) ||
      ACTIVE_POLL_LIFECYCLES.has(snapshotRef.current?.model?.lifecycle) ||
      FAST_REFRESH_ACTIONS.has(requestState.action);

    const interval = setInterval(() => {
      loadDashboard(selectedChatIdRef.current);
    }, shouldUseFastPolling ? 1200 : 6000);

    return () => clearInterval(interval);
  }, [loadDashboard, requestState.action, snapshot?.model?.lifecycle, snapshot?.training?.status]);

  const runAction = useCallback(async (action, options = {}) => {
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
      setSnapshot(nextSnapshot);
      setSelectedChatId(nextSnapshot.activeChat?.id || nextSnapshot.chats?.[0]?.id || null);
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
  }, []);

  const actions = useMemo(() => ({
    refresh: () => loadDashboard(selectedChatIdRef.current),
    saveSettings: (nextSettings) => runAction(() => saveSettings(nextSettings), { actionName: 'saveSettings' }),
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
    addUrlSource: (url) => runAction(() => addUrlSource(url), { actionName: 'addUrlSource' }),
    removeSource: (sourceId) => runAction(() => removeSource(sourceId), { actionName: 'removeSource' }),
    createModel: () => runAction(() => createModel(), { actionName: 'createModel' }),
    trainModel: () => runAction(() => trainModel(), {
      actionName: 'trainModel',
      onSuccess: async (nextSnapshot) => {
        const chatId = nextSnapshot.activeChat?.id || selectedChatIdRef.current;
        await loadDashboard(chatId);
        scheduleDashboardRefresh([500, 1500, 3500], chatId);
      },
    }),
    pauseModel: () => runAction(() => pauseModel(), {
      actionName: 'pauseModel',
      onSuccess: async (nextSnapshot) => {
        const chatId = nextSnapshot.activeChat?.id || selectedChatIdRef.current;
        await loadDashboard(chatId);
        scheduleDashboardRefresh([500, 1500], chatId);
      },
    }),
    resetModel: () => runAction(() => resetModel(), {
      actionName: 'resetModel',
      onSuccess: async (nextSnapshot) => {
        const chatId = nextSnapshot.activeChat?.id || selectedChatIdRef.current;
        await loadDashboard(chatId);
      },
    }),
    createChat: () => runAction(() => createChat(), { actionName: 'createChat' }),
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
  }), [loadDashboard, runAction, scheduleDashboardRefresh]);

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
    actions,
  };
}
