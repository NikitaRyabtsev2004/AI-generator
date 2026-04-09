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

export function useStudioApp() {
  const [snapshot, setSnapshot] = useState(null);
  const [selectedChatId, setSelectedChatId] = useState(null);
  const [requestState, setRequestState] = useState({
    loading: true,
    busy: false,
    error: '',
  });

  const snapshotRef = useRef(null);
  const selectedChatIdRef = useRef(null);
  const dashboardAbortRef = useRef(null);
  const dashboardRequestIdRef = useRef(0);

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

  useEffect(() => {
    loadDashboard(null);

    return () => {
      dashboardAbortRef.current?.abort();
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

    const interval = setInterval(() => {
      loadDashboard(selectedChatIdRef.current);
    }, ACTIVE_POLL_STATUSES.has(snapshotRef.current?.training?.status) ? 1800 : 6000);

    return () => clearInterval(interval);
  }, [loadDashboard, snapshot?.training?.status]);

  const runAction = useCallback(async (action) => {
    setRequestState((current) => ({
      ...current,
      busy: true,
      error: '',
    }));

    try {
      const nextSnapshot = await action();
      setSnapshot(nextSnapshot);
      setSelectedChatId(nextSnapshot.activeChat?.id || nextSnapshot.chats?.[0]?.id || null);
      return nextSnapshot;
    } catch (error) {
      setRequestState((current) => ({
        ...current,
        error: error.message,
      }));
      return null;
    } finally {
      setRequestState((current) => ({
        ...current,
        busy: false,
      }));
    }
  }, []);

  const actions = useMemo(() => ({
    refresh: () => loadDashboard(selectedChatIdRef.current),
    saveSettings: (nextSettings) => runAction(() => saveSettings(nextSettings)),
    uploadFiles: (files) => runAction(() => uploadFiles(files)),
    addUrlSource: (url) => runAction(() => addUrlSource(url)),
    removeSource: (sourceId) => runAction(() => removeSource(sourceId)),
    createModel: () => runAction(() => createModel()),
    trainModel: () => runAction(() => trainModel()),
    pauseModel: () => runAction(() => pauseModel()),
    resetModel: () => runAction(() => resetModel()),
    createChat: () => runAction(() => createChat()),
    deleteChat: (chatId) => runAction(() => deleteChat(chatId)),
    sendMessage: (chatId, content) => runAction(() => sendChatMessage(chatId, content)),
    rateMessage: (messageId, score) => runAction(() => rateMessage(messageId, score)),
  }), [loadDashboard, runAction]);

  return {
    snapshot,
    selectedChatId,
    setSelectedChatId,
    loading: requestState.loading,
    busy: requestState.busy,
    error: requestState.error,
    actions,
  };
}
