import { useEffect, useMemo, useRef, useState } from 'react';
import NotificationsRoundedIcon from '@mui/icons-material/NotificationsRounded';
import { Alert } from '@mui/material';

function joinClassNames(...values) {
  return values.filter(Boolean).join(' ');
}

export default function AppAlerts({
  processBanner = null,
  notices = [],
  bannerClassName = '',
  toastLayerClassName = '',
  toastClassName = '',
  resolvedStatusTheme,
}) {
  const [autoOpen, setAutoOpen] = useState(false);
  const previousNoticeKeysRef = useRef([]);

  const normalizedNotices = useMemo(() => ([
    ...(processBanner ? [{
      id: processBanner.id || 'process-banner',
      severity: processBanner.severity || 'info',
      tone: processBanner.tone || 'training',
      title: processBanner.title || '',
      message: processBanner.message || '',
      process: true,
    }] : []),
    ...notices,
  ]), [notices, processBanner]);

  const noticeCount = normalizedNotices.length;
  const noticeKeys = useMemo(
    () => normalizedNotices.map((notice) => `${notice.id}:${notice.severity}:${notice.title || ''}:${notice.message || ''}`),
    [normalizedNotices]
  );

  useEffect(() => {
    const previous = previousNoticeKeysRef.current;
    if (previous.length) {
      const hasNewNotice = noticeKeys.some((key) => !previous.includes(key));
      if (hasNewNotice) {
        setAutoOpen(true);
      }
    }
    previousNoticeKeysRef.current = noticeKeys;
  }, [noticeKeys]);

  useEffect(() => {
    if (!autoOpen) {
      return undefined;
    }
    const timeoutId = window.setTimeout(() => setAutoOpen(false), 3000);
    return () => window.clearTimeout(timeoutId);
  }, [autoOpen]);

  return (
    <div className={joinClassNames('app-toast-layer', toastLayerClassName, bannerClassName)} aria-live="polite" role="status">
      <div className={joinClassNames('app-toast-hub', autoOpen ? 'app-toast-hub--auto-open' : '')} role="presentation">
        <button
          type="button"
          className={joinClassNames('app-toast-hub__summary', resolvedStatusTheme)}
          aria-label={`Уведомлений: ${noticeCount}`}
        >
          <span className="app-toast-hub__icon" aria-hidden="true">
            <NotificationsRoundedIcon fontSize="inherit" />
          </span>
          <span className="app-toast-hub__count">{noticeCount}</span>
        </button>
        <div className={joinClassNames('app-toast-hub__list', noticeCount ? '' : 'app-toast-hub__list--empty')}>
          {normalizedNotices.map((notice) => (
            <Alert
              key={notice.id}
              severity={notice.severity || 'info'}
              className={joinClassNames('app-toast', notice.process ? `app-toast--${notice.tone || 'training'}` : '', toastClassName)}
              variant="filled"
            >
              {notice.title ? (
                <span className="app-toast__content">
                  <strong className="app-toast__title">{notice.title}</strong>
                  <span className="app-toast__message">{notice.message}</span>
                </span>
              ) : notice.message}
            </Alert>
          ))}
        </div>
      </div>
    </div>
  );
}
