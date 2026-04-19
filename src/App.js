import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {
    Alert,
    Badge,
    Box,
    Button,
    Chip,
    CircularProgress,
    Container,
    IconButton,
    Tab,
    Tabs,
    TextField,
    Tooltip,
    Typography,
} from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ForumIcon from '@mui/icons-material/Forum';
import HistoryRoundedIcon from '@mui/icons-material/HistoryRounded';
import HubRoundedIcon from '@mui/icons-material/HubRounded';
import MemoryIcon from '@mui/icons-material/Memory';
import MenuRoundedIcon from '@mui/icons-material/MenuRounded';
import ChatTab from './components/chat/ChatTab';
import SharedChatView from './components/chat/SharedChatView';
import AppAlerts from './components/shared/AppAlerts';
import GlassPanel from './components/shared/GlassPanel';
import StatusPill from './components/shared/StatusPill';
import TrainingTab from './components/training/TrainingTab';
import {
    confirmPasswordReset,
    fetchAuthSession,
    loginWithPassword,
    logoutSession,
    startPasswordReset,
    startRegistration,
    verifyRegistration,
} from './api/studioApi';
import {useStudioApp} from './hooks/useStudioApp';
import {
    APP_TAB_KEYS,
    buildTabPath,
    getTabIndexByKey,
    getTabKeyByIndex,
    resolveAppRoute,
} from './utils/appRoutes';
import {formatNumber} from './utils/text';
import './styles/global.css';
import './styles/app-shell.css';

const backgroundImg = 'b-17.webp';

function resolveStatusTheme(snapshot) {
    if (!snapshot) {
        return 'not_created';
    }

    const modelExists = Boolean(snapshot.model?.exists);
    const modelKind = String(snapshot.model?.kind || 'local').toLowerCase();
    const isApiModel = modelKind === 'api';
    const trainedEpochs = Number(snapshot.model?.trainedEpochs || 0);
    const apiReady = Boolean(
        modelExists &&
        String(snapshot.model?.externalEndpoint || '').trim() &&
        String(snapshot.model?.externalModelName || '').trim()
    );
    const trainingStatus = String(snapshot.training?.status || '').toLowerCase();
    if (['training', 'paused', 'error'].includes(trainingStatus)) {
        return trainingStatus;
    }

    const lifecycle = String(snapshot.model?.lifecycle || '').toLowerCase();
    if (['training', 'paused', 'generating_reply', 'syncing_knowledge', 'learning_from_feedback', 'error'].includes(lifecycle)) {
        return lifecycle;
    }

    if (!modelExists) {
        return 'not_created';
    }

    if (lifecycle === 'trained') {
        return 'trained';
    }

    if (lifecycle === 'ready_for_training') {
        return isApiModel && apiReady ? 'trained' : 'ready_for_training';
    }

    if (isApiModel && apiReady) {
        return 'trained';
    }

    if (!trainedEpochs) {
        return 'ready_for_training';
    }

    return 'trained';
}

function buildProcessBanner(snapshot) {
    if (!snapshot) {
        return null;
    }

    const trainingStatus = String(snapshot.training?.status || '').toLowerCase();
    const lifecycle = String(snapshot.model?.lifecycle || '').toLowerCase();
    const message = snapshot.training?.message || '';

    if (trainingStatus === 'training' || lifecycle === 'training') {
        return {
            id: 'process-training',
            tone: 'training',
            title: 'Идет обучение модели',
            message: message || 'Процесс обучения запущен в воркере.',
        };
    }

    if (lifecycle === 'generating_reply') {
        return {
            id: 'process-generating-reply',
            tone: 'generating_reply',
            title: 'Генерация ответа',
            message: message || 'Сервер формирует ответ по текущему контексту.',
        };
    }

    if (lifecycle === 'syncing_knowledge') {
        return {
            id: 'process-syncing',
            tone: 'syncing_knowledge',
            title: 'Синхронизация знаний',
            message: message || 'Индекс и артефакты знаний обновляются.',
        };
    }

    if (lifecycle === 'learning_from_feedback') {
        return {
            id: 'process-feedback',
            tone: 'learning_from_feedback',
            title: 'Анализ обратной связи',
            message: message || 'Система обновляет внутренние сигналы качества.',
        };
    }

    return null;
}

function LoadingState() {
    return (
        <div className="loading-state">
            <CircularProgress/>
            <Typography variant="body1">Загрузка...</Typography>
        </div>
    );
}

function isStrongPassword(value = '') {
    const password = String(value || '');
    if (password.length < 12) {
        return false;
    }
    if (!/^[A-Za-z0-9!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]+$/u.test(password)) {
        return false;
    }
    return /[A-Z]/u.test(password) && /[a-z]/u.test(password) && /\d/u.test(password) && /[^A-Za-z0-9]/u.test(password);
}

function AuthView({onAuthenticated}) {
    const [mode, setMode] = useState('login');
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const [info, setInfo] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [passwordConfirm, setPasswordConfirm] = useState('');
    const [code, setCode] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [pendingEmail, setPendingEmail] = useState('');

    const runAuthAction = async (action) => {
        setBusy(true);
        setError('');
        try {
            await action();
        } catch (authError) {
            setError(authError.message || 'Не удалось выполнить действие авторизации.');
        } finally {
            setBusy(false);
        }
    };

    const handleLogin = () => runAuthAction(async () => {
        if (!email.trim() || !password) {
            throw new Error('Введите email и пароль.');
        }
        const user = await loginWithPassword(email.trim(), password);
        onAuthenticated?.(user || null);
    });

    const handleRegisterStart = () => runAuthAction(async () => {
        if (!email.trim() || !password) {
            throw new Error('Введите email и пароль.');
        }
        if (password !== passwordConfirm) {
            throw new Error('Пароли не совпадают.');
        }
        if (!isStrongPassword(password)) {
            throw new Error('Пароль должен быть минимум 12 символов и включать A-Z, a-z, цифру и спецсимвол.');
        }

        await startRegistration(email.trim(), password);
        setPendingEmail(email.trim());
        setCode('');
        setMode('verify');
        setInfo('Код отправлен на email. Введите его для активации аккаунта.');
    });

    const handleVerify = () => runAuthAction(async () => {
        const targetEmail = pendingEmail || email.trim();
        if (!targetEmail || !code.trim()) {
            throw new Error('Введите email и код подтверждения.');
        }
        const user = await verifyRegistration(targetEmail, code.trim());
        onAuthenticated?.(user || null);
    });

    const handleResetStart = () => runAuthAction(async () => {
        if (!email.trim()) {
            throw new Error('Введите email.');
        }
        await startPasswordReset(email.trim());
        setPendingEmail(email.trim());
        setCode('');
        setNewPassword('');
        setMode('reset_confirm');
        setInfo('Если аккаунт существует, код отправлен. Введите код и новый пароль.');
    });

    const handleResetConfirm = () => runAuthAction(async () => {
        const targetEmail = pendingEmail || email.trim();
        if (!targetEmail || !code.trim() || !newPassword) {
            throw new Error('Заполните email, код и новый пароль.');
        }
        if (!isStrongPassword(newPassword)) {
            throw new Error('Новый пароль не проходит требования безопасности.');
        }
        const user = await confirmPasswordReset(targetEmail, code.trim(), newPassword);
        onAuthenticated?.(user || null);
    });

    const modeTitle = {
        login: 'Вход',
        register: 'Регистрация',
        verify: 'Подтверждение email',
        reset_request: 'Сброс пароля',
        reset_confirm: 'Подтвердите сброс',
    }[mode];

    return (
        <Box
            className="app-shell app-shell--status-not_created"
            style={{'--app-bg-image': `url(${process.env.PUBLIC_URL}/${backgroundImg})`}}
        >
            <div
                aria-hidden="true"
                className="app-shell__status-layer app-shell__status-layer--current app-shell__status-layer--not_created"
            />
            <Container maxWidth="sm" className="app-shell__container" sx={{display: 'grid', placeItems: 'center'}}>
                <GlassPanel className="hero-panel">
                    <div className="hero-panel__topline" style={{justifyContent: 'center'}}>
                        <div className="hero-panel__title-wrap">
                            <div className="hero-panel__badge">
                                <AutoAwesomeIcon fontSize="medium"/>
                                <span className="hero-panel__badge-text">
                  <strong className="hero-panel__brand">AI Generator</strong>
                  <strong className="hero-panel__brand-accent">Studio</strong>
                </span>
                            </div>
                        </div>
                    </div>
                    <GlassPanel className="panel--fit" filterStyle={{borderRadius: "28px"}}>
                        <Typography variant="h3" sx={{mb: 1.2}}>{modeTitle}</Typography>
                        <Typography variant="body2" className="muted-text" sx={{mb: 1.2}}>
                            После подтверждения кода с почты откроется доступ к чатам, моделям и обучению.
                        </Typography>

                        {error ? <Alert severity="error" sx={{mb: 1.1}}>{error}</Alert> : null}
                        {info ? <Alert severity="info" sx={{mb: 1.1}}>{info}</Alert> : null}

                        {mode === 'login' ? (
                            <>
                                <TextField
                                    className="text-input"
                                    label="Email"
                                    value={email}
                                    autoComplete="username"
                                    onChange={(event) => setEmail(event.target.value)}
                                />
                                <TextField
                                    className="text-input"
                                    label="Пароль"
                                    type="password"
                                    value={password}
                                    autoComplete="current-password"
                                    onChange={(event) => setPassword(event.target.value)}
                                    sx={{mt: 1.1}}
                                />
                                <div style={{display: 'flex', gap: 10, marginTop: 14, flexWrap: 'wrap'}}>
                                    <Button variant="contained" color='inherit' onClick={handleLogin} disabled={busy}>
                                        {busy ? 'Вход...' : 'Войти'}
                                    </Button>
                                    <Button variant="outlined" onClick={() => {
                                        setMode('register');
                                        setError('');
                                        setInfo('');
                                    }} disabled={busy}>
                                        Регистрация
                                    </Button>
                                    <Button variant="text" onClick={() => {
                                        setMode('reset_request');
                                        setError('');
                                        setInfo('');
                                    }} disabled={busy}>
                                        Забыл пароль
                                    </Button>
                                </div>
                            </>
                        ) : null}

                        {mode === 'register' ? (
                            <>
                                <TextField
                                    className="text-input"
                                    label="Email"
                                    value={email}
                                    autoComplete="username"
                                    onChange={(event) => setEmail(event.target.value)}
                                />
                                <TextField
                                    className="text-input"
                                    label="Пароль"
                                    type="password"
                                    value={password}
                                    autoComplete="new-password"
                                    onChange={(event) => setPassword(event.target.value)}
                                    sx={{mt: 1.1}}
                                />
                                <TextField
                                    className="text-input"
                                    label="Повторите пароль"
                                    type="password"
                                    value={passwordConfirm}
                                    autoComplete="new-password"
                                    onChange={(event) => setPasswordConfirm(event.target.value)}
                                    sx={{mt: 1.1}}
                                />
                                <Typography variant="caption" className="muted-text" sx={{mt: 1, display: 'block'}}>
                                    Минимум 12 символов: A-Z, a-z, цифры и спецсимволы (только английская раскладка).
                                </Typography>
                                <div style={{display: 'flex', gap: 10, marginTop: 14, flexWrap: 'wrap'}}>
                                    <Button variant="contained" color='inherit' onClick={handleRegisterStart}
                                            disabled={busy}>
                                        {busy ? 'Отправка...' : 'Получить код'}
                                    </Button>
                                    <Button variant="text" onClick={() => {
                                        setMode('login');
                                        setError('');
                                        setInfo('');
                                    }} disabled={busy}>
                                        Назад ко входу
                                    </Button>
                                </div>
                            </>
                        ) : null}

                        {mode === 'verify' ? (
                            <>
                                <TextField
                                    className="text-input"
                                    label="Email"
                                    value={pendingEmail || email}
                                    onChange={(event) => {
                                        setPendingEmail(event.target.value);
                                        setEmail(event.target.value);
                                    }}
                                />
                                <TextField
                                    className="text-input"
                                    label="Код из письма"
                                    value={code}
                                    autoComplete="one-time-code"
                                    onChange={(event) => setCode(event.target.value)}
                                    sx={{mt: 1.1}}
                                />
                                <div style={{display: 'flex', gap: 10, marginTop: 14, flexWrap: 'wrap'}}>
                                    <Button variant="contained" onClick={handleVerify} disabled={busy}>
                                        {busy ? 'Проверка...' : 'Подтвердить'}
                                    </Button>
                                    <Button variant="text" onClick={() => {
                                        setMode('register');
                                        setError('');
                                        setInfo('');
                                    }} disabled={busy}>
                                        Изменить данные
                                    </Button>
                                </div>
                            </>
                        ) : null}

                        {mode === 'reset_request' ? (
                            <>
                                <TextField
                                    className="text-input"
                                    label="Email"
                                    value={email}
                                    autoComplete="username"
                                    onChange={(event) => setEmail(event.target.value)}
                                />
                                <div style={{display: 'flex', gap: 10, marginTop: 14, flexWrap: 'wrap'}}>
                                    <Button variant="contained" color='inherit' onClick={handleResetStart}
                                            disabled={busy}>
                                        {busy ? 'Отправка...' : 'Отправить код'}
                                    </Button>
                                    <Button variant="text" onClick={() => {
                                        setMode('login');
                                        setError('');
                                        setInfo('');
                                    }} disabled={busy}>
                                        Назад ко входу
                                    </Button>
                                </div>
                            </>
                        ) : null}

                        {mode === 'reset_confirm' ? (
                            <>
                                <TextField
                                    className="text-input"
                                    label="Email"
                                    value={pendingEmail || email}
                                    onChange={(event) => {
                                        setPendingEmail(event.target.value);
                                        setEmail(event.target.value);
                                    }}
                                />
                                <TextField
                                    className="text-input"
                                    label="Код из письма"
                                    value={code}
                                    autoComplete="one-time-code"
                                    onChange={(event) => setCode(event.target.value)}
                                    sx={{mt: 1.1}}
                                />
                                <TextField
                                    className="text-input"
                                    label="Новый пароль"
                                    type="password"
                                    value={newPassword}
                                    autoComplete="new-password"
                                    onChange={(event) => setNewPassword(event.target.value)}
                                    sx={{mt: 1.1}}
                                />
                                <div style={{display: 'flex', gap: 10, marginTop: 14, flexWrap: 'wrap'}}>
                                    <Button variant="contained" onClick={handleResetConfirm} disabled={busy}>
                                        {busy ? 'Сброс...' : 'Сбросить пароль'}
                                    </Button>
                                    <Button variant="text" onClick={() => {
                                        setMode('login');
                                        setError('');
                                        setInfo('');
                                    }} disabled={busy}>
                                        Назад ко входу
                                    </Button>
                                </div>
                            </>
                        ) : null}
                    </GlassPanel>
                </GlassPanel>
            </Container>
        </Box>
    );
}

function AccountTab({authUser, snapshot, onAuthenticated, onLogout}) {
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const [info, setInfo] = useState('');
    const [resetCode, setResetCode] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [resetRequested, setResetRequested] = useState(false);

    const runAction = async (action) => {
        setBusy(true);
        setError('');
        setInfo('');
        try {
            await action();
        } catch (actionError) {
            setError(actionError.message || 'Не удалось выполнить действие аккаунта.');
        } finally {
            setBusy(false);
        }
    };

    const handleStartPasswordReset = () => runAction(async () => {
        await startPasswordReset(authUser?.email || '');
        setResetRequested(true);
        setResetCode('');
        setNewPassword('');
        setInfo('Код для сброса пароля отправлен на вашу почту.');
    });

    const handleConfirmPasswordReset = () => runAction(async () => {
        if (!resetCode.trim() || !newPassword) {
            throw new Error('Введите код из письма и новый пароль.');
        }
        if (!isStrongPassword(newPassword)) {
            throw new Error('Новый пароль должен быть не короче 12 символов и содержать A-Z, a-z, цифру и спецсимвол.');
        }
        const nextUser = await confirmPasswordReset(authUser?.email || '', resetCode.trim(), newPassword);
        setResetRequested(false);
        setResetCode('');
        setNewPassword('');
        setInfo('Пароль обновлен, сессия подтверждена.');
        onAuthenticated?.(nextUser || authUser);
    });

    return (
        <div className="account-layout">
            <div className="main-panel--plain panel--fit">
                <Typography variant="h3">Аккаунт</Typography>
                <Typography variant="body2" className="muted-text">
                    Здесь собраны данные профиля, состояние рабочей области и операции доступа.
                </Typography>

                {error ? <Alert severity="error" sx={{mt: 1.2}}>{error}</Alert> : null}
                {info ? <Alert severity="info" sx={{mt: 1.2}}>{info}</Alert> : null}

                <div className="account-card-grid">
                    <GlassPanel
                        innerStyle={{padding: 0}}
                        style={{overflow: 'visible !important'}}
                        filterStyle={{borderRadius: '24px'}}
                    >
                        <div className="account-card">
                            <Typography variant="subtitle2">Профиль</Typography>
                            <div className="account-details">
                                <div className="fact-row">
                                    <span className="fact-row__label">Email</span>
                                    <span className="fact-row__value">{authUser?.email || '—'}</span>
                                </div>
                                <div className="fact-row">
                                    <span className="fact-row__label">ID аккаунта</span>
                                    <span className="fact-row__value">{authUser?.id || '—'}</span>
                                </div>
                                <div className="fact-row">
                                    <span className="fact-row__label">Статус</span>
                                    <span className="fact-row__value">
                  <StatusPill label={authUser?.verified ? 'email verified' : 'verification required'}
                              active={Boolean(authUser?.verified)} tone={authUser?.verified ? 'accent' : 'neutral'}/>
                </span>
                                </div>
                                <div className="fact-row">
                                    <span className="fact-row__label">Создан</span>
                                    <span
                                        className="fact-row__value">{authUser?.createdAt ? new Date(authUser.createdAt).toLocaleString() : '—'}</span>
                                </div>
                                <div className="fact-row">
                                    <span className="fact-row__label">Последний вход</span>
                                    <span
                                        className="fact-row__value">{authUser?.lastLoginAt ? new Date(authUser.lastLoginAt).toLocaleString() : '—'}</span>
                                </div>
                            </div>
                        </div>
                    </GlassPanel>

                    <GlassPanel
                        innerStyle={{padding: 0}}
                        style={{overflow: 'visible !important'}}
                        filterStyle={{borderRadius: '24px'}}
                    >
                        <div className="account-card">
                            <Typography variant="subtitle2">Рабочая область</Typography>
                            <div className="account-stats-grid">
                                <div className="metric-card">
                                    <div
                                        className="metric-card__value">{formatNumber(snapshot?.chats?.length || 0)}</div>
                                    <div className="metric-card__label">чатов</div>
                                </div>
                                <div className="metric-card">
                                    <div
                                        className="metric-card__value">{formatNumber(snapshot?.modelRegistry?.items?.length || 0)}</div>
                                    <div className="metric-card__label">моделей</div>
                                </div>
                                <div className="metric-card">
                                    <div
                                        className="metric-card__value">{formatNumber(snapshot?.sources?.length || 0)}</div>
                                    <div className="metric-card__label">источников</div>
                                </div>
                                <div className="metric-card">
                                    <div
                                        className="metric-card__value">{formatNumber(snapshot?.model?.trainedEpochs || 0)}</div>
                                    <div className="metric-card__label">эпох</div>
                                </div>
                            </div>
                        </div>
                    </GlassPanel>
                </div>
            </div>

            <GlassPanel
                style={{overflow: 'visible'}}
                filterStyle={{borderRadius: '24px'}}
                className="panel--fit"
            >
                <Typography variant="h3">Безопасность</Typography>
                <Typography variant="body2" className="muted-text">
                    Сброс пароля подтверждается кодом из письма. Требования к новому паролю такие же, как при
                    регистрации.
                </Typography>

                <div className="account-actions">
                    <Button variant="contained" color="inherit" onClick={handleStartPasswordReset} disabled={busy}>
                        {busy ? 'Отправка...' : 'Сбросить пароль'}
                    </Button>
                    <Button variant="outlined" color="inherit" onClick={onLogout} disabled={busy}>
                        Выйти из аккаунта
                    </Button>
                </div>

                {resetRequested ? (
                    <div className="account-reset-grid">
                        <TextField
                            className="text-input"
                            label="Код из письма"
                            value={resetCode}
                            autoComplete="one-time-code"
                            onChange={(event) => setResetCode(event.target.value)}
                        />
                        <TextField
                            className="text-input"
                            label="Новый пароль"
                            type="password"
                            value={newPassword}
                            autoComplete="new-password"
                            onChange={(event) => setNewPassword(event.target.value)}
                        />
                        <Button variant="contained" onClick={handleConfirmPasswordReset} disabled={busy}>
                            {busy ? 'Проверка...' : 'Подтвердить сброс'}
                        </Button>
                    </div>
                ) : null}
            </GlassPanel>
        </div>
    );
}

function statusToMuiColor(status) {
    switch (status) {
        case 'trained':
            return 'success';
        case 'training':
        case 'learning_from_feedback':
            return 'warning';
        case 'error':
            return 'error';
        case 'ready_for_training':
        case 'syncing_knowledge':
        case 'generating_reply':
            return 'info';
        case 'paused':
            return 'default';
        default:
            return 'default';
    }
}

function ProjectTab({snapshot, resolvedStatusTheme}) {
    const modelName = snapshot?.model?.name || snapshot?.model?.engine || 'Локальная модель';
    const modelKind = String(snapshot?.model?.kind || 'local').toLowerCase() === 'api'
        ? 'API-модель'
        : 'Локальная модель';
    const statusColor = statusToMuiColor(resolvedStatusTheme);

    return (
        <div className="account-tab">
            <GlassPanel
                style={{overflow: 'hidden'}}
                className="account-card">
                <Typography variant="h3">О проекте</Typography>
                <Typography variant="body2" className="muted-text">
                    AI Generator Studio — это единая рабочая среда для локального обучения Transformer-моделей,
                    подключения API-моделей, управления корпусом данных и чатов.
                </Typography>
                <Box sx={{display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1.5, mb: 2}}>
                    <Chip label={'Статус: ' + resolvedStatusTheme} color={statusColor} variant="filled"/>
                    <Chip label={modelKind} color="default" variant="outlined"/>
                    <Chip label={modelName} color={statusColor} variant="outlined"/>
                </Box>
                <div className="account-summary-grid">
                    <div className="account-summary-item">
                        <strong>Обучение локальной модели</strong>
                        <span>
                            Полный цикл внутри студии: токенизация, подготовка tf.data пайплайна, запуск обучения,
                            контроль статусов, пауза, откат и сохранение артефактов.
                        </span>
                    </div>
                    <div className="account-summary-item">
                        <strong>Гибкий режим API</strong>
                        <span>
                            Можно подключить внешнюю LLM и использовать ее в чате без переобучения локальной модели.
                            Это удобно для тестов UX и быстрых проверок промптов.
                        </span>
                    </div>
                    <div className="account-summary-item">
                        <strong>Безопасный шаринг чатов</strong>
                        <span>
                            Чужие чаты открываются только по пригласительной ссылке и только в режиме чтения.
                            Владелец сохраняет полный контроль над своим рабочим пространством.
                        </span>
                    </div>
                    <div className="account-summary-item">
                        <strong>Наблюдаемость и отказоустойчивость</strong>
                        <span>
                            В приложении есть живые статусы, логи и уведомления процессов. Это помогает быстро
                            диагностировать перегрузку, ошибки обучения и проблемы интеграций.
                        </span>
                    </div>
                </div>
            </GlassPanel>
        </div>
    );
}

function replaceRoute(path) {
    window.history.replaceState({}, '', path);
}

function pushRoute(path) {
    window.history.pushState({}, '', path);
}

function App() {
    const [authLoading, setAuthLoading] = useState(true);
    const [authUser, setAuthUser] = useState(null);
    const [routeState, setRouteState] = useState(() => resolveAppRoute(window.location.pathname));
    const [tab, setTab] = useState(() => getTabIndexByKey(resolveAppRoute(window.location.pathname).tabKey));
    const [statusDialogOpen, setStatusDialogOpen] = useState(false);
    const [apiExportBlockedDialogOpen, setApiExportBlockedDialogOpen] = useState(false);
    const [baseStatusTheme, setBaseStatusTheme] = useState('not_created');
    const [overlayStatusTheme, setOverlayStatusTheme] = useState('not_created');
    const [overlayActive, setOverlayActive] = useState(false);
    const [trainingNotices, setTrainingNotices] = useState([]);
    const [chatNotices, setChatNotices] = useState([]);
    const [mobileHeroMenuOpen, setMobileHeroMenuOpen] = useState(false);
    const [mobileChatMenuOpen, setMobileChatMenuOpen] = useState(false);
    const importInputRef = useRef(null);

    const studio = useStudioApp({
        enabled: Boolean(authUser),
    });
    const selectedChatId = studio.selectedChatId;
    const setSelectedChatId = studio.setSelectedChatId;

    const applyRoute = useCallback((nextPath, options = {}) => {
        const resolved = resolveAppRoute(nextPath);
        if (options.replace) {
            replaceRoute(resolved.path);
        } else {
            pushRoute(resolved.path);
        }
        setRouteState(resolved);
        if (resolved.tabKey) {
            setTab(getTabIndexByKey(resolved.tabKey));
        }
    }, []);

    useEffect(() => {
        const handlePopState = () => {
            const nextRoute = resolveAppRoute(window.location.pathname);
            setRouteState(nextRoute);
            if (nextRoute.tabKey) {
                setTab(getTabIndexByKey(nextRoute.tabKey));
            }
        };

        window.addEventListener('popstate', handlePopState);
        return () => {
            window.removeEventListener('popstate', handlePopState);
        };
    }, []);

    useEffect(() => {
        let cancelled = false;

        fetchAuthSession()
            .then((session) => {
                if (cancelled) {
                    return;
                }
                if (session?.authenticated && session.user) {
                    setAuthUser(session.user);
                } else {
                    setAuthUser(null);
                }
            })
            .catch(() => {
                if (!cancelled) {
                    setAuthUser(null);
                }
            })
            .finally(() => {
                if (!cancelled) {
                    setAuthLoading(false);
                }
            });

        return () => {
            cancelled = true;
        };
    }, []);

    useEffect(() => {
        if (routeState.type === 'shared-chat') {
            return;
        }

        if (routeState.tabKey) {
            const nextTab = getTabIndexByKey(routeState.tabKey);
            if (nextTab !== tab) {
                setTab(nextTab);
            }
        }

        if (routeState.type === 'chat' && routeState.chatId && routeState.chatId !== selectedChatId) {
            setSelectedChatId(routeState.chatId);
        }
    }, [routeState, selectedChatId, setSelectedChatId, tab]);

    useEffect(() => {
        if (!authUser || routeState.type === 'shared-chat') {
            return;
        }

        const currentTabKey = getTabKeyByIndex(tab);
        if (currentTabKey === APP_TAB_KEYS.CHATS) {
            const routeChatId = routeState.type === 'chat' ? routeState.chatId : null;
            const routeChatExists = !routeChatId || (studio.snapshot?.chats || []).some((chat) => chat.id === routeChatId);
            if (routeChatId && routeChatExists) {
                return;
            }

            const targetPath = buildTabPath(APP_TAB_KEYS.CHATS, {
                chatId: selectedChatId || null,
            });
            if (targetPath !== routeState.path) {
                applyRoute(targetPath, {replace: true});
            }
            return;
        }

        const targetPath = buildTabPath(currentTabKey);
        if (targetPath !== routeState.path) {
            applyRoute(targetPath, {replace: true});
        }
    }, [applyRoute, authUser, routeState.chatId, routeState.path, routeState.type, selectedChatId, studio.snapshot?.chats, tab]);

    const resolvedStatusTheme = useMemo(
        () => resolveStatusTheme(studio.snapshot),
        [studio.snapshot]
    );

    useEffect(() => {
        if (resolvedStatusTheme === baseStatusTheme) {
            setOverlayStatusTheme(resolvedStatusTheme);
            setOverlayActive(false);
            return;
        }

        setOverlayStatusTheme(resolvedStatusTheme);
        setOverlayActive(false);

        const revealTimeoutId = setTimeout(() => {
            setOverlayActive(true);
        }, 24);
        const settleTimeoutId = setTimeout(() => {
            setBaseStatusTheme(resolvedStatusTheme);
            setOverlayStatusTheme(resolvedStatusTheme);
            setOverlayActive(false);
        }, 1200);

        return () => {
            clearTimeout(revealTimeoutId);
            clearTimeout(settleTimeoutId);
        };
    }, [baseStatusTheme, resolvedStatusTheme]);

    const summaryItems = useMemo(() => {
        if (!studio.snapshot) {
            return [];
        }

        return [
            `${studio.snapshot.sources.length} источников`,
            `${formatNumber(studio.snapshot.model.chunkCount)} фрагментов`,
            `${formatNumber(studio.snapshot.model.trainedEpochs)} эпох`,
            `${studio.snapshot.chats.length} чатов`,
        ];
    }, [studio.snapshot]);

    const processBanner = useMemo(
        () => buildProcessBanner(studio.snapshot),
        [studio.snapshot]
    );

    const toastNotices = useMemo(() => {
        const scopedNotices = tab === 0 ? trainingNotices : tab === 1 ? chatNotices : [];
        const notices = [];

        if (studio.error) {
            notices.push({
                id: 'global-error',
                severity: 'error',
                message: studio.error,
            });
        }

        scopedNotices.forEach((notice) => {
            if (!notice?.id || !notice?.message) {
                return;
            }
            notices.push(notice);
        });

        const deduped = [];
        const seen = new Set();
        notices.forEach((notice) => {
            const identityKey = [
                notice.id || '',
                notice.severity || '',
                notice.title || '',
                String(notice.message || '').trim(),
            ].join('::');
            const fallbackMessageKey = [
                notice.severity || '',
                notice.title || '',
                String(notice.message || '').trim(),
            ].join('::');

            if (seen.has(identityKey) || seen.has(fallbackMessageKey)) {
                return;
            }
            seen.add(identityKey);
            seen.add(fallbackMessageKey);
            deduped.push(notice);
        });

        const priority = {error: 0, warning: 1, success: 2, info: 3};
        return deduped
            .sort((left, right) => (priority[left.severity] ?? 10) - (priority[right.severity] ?? 10))
            .slice(0, 5);
    }, [chatNotices, studio.error, tab, trainingNotices]);

    useEffect(() => {
        setMobileHeroMenuOpen(false);
        setMobileChatMenuOpen(false);
    }, [tab]);

    useEffect(() => {
        if (!statusDialogOpen && !apiExportBlockedDialogOpen) {
            return undefined;
        }

        const handleEscape = (event) => {
            if (event.key !== 'Escape') {
                return;
            }
            setStatusDialogOpen(false);
            setApiExportBlockedDialogOpen(false);
        };

        window.addEventListener('keydown', handleEscape);
        return () => {
            window.removeEventListener('keydown', handleEscape);
        };
    }, [apiExportBlockedDialogOpen, statusDialogOpen]);

    const handleLogout = async () => {
        try {
            await logoutSession();
        } catch (_error) {
            // Ignore and still clear local session.
        } finally {
            setAuthUser(null);
            applyRoute(buildTabPath(APP_TAB_KEYS.MODELS), {replace: true});
        }
    };

    if (authLoading) {
        return (
            <Box className="app-shell app-shell--loading app-shell--status-not_created">
                <div
                    aria-hidden="true"
                    className="app-shell__status-layer app-shell__status-layer--current app-shell__status-layer--not_created"
                />
                <LoadingState/>
            </Box>
        );
    }

    if (routeState.type === 'shared-chat' && routeState.token) {
        return (
            <Box
                className="app-shell app-shell--status-trained"
                style={{'--app-bg-image': `url(${process.env.PUBLIC_URL}/${backgroundImg})`}}
            >
                <div
                    aria-hidden="true"
                    className="app-shell__status-layer app-shell__status-layer--current app-shell__status-layer--trained"
                />
                <SharedChatView
                    token={routeState.token}
                    onReturnHome={() => applyRoute(buildTabPath(APP_TAB_KEYS.MODELS), {replace: false})}
                />
            </Box>
        );
    }

    if (!authUser) {
        return (
            <AuthView
                onAuthenticated={(nextUser) => {
                    setAuthUser(nextUser || null);
                    applyRoute(buildTabPath(APP_TAB_KEYS.MODELS), {replace: true});
                }}
            />
        );
    }

    if (studio.loading || !studio.snapshot) {
        return (
            <Box
                className={`app-shell app-shell--loading app-shell--status-${resolvedStatusTheme}`}
                style={{'--app-bg-image': `url(${process.env.PUBLIC_URL}/${backgroundImg})`}}
            >
                <div
                    aria-hidden="true"
                    className={`app-shell__status-layer app-shell__status-layer--current app-shell__status-layer--${baseStatusTheme}`}
                />
                <div
                    aria-hidden="true"
                    className={`app-shell__status-layer app-shell__status-layer--overlay app-shell__status-layer--${overlayStatusTheme} ${overlayActive ? 'app-shell__status-layer--active' : ''}`.trim()}
                />
                <LoadingState/>
            </Box>
        );
    }

    const statusEntries = studio.snapshot.training?.recentStatuses || [];
    const statusCount = statusEntries.length;
    const handleOpenMobileHeroMenu = () => {
        setMobileChatMenuOpen(false);
        setMobileHeroMenuOpen(true);
    };

    const handleCloseMobileHeroMenu = () => {
        setMobileHeroMenuOpen(false);
    };

    const handleOpenMobileChatMenu = () => {
        setMobileHeroMenuOpen(false);
        setMobileChatMenuOpen(true);
    };

    const handleCloseMobileChatMenu = () => {
        setMobileChatMenuOpen(false);
    };

    const handleNavigateChat = (chatId) => {
        if (!chatId) {
            applyRoute(buildTabPath(APP_TAB_KEYS.CHATS), {replace: false});
            return;
        }
        applyRoute(buildTabPath(APP_TAB_KEYS.CHATS, {chatId}), {replace: false});
    };

    const isExportingModel = studio.pendingAction === 'exportModel';
    const isImportingModel = studio.pendingAction === 'importModel';
    const apiExportBlocked = String(studio.snapshot?.model?.kind || 'local').toLowerCase() === 'api';

    return (
        <Box
            className={`app-shell app-shell--status-${resolvedStatusTheme}`}
            style={{'--app-bg-image': `url(${process.env.PUBLIC_URL}/${backgroundImg})`}}
        >
            <div
                aria-hidden="true"
                className={`app-shell__status-layer app-shell__status-layer--current app-shell__status-layer--${baseStatusTheme}`}
            />
            <div
                aria-hidden="true"
                className={`app-shell__status-layer app-shell__status-layer--overlay app-shell__status-layer--${overlayStatusTheme} ${overlayActive ? 'app-shell__status-layer--active' : ''}`.trim()}
            />
            <Container maxWidth={false} className="app-shell__container">
                <GlassPanel
                    style={{overflow: 'visible', marginBottom: '10px', zIndex:'99'}}
                    filterStyle={{borderRadius: '24px'}}
                >
                    <div className="hero-panel__topline">
                        <div className="hero-panel__title-wrap">
                            <div className="hero-panel__badge">
                                <AutoAwesomeIcon fontSize="medium"/>
                                <span className="hero-panel__badge-text">
                  <strong className="hero-panel__brand">AI Generator</strong>
                  <strong className="hero-panel__brand-accent">Studio</strong>
                </span>
                            </div>
                        </div>

                        <div className="hero-panel__mobile-toggle">
                            <AppAlerts
                                processBanner={processBanner}
                                notices={toastNotices}
                                resolvedStatusTheme={resolvedStatusTheme}
                                toastLayerClassName="app-toast-layer--inline app-toast-layer--mobile"
                            />
                            <Button
                                variant="contained"
                                className="hero-model-button hero-mobile-menu-button"
                                startIcon={<MenuRoundedIcon/>}
                                onClick={handleOpenMobileHeroMenu}
                            >
                                Меню
                            </Button>
                        </div>

                        <div className="hero-panel__status-wrap">
                            <div className="hero-panel__actions-row">
                                <AppAlerts
                                    processBanner={processBanner}
                                    notices={toastNotices}
                                    resolvedStatusTheme={resolvedStatusTheme}
                                    toastLayerClassName="app-toast-layer--inline"
                                />
                                <input
                                    ref={importInputRef}
                                    type="file"
                                    accept=".json,.aistudio,.aistudio.json,application/json"
                                    style={{display: 'none'}}
                                    onChange={(event) => {
                                        const selectedFile = event.target.files?.[0] || null;
                                        if (selectedFile) {
                                            studio.actions.importModel(selectedFile);
                                        }
                                        event.target.value = '';
                                    }}
                                />
                                <Tooltip title={`Статусы обучения: ${statusCount}`}>
                                    <IconButton
                                        className="hero-status-button"
                                        onClick={() => setStatusDialogOpen(true)}
                                        size="small"
                                    >
                                        <Badge badgeContent={statusCount} color="default" max={999}>
                                            <HistoryRoundedIcon fontSize="small"/>
                                        </Badge>
                                    </IconButton>
                                </Tooltip>
                            </div>
                            <div className="hero-panel__meta-row">
                                {/*<StatusPill label={studio.realtimeConnected ? 'Realtime online' : 'Realtime reconnecting'} tone="neutral" />*/}
                                <StatusPill label={studio.snapshot.model.engine} active tone="accent"/>
                                <StatusPill label={`Статус: ${studio.snapshot.model.lifecycle}`} tone="neutral"/>
                                {/*{summaryItems.map((item) => (*/}
                                {/*  <StatusPill key={item} label={item} tone="neutral" />*/}
                                {/*))}*/}
                            </div>
                        </div>
                    </div>

                    <div className="hero-panel__tabs">
                        <Tabs
                            value={tab}
                            onChange={(_event, nextTab) => applyRoute(buildTabPath(getTabKeyByIndex(nextTab)))}
                            variant="scrollable"
                            allowScrollButtonsMobile
                            scrollButtons="auto"
                        >
                            <Tab icon={<MemoryIcon/>} iconPosition="start" label="Модели"/>
                            <Tab icon={<ForumIcon/>} iconPosition="start" label="Чаты"/>
                            <Tab icon={<HubRoundedIcon/>} iconPosition="start" label="О проекте"/>
                            <Tab icon={<AutoAwesomeIcon/>} iconPosition="start" label="Аккаунт"/>
                        </Tabs>
                    </div>
                </GlassPanel>

                <div className={`hero-mobile-panel ${mobileHeroMenuOpen ? 'hero-mobile-panel--open' : ''}`.trim()}>
                    <div className="hero-mobile-panel__inner">
                        <div className="hero-mobile-panel__section">
                            <Button
                                variant="contained"
                                className="hero-model-button"
                                startIcon={<HistoryRoundedIcon fontSize="small"/>}
                                onClick={() => {
                                    setStatusDialogOpen(true);
                                    handleCloseMobileHeroMenu();
                                }}
                                fullWidth
                            >
                                {`Статусы обучения (${statusCount})`}
                            </Button>
                        </div>
                        <div className="hero-mobile-panel__section hero-mobile-panel__section--meta">
                            <StatusPill label={studio.realtimeConnected ? 'Realtime online' : 'Realtime reconnecting'}
                                        tone="neutral"/>
                            <StatusPill label={studio.snapshot.model.engine} active tone="accent"/>
                            <StatusPill label={`Статус: ${studio.snapshot.model.lifecycle}`} tone="neutral"/>
                            {summaryItems.map((item) => (
                                <StatusPill key={`mobile-${item}`} label={item} tone="neutral"/>
                            ))}
                        </div>
                    </div>
                </div>
                <button
                    type="button"
                    className={`hero-mobile-backdrop ${mobileHeroMenuOpen ? 'hero-mobile-backdrop--visible' : ''}`.trim()}
                    aria-label="Закрыть меню шапки"
                    onClick={handleCloseMobileHeroMenu}
                />

                <div
                    className={`app-shell__content ${tab === 0 ? 'app-shell__content--training' : tab === 1 ? 'app-shell__content--chat' : 'app-shell__content--account'}`}>
                    {tab === 0 ? (
                        <TrainingTab
                            snapshot={studio.snapshot}
                            busy={studio.busy}
                            error={studio.error}
                            onNoticesChange={setTrainingNotices}
                            pendingAction={studio.pendingAction}
                            realtimeConnected={studio.realtimeConnected}
                            serverLogs={studio.serverLogs}
                            serverStatus={studio.serverStatus}
                            uploadProgress={studio.uploadProgress}
                            processingProgress={studio.processingProgress}
                            onSaveSettings={studio.actions.saveSettings}
                            onSaveRuntimeConfig={studio.actions.saveRuntimeConfig}
                            onUploadFiles={studio.actions.uploadFiles}
                            onCreateTrainingQueue={studio.actions.createTrainingQueue}
                            onUploadQueueFiles={studio.actions.uploadQueueFiles}
                            onRemoveTrainingQueueSource={studio.actions.removeTrainingQueueSource}
                            onDeleteTrainingQueue={studio.actions.deleteTrainingQueue}
                            onAddUrlSource={studio.actions.addUrlSource}
                            onClearSources={studio.actions.clearSources}
                            onRemoveSource={studio.actions.removeSource}
                            onCreateModel={studio.actions.createModel}
                            onCreateNamedModel={studio.actions.createNamedModel}
                            onCreateApiModel={studio.actions.createApiModel}
                            onUpdateApiModel={studio.actions.updateApiModel}
                            onSelectModel={studio.actions.selectModel}
                            onDeleteLibraryModel={studio.actions.deleteLibraryModel}
                            onTrainModel={studio.actions.trainModel}
                            onPauseModel={studio.actions.pauseModel}
                            onRollbackModel={studio.actions.rollbackTraining}
                            onResetModel={studio.actions.resetModel}
                            onExportModel={studio.actions.exportModel}
                            onRequestImportModel={() => importInputRef.current?.click()}
                            onApiExportBlocked={() => setApiExportBlockedDialogOpen(true)}
                            isExportingModel={isExportingModel}
                            isImportingModel={isImportingModel}
                            apiExportBlocked={apiExportBlocked}
                        />
                    ) : tab === 1 ? (
                        <ChatTab
                            snapshot={studio.snapshot}
                            selectedChatId={studio.selectedChatId}
                            setSelectedChatId={studio.setSelectedChatId}
                            busy={studio.busy}
                            pendingAction={studio.pendingAction}
                            pendingReply={studio.pendingReply}
                            pendingRatings={studio.pendingRatings}
                            error={studio.error}
                            onNoticesChange={setChatNotices}
                            mobileChatsOpen={mobileChatMenuOpen}
                            onOpenMobileChatsMenu={handleOpenMobileChatMenu}
                            onCloseMobileChatsMenu={handleCloseMobileChatMenu}
                            onCreateChat={studio.actions.createChat}
                            onCreateShareLink={studio.actions.createChatShareLink}
                            onNavigateChat={handleNavigateChat}
                            onDeleteChat={studio.actions.deleteChat}
                            onStopReply={studio.actions.stopChatReply}
                            onSendMessage={studio.actions.sendMessage}
                            onEditMessage={studio.actions.updateChatMessage}
                            onRateMessage={studio.actions.rateMessage}
                            routeChatMissing={
                                routeState.type === 'chat' &&
                                Boolean(routeState.chatId) &&
                                routeState.chatId !== studio.snapshot.activeChat?.id
                            }
                        />
                    ) : tab === 2 ? (
                        <ProjectTab snapshot={studio.snapshot} resolvedStatusTheme={resolvedStatusTheme}/>
                    ) : (
                        <AccountTab
                            authUser={authUser}
                            snapshot={studio.snapshot}
                            onAuthenticated={(nextUser) => setAuthUser(nextUser || authUser)}
                            onLogout={handleLogout}
                        />
                    )}
                </div>

                {statusDialogOpen ? (
                    <div
                        className="app-modal app-modal--status"
                        role="dialog"
                        aria-modal="true"
                        aria-label="Статусы обучения"
                        onMouseDown={(event) => {
                            if (event.target === event.currentTarget) {
                                setStatusDialogOpen(false);
                            }
                        }}
                    >
                        <div className="app-modal__backdrop" onMouseDown={() => setStatusDialogOpen(false)}/>
                        <div className="app-modal__panel status-dialog-paper status-dialog-paper--dense">
                            <div className="app-modal__head">
                                <div className="status-glass-head">
                                    <Typography variant="h3">Статусы обучения</Typography>
                                    <StatusPill label={`${statusCount}`} active tone="accent"/>
                                </div>
                            </div>
                            <div className="app-modal__body status-dialog-content status-dialog-content--plain">
                                <div className="status-glass-list">
                                    {statusEntries.length ? (
                                        statusEntries.map((entry) => (
                                            <div key={entry.id} className="status-glass-card">
                                                <div className="status-glass-card__head">
                                                    <StatusPill label={entry.status} active tone="accent"/>
                                                    <span
                                                        className="status-dialog-time">{new Date(entry.createdAt).toLocaleString()}</span>
                                                </div>
                                                <Typography variant="subtitle2">{entry.phase}</Typography>
                                                <Typography variant="body2" className="muted-text">
                                                    {entry.message}
                                                </Typography>
                                            </div>
                                        ))
                                    ) : (
                                        <Typography variant="body2" className="muted-text">
                                            Статусы пока не поступали.
                                        </Typography>
                                    )}
                                </div>
                            </div>
                            <div className="app-modal__actions">
                                <Button variant="contained" color='inherit' onClick={() => setStatusDialogOpen(false)}>
                                    Закрыть
                                </Button>
                            </div>
                        </div>
                    </div>
                ) : null}

                {apiExportBlockedDialogOpen ? (
                    <div
                        className="app-modal app-modal--status"
                        role="dialog"
                        aria-modal="true"
                        aria-label="Экспорт API-модели недоступен"
                        onMouseDown={(event) => {
                            if (event.target === event.currentTarget) {
                                setApiExportBlockedDialogOpen(false);
                            }
                        }}
                    >
                        <div className="app-modal__backdrop" onMouseDown={() => setApiExportBlockedDialogOpen(false)}/>
                        <div
                            className="app-modal__panel status-dialog-paper status-dialog-paper--dense app-modal__panel--compact">
                            <div className="app-modal__head">
                                <Typography variant="h3">Экспорт API-модели недоступен</Typography>
                            </div>
                            <div className="app-modal__body status-dialog-content status-dialog-content--plain">
                                <Typography variant="body2" className="muted-text">
                                    Сейчас активна внешняя API-модель. Выберите локальную модель в библиотеке и
                                    повторите
                                    экспорт.
                                </Typography>
                            </div>
                            <div className="app-modal__actions">
                                <Button variant="contained" onClick={() => setApiExportBlockedDialogOpen(false)}>
                                    Понял
                                </Button>
                            </div>
                        </div>
                    </div>
                ) : null}
            </Container>
        </Box>
    );
}

export default App;
