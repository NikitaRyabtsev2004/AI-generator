# Клиентская часть (`src`)

React-интерфейс студии, где пользователь настраивает модель, обучает ее, работает с источниками и ведет чаты.

## Главные файлы

## Точка входа

- `src/index.js` — монтирование приложения в DOM.
- `src/App.js` — корневой контейнер, вкладки, маршрутизация, модальные окна, связка с `useStudioApp`.

## API и состояние

- `src/api/studioApi.js` — HTTP/SSE слой для общения с backend.
- `src/hooks/useStudioApp.js` — центральный orchestration-hook: snapshot, события, экшены, busy/error-состояния.

## Вкладки интерфейса

- `src/components/training/TrainingTab.jsx` — обучение, модели, источники, импорт/экспорт.
- `src/components/chat/ChatTab.jsx` — список чатов, сообщения, шаринг, копирование, редактирование, стоп генерации.
- `src/components/chat/SharedChatView.jsx` — публичный read-only просмотр чата по ссылке.

## Общие компоненты

- `src/components/shared/GlassPanel.jsx` — базовая liquid-glass панель.
- `src/components/shared/StatusPill.jsx` — компактный статусный бейдж.
- `src/components/shared/AppAlerts.jsx` — центр уведомлений (включая автораскрытие при новых уведомлениях).

## Стили

- `src/styles/app-shell.css` — каркас страницы, шапка, фоновые слои, модальные окна, уведомления.
- `src/styles/training-tab.css` — стили вкладки обучения.
- `src/styles/chat-tab.css` — стили чатов и сообщений.
- `src/styles/global.css` — базовые глобальные правила.

## Утилиты

- `src/utils/appRoutes.js` — роутинг вкладок и shared-чата.
- `src/utils/text.js` — форматирование дат/чисел/превью.
- `src/utils/chatContent.js` — разбор и рендер контента ответов (код-блоки, текст, ссылки).

## Как запускать клиент

```bash
npm run client:start
```

Сборка production:

```bash
npm run build
```

## Что править в первую очередь при изменениях UI

1. `src/App.js` — если меняется общий layout, вкладки, шапка, модалки.
2. `src/components/training/TrainingTab.jsx` — если меняется управление обучением и моделью.
3. `src/components/chat/ChatTab.jsx` — если меняется поведение чата.
4. `src/styles/*.css` — визуальные правки и адаптив.
