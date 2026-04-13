# Клиент (`src`) — структура и назначение файлов

Это React-клиент студии обучения/чата. Ниже кратко по основным файлам и где что менять.

## Точка входа и каркас приложения

### `src/index.js`
- Старт React-приложения и монтирование в DOM.

### `src/App.js`
- Главная компоновка интерфейса.
- Связывает UI-вкладки с состоянием из `useStudioApp`.
- Прокидывает глобальные статусы, действия и события.

### `src/LiquidGlass.jsx`
- Визуальный контейнер под glass-оформление интерфейса.

### `src/theme.js`
- Глобальная тема MUI (цвета, типографика, базовые настройки компонентов).

## API-слой

### `src/api/studioApi.js`
- HTTP и SSE-клиент для backend API.
- Методы dashboard/settings/model/chat/source/training.
- Подписка на realtime-события (`snapshot`, `logs`, `status`).

## Оркестрация состояния

### `src/hooks/useStudioApp.js`
- Основной orchestration hook приложения.
- Хранит snapshot, loading/error-флаги, логи и server status.
- Содержит действия: train/pause/reset/rollback, create/select model, upload sources, chat actions.

## Экран обучения

### `src/components/training/TrainingTab.jsx`
- Управление моделью, настройками и источниками.
- Запуск/пауза/откат/сброс обучения.
- Отображение статусов сервера и обучения.

### `src/components/training/TrainingChart.jsx`
- График динамики обучения (loss/history).

## Экран чата

### `src/components/chat/ChatTab.jsx`
- UI чатов и сообщений.
- Отправка сообщений и отображение ответов модели.
- Оценка ответов (feedback).

## Общие компоненты

### `src/components/shared/GlassPanel.jsx`
- Базовая glass-панель для карточек/блоков.

### `src/components/shared/MetricCard.jsx`
- Карточка метрики (label/value/subtext).

### `src/components/shared/StatusPill.jsx`
- Компонент статусного бейджа.

## Константы и утилиты

### `src/constants/modelConfig.js`
- Описания полей и диапазонов для UI-настроек модели/обучения.

### `src/utils/text.js`
- Форматирование текста, дат, чисел и вспомогательные преобразования.

## Стили

### `src/styles/global.css`
- Базовые глобальные стили.

### `src/styles/app-shell.css`
- Стили каркаса приложения.

### `src/styles/training-tab.css`
- Стили вкладки обучения.

### `src/styles/chat-tab.css`
- Стили вкладки чата.

### `src/styles/glass-panel.css`
- Общие стили glass-компонентов.
