# Сервер (`server`) — структура и назначение файлов

Этот backend разделен на уровни: точка входа, инфраструктура API, доменная логика, движок обучения и слой хранения.

## Точка входа

### `server/index.js`
- Инициализирует логгер, runtime-конфиг и хранилище.
- Поднимает Express API и SSE-канал realtime-событий.
- Подключает маршруты (`routes`) и защиту от перегруза (`core/overloadGuard`).
- Отдает клиентский `build`, если он существует.

## Инфраструктура API (`server/core`)

### `server/core/apiResponse.js`
- Единый формат ответов API: `sendSuccess`, `sendError`.
- Вспомогательные writers для NDJSON и SSE.

### `server/core/overloadGuard.js`
- Ограничение тяжелых запросов (active + queue).
- Отказ с `503`, если сервер перегружен.

### `server/core/serverStatus.js`
- Формирует payload для `/api/status`.
- Метрики Node/process/event-loop + состояние обучения + статус Python bridge.

## Маршруты API (`server/routes`)

### `server/routes/systemRoutes.js`
- `/api/health`, `/api/status`, `/api/logs/recent`, `/api/events`, `/api/dashboard`.

### `server/routes/sourcesRoutes.js`
- Источники знаний: загрузка файлов, URL-источники, удаление источника.

### `server/routes/trainingRoutes.js`
- Очереди обучения и команды управления: train/pause/reset/rollback.

### `server/routes/modelsRoutes.js`
- Создание/выбор/удаление модели, runtime-настройки, import/export пакета модели.

### `server/routes/chatsRoutes.js`
- Чаты, отправка сообщений, оценка ответов.

## Сервисы (`server/services`)

### `server/services/uploadSourceService.js`
- Multer middlewares для загрузки файлов.
- Парсинг TXT/CSV/JSON.
- NDJSON-прогресс подготовки источников.
- Ограничения и троттлинг для больших загрузок.

## Движок модели и обучения (`server/engine`)

### `server/engine/modelEngine.js`
- Главная доменная логика приложения.
- Жизненный цикл модели, сбор корпуса, запуск обучения, rollback.
- Генерация ответов, retrieval, обратная связь пользователя.
- Реестр моделей (создание/выбор/удаление/импорт/экспорт).

### `server/engine/neuralModel.js`
- Node-обертка для Python runtime (load/generate/tokenize).

### `server/engine/trainingWorker.js`
- Worker thread для долгих задач обучения.
- Общается с Python backend и стримит статусы в engine.

### `server/engine/pythonBridge.js`
- Поиск Python runtime.
- Автоподготовка `server/python/.venv` (если нужно).
- Запуск Python-команд и телеметрия состояния bridge.

## Библиотеки/утилиты (`server/lib`)

### `server/lib/config.js`
- Фабрики default-state и базовые конфиги обучения/генерации.

### `server/lib/logger.js`
- Структурный логгер в память + файл (`server/data/server.log`).

### `server/lib/text.js`
- Нормализация текста, токенизация, preview/title, статистика.

### `server/lib/modelSettings.js`
- Валидация и нормализация training-настроек из UI.

### `server/lib/content.js`
- Извлечение и очистка текста веб-страниц по URL.

### `server/lib/webSearch.js`
- Поиск (DuckDuckGo HTML) и загрузка релевантных страниц для web-режима.

## Хранилище и артефакты (`server/storage`)

### `server/storage/store.js`
- SQLite-хранилище состояния студии и атомарные обновления `updateState`.
- Миграции/нормализация и восстановление состояния.

### `server/storage/runtimeConfig.js`
- Чтение/запись runtime-конфига (`server/data/runtime-config.json`).

### `server/storage/modelLibraryStorage.js`
- Хранение пакетов моделей (`server/data/model-library`).

### `server/storage/trainingQueueStorage.js`
- Дисковое хранилище источников очередей обучения и job payload-файлов.

## Python backend (`server/python`)

Содержит TensorFlow/Keras-часть (архитектура модели, токенизатор, tf.data pipeline, train loop, setup-venv).
Подробности смотри в [server/python/README.md](./python/README.md).

## Генерируемые каталоги

- `server/data/*` — БД, логи, runtime-конфиг, библиотеки моделей.
- `server/artifacts/*` — веса, токенизатор, манифесты, индексы.

Не редактируй эти каталоги вручную без необходимости восстановления.
