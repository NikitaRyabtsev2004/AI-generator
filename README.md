# AI Generator Studio

Локальная студия для обучения и тестирования LLM-подобной модели: клиент на React, backend на Node.js и вычислительный контур на Python (TensorFlow/Keras).

## Что умеет проект

- Создание и переключение локальных и API-моделей.
- Загрузка источников данных (`txt`, `csv`, `json`, `jsonl`, `parquet`, URL).
- Подготовка корпуса, запуск и контроль обучения.
- Чат с моделью, оценка ответов, редактирование сообщений, остановка генерации.
- Шаринг чатов по ссылке в read-only режиме.
- Регистрация/авторизация пользователей и изоляция рабочих данных.

## Структура репозитория

- `src` — клиентская часть (интерфейс студии).
- `server` — Node.js API, оркестрация и хранение состояния.
- `server/python` — Python backend для обучения/генерации.

Подробные README:

- [Клиент](/C:/Users/nekit/OneDrive/Desktop/W/че-то/GitHub/react.js/AI-generator/src/README.md)
- [Server](/C:/Users/nekit/OneDrive/Desktop/W/че-то/GitHub/react.js/AI-generator/server/README.md)
- [Python](/C:/Users/nekit/OneDrive/Desktop/W/че-то/GitHub/react.js/AI-generator/server/python/README.md)

## Быстрый старт

## Требования

- Node.js 18+ (рекомендуется LTS)
- npm 9+
- Python 3.10–3.11 для backend обучения

## Установка

```bash
npm install
```

Для Python-части (Windows):

```powershell
cd server/python
./setup-venv.ps1
```

## Запуск в dev-режиме

Сервер:

```bash
npm run server:dev
```

Клиент:

```bash
npm run client:start
```

По умолчанию:

- API: `http://localhost:4000`
- Клиент (CRA dev): `http://localhost:3000` (или следующий свободный порт)

## Сборка

```bash
npm run build
```

## Проверка качества

```bash
npm run lint
```

Если линтер не настроен в текущей ветке, используйте проверку сборки:

```bash
npm run build
```

## Важные заметки

- Не редактируйте вручную артефакты в `server/artifacts` и runtime-данные в `server/data`, если нет задачи на восстановление/миграцию.
- Для API-моделей экспорт локальной модели недоступен по логике приложения.
- Если shared-ссылка чата устарела или чат удален, откроется корректное read-only сообщение об ошибке.
