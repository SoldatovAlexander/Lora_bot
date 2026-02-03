# Simple Bot - LLM на своем сервере

Простой бот для запуска LLaMA 3 8B модели на своем сервере с FastAPI.

## Что это?

- REST API для генерации текста с использованием LLaMA 3 8B
- Работает на CPU (медленнее) или GPU (быстрее)
- HTML интерфейс и Prometheus метрики
- Production-ready через Docker

## Быстрый старт

### На локальной машине

```bash
# 1. Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Запустить бот
uvicorn app:app --host 0.0.0.0 --port 8000
```

Открыть: **http://localhost:8000**

### В Docker

```bash
docker compose up --build
```

Доступ:
- API: http://localhost:8000
- Документация: http://localhost:8000/docs
- Метрики: http://localhost:8000/metrics

## На сервере с GPU

```bash
# Отредактировать .env
cp .env.example .env

# Развернуть
docker compose up -d --build
```

## Структура проекта

- `app.py` — FastAPI приложение
- `model.py` — работа с LLaMA 3 моделью
- `system_checks.py` — проверка окружения (CUDA, драйверы, bitsandbytes)
- `docker-compose.yml` — конфигурация для запуска
- `requirements.txt` — зависимости

## Тестирование

```bash
# Проверить компоненты
python test_bot.py

# Проверить API
python test_api.py
```

Результат: ✅ Все компоненты работают

## Требования

- Python 3.8+
- GPU NVIDIA 8GB+ (опционально)
- CUDA 11.8+ (если есть GPU)

## Документация

- `FINAL_REPORT.md` — итоговый отчет тестирования
- `TEST_REPORT.md` — детальные результаты тестов
- `DOCUMENTATION_INDEX.md` — индекс всей документации
