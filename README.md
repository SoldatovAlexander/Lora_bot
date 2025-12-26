# Нейро‑сотрудник (FastAPI) + LLaMA 3 8B (4-bit) + предобученные QLoRA адаптеры

Этот проект создан по коду из лекции и адаптирован под реальный деплой на сервер с GPU (Selectel, Tesla T4 16GB).

Важно:
- **обучения нет**
- мы берём базовую модель `unsloth/llama-3-8b-Instruct-bnb-4bit`
- накладываем **предобученные** QLoRA/LoRA адаптеры из `./model/LoRA_outputs`
- запускаем API + мониторинг (Prometheus/Grafana) через docker-compose

## Структура
- `app.py` — FastAPI API + HTML демо + /metrics
- `model.py` — загрузка модели (4-bit) и адаптеров (PEFT), генерация
- `Dockerfile` — GPU образ на Ubuntu 24.04 + CUDA runtime
- `docker-compose.yml` — app + prometheus + grafana + dcgm-exporter
- `prometheus/prometheus.yml` — сбор метрик с app и GPU
- `grafana/dashboard_llm_fastapi_gpu.json` — пример дашборда
- `docs/deploy_selectel_ubuntu24.md` — пошаговый деплой на сервер
- `model/adapters/` — сюда кладём адаптеры (и токенайзер, если сохранён рядом)

## Быстрый старт (локально)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

cp .env.example .env
uvicorn app:app --host 0.0.0.0 --port 8000
```

Открыть:
- http://localhost:8000/ (HTML)
- http://localhost:8000/docs (Swagger)
- http://localhost:8000/metrics (Prometheus)

## Docker + GPU
```bash
cp .env.example .env
docker build -t uii-llm-api:latest .
docker run --rm --gpus all -p 8000:8000   -v $(pwd)/model/adapters:/app/model/adapters:ro   --env-file .env   uii-llm-api:latest
```

## Compose (API + Prometheus + Grafana + GPU metrics)
```bash
cp .env.example .env
docker compose up -d --build
```

Доступ:
- API: http://SERVER_IP:8000/docs
- Prometheus: http://SERVER_IP:9090
- Grafana: http://SERVER_IP:3000 (admin/admin)


## Проверка окружения

Перед запуском на сервере полезно проверить CUDA/драйверы:

```bash
python scripts/check_env.py
```

Также есть bash-скрипт проверки/установки (best effort):

```bash
bash scripts/check_install.sh
```


## Проверка окружения (разделено по сценариям)

### Сценарий A: запуск на хосте (как из GitHub / venv)
1) Проверка драйверов и nvidia-smi:
```bash
bash scripts/check_host_cuda.sh
```
2) Проверка Python (torch.cuda + bitsandbytes):
```bash
python scripts/check_env_host.py
```

### Сценарий B: запуск в Docker/Compose
1) Проверка, что Docker видит GPU:
```bash
bash scripts/check_docker_runtime.sh
```
2) Проверка Python внутри контейнера:
```bash
docker compose run --rm app python scripts/check_env_docker.py
```

### Health endpoint
Сервис отдаёт `/health` (JSON), который показывает готовность окружения.
Можно открыть в браузере или подключить к внешнему мониторингу.
