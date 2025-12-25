# Деплой на Selectel (Санкт‑Петербург), Ubuntu 24.04 LTS, Tesla T4 16GB



## 0) Вход на сервер
```bash
ssh root@SERVER_IP # заменить на свои реквизиты входа
```

## 1) Обновляем систему
```bash
sudo apt update && sudo apt -y upgrade 
sudo reboot
```

## 2) Проверяем, что GPU виден системой
```bash
nvidia-smi
```
Если команды нет — драйвер NVIDIA не установлен (обычно у провайдера он уже есть или ставится по инструкции Selectel).

## 3) Ставим Docker + Compose
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu   $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

docker --version
docker compose version
```

## 4) NVIDIA Container Toolkit (нужно для GPU в Docker)
Если у вас `docker run --gpus all ...` не видит GPU — ставим toolkit.

Проверка:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu24.04 nvidia-smi
```

Если GPU не виден — установите nvidia-container-toolkit по официальной инструкции NVIDIA.

## 5) Клонируем репозиторий из GitHub
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <REPO_FOLDER>
```

## 6) Кладём адаптеры в проект
Скопируйте папку с адаптерами в:
`./model/adapters/`

Проверьте наличие:
- adapter_config.json
- adapter_model.safetensors (или .bin)

## 7) Настраиваем .env
```bash
cp .env.example .env
nano .env
```
Проверьте, что:
`BASE_MODEL_NAME=unsloth/llama-3-8b-Instruct-bnb-4bit`

## 8) Запуск через docker-compose
```bash
docker compose up -d --build
```

## 9) Проверяем сервисы
- FastAPI docs: http://SERVER_IP:8000/docs
- Web UI: http://SERVER_IP:8000/
- Metrics: http://SERVER_IP:8000/metrics
- Prometheus: http://SERVER_IP:9090
- Grafana: http://SERVER_IP:3000 (admin/admin)

## 10) Если первый запуск долго “думает”
Это нормально: скачивается базовая модель (~16GB) в HF cache.

Можно скачать заранее:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_base_model.py
```


## 11) Проверка окружения перед запуском

Python-проверка:
```bash
python scripts/check_env.py
```

Bash-проверка/установка (best effort):
```bash
bash scripts/check_install.sh
```


## 12) Проверка окружения (по сценариям)

### A) Запуск на хосте (venv)
```bash
bash scripts/check_host_cuda.sh
python scripts/check_env_host.py
```

### B) Запуск в Docker/Compose
```bash
bash scripts/check_docker_runtime.sh
# Проверка Python внутри контейнера:
docker compose run --rm app python scripts/check_env_docker.py
```

Health endpoint:
- http://SERVER_IP:8000/health
