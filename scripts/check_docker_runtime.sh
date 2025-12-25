#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Скрипт проверки для сценария Docker/Compose.
#
# В Docker-образе уже есть CUDA runtime, НО драйверы NVIDIA
# всегда находятся на ХОСТЕ. Поэтому:
# - мы НЕ устанавливаем драйверы
# - мы проверяем, что docker запущен и умеет работать с GPU
# ============================================================

log()  { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*"; }
err()  { echo "[ERROR] $*" >&2; }

if ! command -v docker >/dev/null 2>&1; then
  err "Docker не найден. Установите docker + compose plugin."
  exit 2
fi

log "Docker: $(docker --version)"
if docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
  log "OK: Docker видит GPU (nvidia-container-toolkit настроен)."
else
  err "Docker НЕ видит GPU."
  warn "Причины обычно две: (1) нет драйвера NVIDIA на хосте, (2) не установлен/не настроен nvidia-container-toolkit."
  warn "После настройки toolkit обычно нужно: sudo systemctl restart docker"
  exit 2
fi

log "Готово. Для проверки Python внутри контейнера используйте:"
echo "  docker compose run --rm app python scripts/check_env_docker.py"
