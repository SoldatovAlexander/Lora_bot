#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Скрипт проверки CUDA/NVIDIA для запуска "как из GitHub" (venv).
#
# Этот сценарий предполагает, что вы запускаете Python на ХОСТЕ.
# Поэтому мы проверяем:
# - наличие и работоспособность nvidia-smi (драйвер NVIDIA на хосте)
# - наличие GPU в системе
# - рекомендованные утилиты
# ============================================================

log()  { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*"; }
err()  { echo "[ERROR] $*" >&2; }

# 1) nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi найден. Проверяем запуск..."
  if nvidia-smi >/dev/null 2>&1; then
    log "OK: nvidia-smi работает."
  else
    err "nvidia-smi есть, но не запускается. Проверьте драйвер NVIDIA и перезагрузку."
    exit 2
  fi
else
  err "nvidia-smi не найден. Драйвер NVIDIA не установлен."
  warn "Подсказка (Ubuntu 24.04): sudo apt-get install -y ubuntu-drivers-common && sudo ubuntu-drivers autoinstall && sudo reboot"
  exit 2
fi

# 2) Проверка, что GPU видна (краткий вывод)
log "GPU список:"
nvidia-smi -L || true

log "Готово. Следующий шаг: Python-проверка (torch + bitsandbytes):"
echo "  python scripts/check_env_host.py"
