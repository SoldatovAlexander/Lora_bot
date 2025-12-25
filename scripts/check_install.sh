#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Проверка и (по возможности) установка окружения для проекта
# Ubuntu 24.04 LTS (Selectel) + NVIDIA Tesla T4 + Docker/Compose
#
# ВАЖНО:
# - установка NVIDIA драйверов и NVIDIA Container Toolkit может
#   отличаться у разных провайдеров.
# - на Selectel драйверы часто уже установлены или ставятся по
#   их инструкции. Этот скрипт даёт "best effort".
# ============================================================

log()  { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*"; }
err()  { echo "[ERROR] $*" >&2; }

need_sudo() {
  if [[ $EUID -ne 0 ]]; then
    echo "sudo"
  else
    echo ""
  fi
}

SUDO="$(need_sudo)"

# ---------- 1) OS check ----------
if ! grep -qi "ubuntu" /etc/os-release; then
  warn "Скрипт рассчитан на Ubuntu. Продолжаю, но шаги могут отличаться."
fi

# ---------- 2) NVIDIA driver / nvidia-smi ----------
if command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi найден. Проверяем запуск..."
  if nvidia-smi >/dev/null 2>&1; then
    log "OK: nvidia-smi работает."
  else
    err "nvidia-smi есть, но не запускается. Проверьте драйвер NVIDIA и перезагрузку."
  fi
else
  warn "nvidia-smi не найден. Драйвер NVIDIA, вероятно, не установлен."
  warn "Пробую установить драйвер автоматически через ubuntu-drivers (может потребоваться перезагрузка)."
  $SUDO apt-get update
  $SUDO apt-get install -y ubuntu-drivers-common
  $SUDO ubuntu-drivers autoinstall || true
  warn "Если драйвер был установлен: выполните sudo reboot, затем повторите скрипт."
fi

# ---------- 3) Docker ----------
if command -v docker >/dev/null 2>&1; then
  log "Docker уже установлен: $(docker --version)"
else
  log "Docker не найден. Устанавливаем docker.io + compose plugin (упрощённый вариант)."
  $SUDO apt-get update
  $SUDO apt-get install -y docker.io docker-compose-plugin
  $SUDO systemctl enable --now docker
  log "Docker установлен: $(docker --version)"
fi

# Add current user to docker group (optional)
if [[ $EUID -ne 0 ]]; then
  if groups | grep -q "\bdocker\b"; then
    log "Пользователь уже в группе docker."
  else
    warn "Добавляю пользователя в группу docker (выйдите/войдите в ssh, чтобы применилось)."
    $SUDO usermod -aG docker "$USER" || true
  fi
fi

# ---------- 4) NVIDIA Container Toolkit (для Docker GPU) ----------
# Проверка: сможет ли контейнер увидеть GPU?
if command -v nvidia-smi >/dev/null 2>&1; then
  log "Пробуем тестовый запуск GPU контейнера..."
  if docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    log "OK: Docker видит GPU (nvidia-container-toolkit настроен)."
  else
    warn "Docker пока НЕ видит GPU. Вероятно, не установлен/не настроен NVIDIA Container Toolkit."
    warn "Установка toolkit зависит от репозиториев NVIDIA. Рекомендуется следовать официальной инструкции NVIDIA."
    warn "После установки toolkit обычно нужно: sudo systemctl restart docker"
  fi
fi

# ---------- 5) Python deps ----------
if command -v python3 >/dev/null 2>&1; then
  log "Python найден: $(python3 --version)"
else
  err "python3 не найден. Установите Python3."
  exit 1
fi

# venv check
if [[ ! -d ".venv" ]]; then
  warn "Виртуальное окружение .venv не найдено. Можно создать:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi

log "Готово. Для детальной проверки Python-окружения запустите:"
echo "  python scripts/check_env.py"
