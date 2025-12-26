# CUDA runtime (Ubuntu 24.04)
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ВАЖНО:
# В CUDA-образах часто прописан NVIDIA/CUDA apt-репозиторий.
# Он нам не нужен (CUDA runtime уже внутри образа),
# а apt-get update по нему вызывает warning про legacy trusted.gpg.
# Поэтому просто удаляем .list файлы репо перед apt-get update.
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true

# Ставим системные пакеты (минимально)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    ca-certificates \
    git \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Создаем venv (PEP 668 не мешает)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Обновляем pip и базовые инструменты уже внутри venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Ставим Python зависимости
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем код
COPY . /app

EXPOSE 8000

# Старт сервиса
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
