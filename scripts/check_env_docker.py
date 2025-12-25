"""
DOCKER (контейнер) проверка окружения.

В контейнере мы НЕ проверяем установку драйверов и nvidia-smi на хосте.
Проверяем то, что важно для нашего Python процесса:
- torch.cuda
- bitsandbytes

Запуск (пример):
  docker compose run --rm app python scripts/check_env_docker.py
"""

import json
from system_checks import summarize_checks_docker


def main() -> int:
    report = summarize_checks_docker()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("all_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
