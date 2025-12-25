"""
HOST (GitHub/venv) проверка окружения.

Проверяем:
- nvidia-smi
- torch.cuda
- bitsandbytes

Запуск:
  python scripts/check_env_host.py
"""

import json
from system_checks import summarize_checks_host


def main() -> int:
    report = summarize_checks_host()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("all_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
