"""
Проверка окружения для запуска нейро‑сотрудника (FastAPI + LLaMA 3 8B 4-bit Base Model).

Задача:
- проверить наличие `nvidia-smi` (драйвер NVIDIA установлен и работает)
- проверить, что PyTorch видит CUDA (torch.cuda.is_available())
- проверить, что bitsandbytes импортируется (нужно для 4-bit загрузки)
- вывести подсказки, что делать, если что-то не так

Запуск:
  python scripts/check_env.py
"""

import json
import sys

from system_checks import summarize_checks


def main() -> int:
    report = summarize_checks()
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not report.get("all_ok", False):
        print("\nОкружение НЕ готово. Исправьте проблемы и запустите проверку снова.")
        print("Подсказка: смотрите docs/deploy_selectel_ubuntu24.md и scripts/check_install.sh")
        return 2

    print("\nОкружение готово. Можно запускать FastAPI сервис.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
