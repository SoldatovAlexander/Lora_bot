import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
except Exception:
    torch = None


@dataclass
class CheckResult:
    ok: bool
    message: str
    details: Optional[str] = None


def _run(cmd: list[str], timeout: int = 10) -> Tuple[int, str, str]:
    """Run a command and capture output."""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def check_nvidia_smi() -> CheckResult:
    """Checks that `nvidia-smi` exists and can run successfully."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return CheckResult(
            ok=False,
            message="nvidia-smi не найден в PATH. Вероятно, драйвер NVIDIA не установлен.",
            details="Установите драйвер NVIDIA (обычно через `sudo ubuntu-drivers autoinstall`) и перезагрузите сервер.",
        )
    try:
        code, out, err = _run([exe], timeout=10)
        if code != 0:
            return CheckResult(
                ok=False,
                message="nvidia-smi запускается с ошибкой (код != 0).",
                details=err or out or "Проверьте установку драйвера NVIDIA и перезагрузите сервер.",
            )
        return CheckResult(ok=True, message="nvidia-smi работает.", details=out[:1000])
    except Exception as e:
        return CheckResult(ok=False, message="nvidia-smi не удалось запустить (исключение).", details=str(e))


def check_torch_cuda() -> CheckResult:
    """Checks torch + CUDA availability from Python."""
    if torch is None:
        return CheckResult(
            ok=False,
            message="PyTorch не импортируется. Проверьте установку зависимостей (requirements.txt).",
            details="Попробуйте: pip install -r requirements.txt",
        )
    try:
        available = torch.cuda.is_available()
        if not available:
            return CheckResult(
                ok=False,
                message="torch.cuda.is_available() == False. Python не видит CUDA/GPU.",
                details="Проверьте драйвер NVIDIA, наличие CUDA runtime и корректный запуск (особенно в Docker).",
            )
        name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
        return CheckResult(ok=True, message="PyTorch видит CUDA.", details=f"GPU: {name}; count={torch.cuda.device_count()}")
    except Exception as e:
        return CheckResult(ok=False, message="Ошибка при проверке CUDA через PyTorch.", details=str(e))


def check_bitsandbytes() -> CheckResult:
    """Checks bitsandbytes import - required for 4-bit loading."""
    try:
        import bitsandbytes  # noqa: F401
        return CheckResult(ok=True, message="bitsandbytes импортируется (4-bit загрузка возможна).")
    except Exception as e:
        return CheckResult(
            ok=False,
            message="bitsandbytes не импортируется. 4-bit загрузка может не работать.",
            details=str(e),
        )


def summarize_checks() -> dict:
    """Return a machine-readable summary of checks."""
    results = {
        "nvidia_smi": check_nvidia_smi().__dict__,
        "torch_cuda": check_torch_cuda().__dict__,
        "bitsandbytes": check_bitsandbytes().__dict__,
    }
    results["all_ok"] = all(v["ok"] for v in results.values() if isinstance(v, dict) and "ok" in v)
    return results


def assert_runtime_ready() -> None:
    """Raises RuntimeError if critical checks fail."""
    smi = check_nvidia_smi()
    cuda = check_torch_cuda()
    bnb = check_bitsandbytes()

    # nvidia-smi and torch cuda are considered critical for this project
    errors = []
    if not smi.ok:
        errors.append(f"[nvidia-smi] {smi.message}")
    if not cuda.ok:
        errors.append(f"[torch.cuda] {cuda.message}")
    # bitsandbytes is critical for 4-bit loading; treat as critical too
    if not bnb.ok:
        errors.append(f"[bitsandbytes] {bnb.message}")

    if errors:
        details = "\n".join(errors)
        raise RuntimeError("Среда не готова для запуска LLM (CUDA/driver checks failed):\n" + details)

def check_docker_gpu_runtime() -> CheckResult:
    """
    Проверка для сценария Docker/Compose:
    - драйверы NVIDIA ДОЛЖНЫ быть на хосте
    - но внутри контейнера мы обычно проверяем, что GPU вообще доступна процессу
    Поэтому минимально проверяем torch.cuda + bitsandbytes.
    """
    cuda = check_torch_cuda()
    if not cuda.ok:
        return cuda
    bnb = check_bitsandbytes()
    if not bnb.ok:
        return bnb
    return CheckResult(ok=True, message="Docker runtime check: torch.cuda + bitsandbytes OK")


def summarize_checks_host() -> dict:
    """Хостовый запуск (GitHub/venv) - проверяем и nvidia-smi, и torch.cuda, и bitsandbytes."""
    results = {
        "nvidia_smi": check_nvidia_smi().__dict__,
        "torch_cuda": check_torch_cuda().__dict__,
        "bitsandbytes": check_bitsandbytes().__dict__,
    }
    results["all_ok"] = all(v["ok"] for v in results.values() if isinstance(v, dict) and "ok" in v)
    return results


def summarize_checks_docker() -> dict:
    """Docker/Compose - проверяем то, что видит Python процесс (torch.cuda + bitsandbytes)."""
    cuda = check_torch_cuda()
    bnb = check_bitsandbytes()
    results = {
        "torch_cuda": cuda.__dict__,
        "bitsandbytes": bnb.__dict__,
    }
    results["all_ok"] = cuda.ok and bnb.ok
    return results
