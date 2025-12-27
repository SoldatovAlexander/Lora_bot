import os
import logging
from typing import Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

load_dotenv()

BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "unsloth/llama-3-8b-Instruct-bnb-4bit")

logger = logging.getLogger("uii-llm-api")

def resolve_adapter_dir() -> str:
    """
    Ищет папку с adapter_config.json и возвращает путь к ней.

    Поддерживает 2 режима:
    1) Локально (venv): обычно ./model/LoRA_outputs или /home/.../model/LoRA_outputs
    2) Docker: /app/model/LoRA_outputs (volume mount)
    """

    # 1) Берём из env (если задано)
    raw = os.getenv("ADAPTER_DIR", "").strip()

    # 2) Если env не задан — выбираем дефолт по "контейнерности"
    # Простая эвристика: в Docker обычно существует /.dockerenv
    in_docker = Path("/.dockerenv").exists()

    if not raw:
        raw = "/app/model/LoRA_outputs" if in_docker else "./model/LoRA_outputs"

    # 3) Нормализуем относительные пути для локального режима
    # Если запуск из корня проекта, ./model/LoRA_outputs корректен.
    # Если запуск из другой папки — лучше поставить абсолютный путь через ADAPTER_DIR.
    base = Path(raw).expanduser().resolve() if not raw.startswith("/app/") else Path(raw)

    # 4) Быстрая проверка: файл в корне
    if (base / "adapter_config.json").is_file():
        logger.warning("MODEL: adapter_config found in %s", base)
        return str(base)

    # 5) Ищем в подпапках (на 3 уровня)
    if base.exists():
        candidates = list(base.rglob("adapter_config.json"))
        candidates = [p for p in candidates if p.is_file()]
        if candidates:
            adapter_dir = candidates[0].parent
            logger.warning("MODEL: adapter_config found in subdir %s", adapter_dir)
            return str(adapter_dir)

    # 6) Диагностика: что реально есть
    logger.error("MODEL: ADAPTER_DIR raw='%s', resolved='%s', exists=%s", raw, base, base.exists())

    raise FileNotFoundError(
        f"Не найден adapter_config.json в {base} и подпапках. "
        f"Проверьте ADAPTER_DIR и то, откуда вы запускаете проект."
    )


def load_tokenizer(adapter_dir: str, logger: logging.Logger):
    """
    В ваших артефактах токенайзер часто сохранён рядом с адаптером.
    Поэтому пытаемся загрузить токенайзер из adapter_dir.
    Если файлов токенайзера нет, откатываемся на BASE_MODEL_NAME.
    """
    tokenizer_hint_files = [
        os.path.join(adapter_dir, "tokenizer.json"),
        os.path.join(adapter_dir, "tokenizer_config.json"),
        os.path.join(adapter_dir, "special_tokens_map.json"),
    ]
    has_any = any(os.path.exists(p) for p in tokenizer_hint_files)

    if has_any:
        logger.warning("MODEL: loading tokenizer from adapter_dir=%s", adapter_dir)
        return AutoTokenizer.from_pretrained(adapter_dir)

    logger.warning("MODEL: tokenizer files not found in adapter_dir; fallback to base model tokenizer: %s", BASE_MODEL_NAME)
    return AutoTokenizer.from_pretrained(BASE_MODEL_NAME)


def load_model_and_tokenizer(logger: Optional[logging.Logger] = None):
    """
    ИНФЕРЕНС-ЗАГРУЗКА (без обучения):
    1) Загружаем токенайзер (из папки адаптера, если он там есть)
    2) Загружаем базовую LLaMA 3 8B в 4-bit режиме
    3) Накладываем предобученные QLoRA/LoRA адаптеры (PEFT)
    """
    if logger is None:
        logger = logging.getLogger("model")

    adapter_dir = resolve_adapter_dir()

    logger.warning("MODEL: BASE_MODEL_NAME=%s", BASE_MODEL_NAME)
    logger.warning("MODEL: ADAPTER_DIR=%s", adapter_dir)

    tokenizer = load_tokenizer(adapter_dir, logger)

    logger.warning("MODEL: loading base model in 4-bit (first run may download ~16GB)")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
    )

    logger.warning("MODEL: applying PEFT adapters")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        device_map="auto",
    )

    model.eval()
    logger.warning("MODEL: ready (eval mode)")
    return model, tokenizer


def build_llama3_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Формат промпта для LLaMA 3 Instruct.
    Важно: если нарушить формат, качество ответов часто падает.
    """
    return (
        f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def clean_llama3_output(decoded: str) -> str:
    """
    Чистим ответ: убираем служебные токены и оставляем только текст ассистента.
    """
    assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>\n"
    if assistant_prefix in decoded:
        decoded = decoded.split(assistant_prefix)[-1]
    return decoded.replace("<|eot_id|>", "").strip()


def generate_answer(
    question: str,
    model,
    tokenizer,
    system_prompt: str,
    max_new_tokens: int = 180,
    temperature: float = 0.7,
) -> str:
    prompt = build_llama3_prompt(system_prompt, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return clean_llama3_output(decoded)
