import os
import logging
from typing import Optional, Tuple

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from unsloth import FastLanguageModel
load_dotenv()




# В Docker (и docker-compose) адаптеры монтируются сюда
ADAPTER_DIR_DOCKER = "/app/model/LoRA_outputs"


def resolve_adapter_dir(logger: logging.Logger) -> str:
    """
    Определяем, где лежат адаптеры:
    - в контейнере: /app/model/LoRA_outputs
    - локально: путь из ADAPTER_DIR (по умолчанию ./model/LoRA_outputs)
    """
    docker_cfg = os.path.join(ADAPTER_DIR_DOCKER, "adapter_config.json")
    local_cfg = os.path.join(ADAPTER_DIR_ENV, "adapter_config.json")

    if os.path.exists(docker_cfg):
        logger.warning("MODEL: adapter dir resolved to Docker path: %s", ADAPTER_DIR_DOCKER)
        return ADAPTER_DIR_DOCKER

    if os.path.exists(local_cfg):
        logger.warning("MODEL: adapter dir resolved to local path: %s", ADAPTER_DIR_ENV)
        return ADAPTER_DIR_ENV

    # Если адаптеров нет, это критично — сервис не должен молча продолжать.
    raise FileNotFoundError(
        "Не найден adapter_config.json. "
        "Положите предобученные адаптеры в ./model/adapters (локально) "
        "или смонтируйте их в /app/model/adapters (Docker)."
    )





def load_model_and_tokenizer(logger: Optional[logging.Logger] = None):
    """
    ИНФЕРЕНС-ЗАГРУЗКА (без обучения):
    1) Загружаем токенайзер (из папки адаптера, если он там есть)
    2) Загружаем базовую LLaMA 3 8B в 4-bit режиме
    3) Накладываем предобученные QLoRA/LoRA адаптеры (PEFT)
    """
    if logger is None:
        logger = logging.getLogger("model")

    adapter_dir = resolve_adapter_dir(logger)

    logger.warning("MODEL: BASE_MODEL_NAME=%s", BASE_MODEL_NAME)
    logger.warning("MODEL: ADAPTER_DIR=%s", adapter_dir)

    
    logger.warning("MODEL: loading base model in 4-bit (first run may download ~16GB)")
    # Шаг 1. Загружаем базовую 4-битную модель и токенизатор
    base_model, tokenizer_new = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length=MAX_SEQ_LENGTH,   # тот же MAX_SEQ_LENGTH, что и при обучении
        dtype=None,
        load_in_4bit=True,
    )
    logger.warning("MODEL: applying PEFT adapters")
    # Шаг 2. "Обернём" базовую модель в PEFT-модель,
# подгрузив обученные LoRA-адаптеры из директории LORA_OUTPUT_DIR.
# Конфиг адаптера (target_modules, r, и т.д.) возьмётся из сохранённых файлов.
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )


# Переводим model_new в режим инференса (ускоренный вывод, без градиентов)
    FastLanguageModel.for_inference(model)
    logger.warning("MODEL: ready (eval mode)")
    return model, tokenizer_new


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
