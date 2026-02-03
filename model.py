import os
import logging
from typing import Optional
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "unsloth/llama-3-8b-Instruct-bnb-4bit")

logger = logging.getLogger("uii-llm-api")

def load_model_and_tokenizer(logger: Optional[logging.Logger] = None):
    """
    ИНФЕРЕНС-ЗАГРУЗКА базовой модели (без обучения и адаптеров):
    1) Загружаем токенайзер из базовой модели
    2) Загружаем LLaMA 3 8B в 4-bit режиме
    """
    if logger is None:
        logger = logging.getLogger("model")

    logger.warning("MODEL: BASE_MODEL_NAME=%s", BASE_MODEL_NAME)

    logger.warning("MODEL: loading tokenizer from base model")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    logger.warning("MODEL: loading base model in 4-bit (first run may download ~16GB)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
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
