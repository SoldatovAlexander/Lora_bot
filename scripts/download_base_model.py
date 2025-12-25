"""
Скрипт для предварительного скачивания базовой модели в кэш HuggingFace.

Зачем это нужно на сервере?
- первый запуск сервиса скачивает ~16GB модели и может занять время;
- лучше скачать заранее во время подготовки.
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "unsloth/llama-3-8b-Instruct-bnb-4bit")


def main():
    print("Downloading tokenizer...")
    AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    print("Downloading base model (4-bit)...")
    AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
