#!/usr/bin/env python3
"""
Комплексный тест функциональности бота
"""
import sys
import os

print("=" * 60)
print("ТЕСТ 1: Проверка импортов всех зависимостей")
print("=" * 60)

modules = {
    'dotenv': 'python-dotenv',
    'fastapi': 'fastapi',
    'pydantic': 'pydantic',
    'prometheus_client': 'prometheus-client',
    'transformers': 'transformers',
    'torch': 'torch',
    'bitsandbytes': 'bitsandbytes',
}

all_ok = True
for module, name in modules.items():
    try:
        __import__(module)
        print(f"✓ {name:25} OK")
    except ImportError as e:
        print(f"✗ {name:25} ОШИБКА: {e}")
        all_ok = False

print("\n" + "=" * 60)
print("ТЕСТ 2: Проверка CUDA/GPU")
print("=" * 60)

try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"Количество GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ CUDA не обнаружена (CPU режим)")
except Exception as e:
    print(f"✗ Ошибка при проверке CUDA: {e}")
    all_ok = False

print("\n" + "=" * 60)
print("ТЕСТ 3: Проверка функций model.py")
print("=" * 60)

try:
    from model import build_llama3_prompt, clean_llama3_output
    
    # Тест 3a: build_llama3_prompt
    system = "Ты помощник"
    user = "Привет"
    prompt = build_llama3_prompt(system, user)
    if "<|start_header_id|>system<|end_header_id|>" in prompt and \
       "<|start_header_id|>user<|end_header_id|>" in prompt and \
       "<|start_header_id|>assistant<|end_header_id|>" in prompt:
        print("✓ build_llama3_prompt(): корректно формирует промпт")
    else:
        print("✗ build_llama3_prompt(): не содержит требуемые теги")
        all_ok = False
    
    # Тест 3b: clean_llama3_output
    test_output = "<|start_header_id|>assistant<|end_header_id|>\nОтвет здесь<|eot_id|>"
    cleaned = clean_llama3_output(test_output)
    if cleaned == "Ответ здесь":
        print("✓ clean_llama3_output(): корректно очищает вывод")
    else:
        print(f"✗ clean_llama3_output(): ошибка очистки. Результат: '{cleaned}'")
        all_ok = False
        
except Exception as e:
    print(f"✗ Ошибка в model.py: {e}")
    import traceback
    traceback.print_exc()
    all_ok = False

print("\n" + "=" * 60)
print("ТЕСТ 4: Проверка system_checks.py")
print("=" * 60)

try:
    from system_checks import (
        check_nvidia_smi, 
        check_torch_cuda, 
        check_bitsandbytes,
        CheckResult
    )
    
    print("✓ Все функции system_checks импортируются")
    
    # Запуск проверок
    smi = check_nvidia_smi()
    print(f"  nvidia-smi: {'✓' if smi.ok else '⚠'} {smi.message[:50]}...")
    
    cuda = check_torch_cuda()
    print(f"  torch.cuda: {'✓' if cuda.ok else '✗'} {cuda.message[:50]}...")
    
    bnb = check_bitsandbytes()
    print(f"  bitsandbytes: {'✓' if bnb.ok else '✗'} {bnb.message[:50]}...")
    
except Exception as e:
    print(f"✗ Ошибка в system_checks.py: {e}")
    import traceback
    traceback.print_exc()
    all_ok = False

print("\n" + "=" * 60)
print("ТЕСТ 5: Проверка FastAPI приложения")
print("=" * 60)

try:
    # Отключаем загрузку модели для быстрой проверки
    os.environ['BASE_MODEL_NAME'] = 'dummy'
    
    from app import app, GenerateRequest, GenerateResponse, SYSTEM_PROMPT
    
    print("✓ FastAPI приложение загружено")
    print(f"  Название: {app.title}")
    print(f"  Версия: {app.version}")
    
    # Проверяем эндпоинты
    routes = [route.path for route in app.routes]
    expected = ['/generate', '/health', '/', '/metrics']
    for route in expected:
        if route in routes:
            print(f"  ✓ Маршрут {route}: существует")
        else:
            print(f"  ⚠ Маршрут {route}: не найден")
    
    # Проверяем Pydantic модели
    req = GenerateRequest(prompt="Тест", max_new_tokens=100, temperature=0.7)
    print(f"  ✓ GenerateRequest: OK")
    
    resp = GenerateResponse(result="Результат")
    print(f"  ✓ GenerateResponse: OK")
    
    print(f"  ✓ SYSTEM_PROMPT определена: {len(SYSTEM_PROMPT)} символов")
    
except Exception as e:
    print(f"✗ Ошибка в app.py: {e}")
    import traceback
    traceback.print_exc()
    all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
else:
    print("⚠️  НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
print("=" * 60)
