import os
import time
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from model import load_model_and_tokenizer, generate_answer
from system_checks import check_nvidia_smi, check_torch_cuda, check_bitsandbytes, summarize_checks_host, summarize_checks_docker

load_dotenv()

# -----------------------
# Logging: base level WARNING (per requirement)
# -----------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("uii-llm-api")

# -----------------------
# System prompt (as provided)
# -----------------------
SYSTEM_PROMPT = (
    "Вы профессиональный менеджер поддержки в чате компании (компания продает курсы по AI) "
    "Университет Искусственного интеллекта.\n"
    "Ответьте на вопрос так, чтобы человек захотел после ответа купить обучение. Отвечайте на русском языке!"
)

DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "180"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

# -----------------------
# FastAPI init
# -----------------------
app = FastAPI(
    title="Нейро‑сотрудник (LLaMA 3 8B + QLoRA adapters)",
    version="1.0.0",
)

# -----------------------
# Prometheus: custom metrics + auto metrics for FastAPI
# -----------------------
LLM_REQUESTS_TOTAL = Counter("llm_requests_total", "Количество запросов к /generate")
LLM_GENERATION_LATENCY = Histogram(
    "llm_generation_latency_seconds",
    "Время генерации ответа LLM",
    buckets=(0.2,0.5,1,2,3,5,8,13,21,34),
)

Instrumentator().instrument(app).expose(app, include_in_schema=False)

# -----------------------
# Load model once at startup
# -----------------------
logger.warning("STARTUP: проверки CUDA/NVIDIA окружения (nvidia-smi, torch.cuda, bitsandbytes)")
smi = check_nvidia_smi()
if smi.ok:
    logger.warning("CHECK OK: %s", smi.message)
else:
    logger.error("CHECK FAIL: %s | %s", smi.message, smi.details or "")

cuda = check_torch_cuda()
if cuda.ok:
    logger.warning("CHECK OK: %s", cuda.message)
else:
    logger.error("CHECK FAIL: %s | %s", cuda.message, cuda.details or "")

bnb = check_bitsandbytes()
if bnb.ok:
    logger.warning("CHECK OK: %s", bnb.message)
else:
    logger.error("CHECK FAIL: %s | %s", bnb.message, bnb.details or "")

logger.warning("STARTUP: загрузка базовой модели + предобученных QLoRA/LoRA адаптеров (один раз)")
try:
    MODEL, TOKENIZER = load_model_and_tokenizer(logger=logger)
    logger.warning("STARTUP: модель успешно загружена, сервис готов")
except Exception as e:
    logger.error("STARTUP ERROR: модель не загрузилась: %s", str(e))
    MODEL, TOKENIZER = None, None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE


class GenerateResponse(BaseModel):
    result: str


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model is not initialized. Check service logs.")

    LLM_REQUESTS_TOTAL.inc()
    t0 = time.time()

    logger.warning("POST /generate prompt_prefix=%r max_new_tokens=%s temperature=%s",
                   req.prompt[:80], req.max_new_tokens, req.temperature)

    try:
        with LLM_GENERATION_LATENCY.time():
            answer = generate_answer(
                question=req.prompt,
                model=MODEL,
                tokenizer=TOKENIZER,
                system_prompt=SYSTEM_PROMPT,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
            )
        logger.warning("OK /generate elapsed=%.3fs", time.time() - t0)
        return GenerateResponse(result=answer)
    except Exception as e:
        logger.error("ERROR /generate elapsed=%.3fs err=%s", time.time() - t0, str(e))
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/health")
def health():
    """
    Health endpoint for monitoring.
    Возвращает JSON с проверками окружения.
    По умолчанию отдаём docker-проверки, если внутри контейнера видим /app (косвенный признак),
    иначе — host-проверки.

    Это удобно в учебной практике: студент сразу видит, что именно не готово.
    """
    inside_docker = os.path.exists("/.dockerenv") or os.path.isdir("/app")
    return summarize_checks_docker() if inside_docker else summarize_checks_host()

# -----------------------
# Minimal HTML + JS UI
# -----------------------
HTML = """<!doctype html>
<html lang='ru'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Нейро‑сотрудник (UII)</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 24px; }
    textarea { width: 100%; height: 140px; }
    .row { display:flex; gap:12px; align-items:center; margin: 10px 0; flex-wrap: wrap; }
    input { width: 140px; padding: 6px; }
    button { padding: 10px 14px; cursor: pointer; }
    pre { white-space: pre-wrap; background: #f6f6f6; padding: 12px; border-radius: 8px; }
    small { color: #666; }
  </style>
</head>
<body>
  <h1>Нейро‑сотрудник (LLaMA 3 8B + QLoRA)</h1>
  <p><small>Демо‑страница: запрос уходит на <code>/generate</code>. Метрики: <code>/metrics</code>.</small></p>

  <label>Вопрос:</label>
  <textarea id='prompt'>Чем тариф "Базовый" отличается от "Основного" в Университете ИИ?</textarea>

  <div class='row'>
    <label>max_new_tokens:</label>
    <input id='max_new_tokens' type='number' value='180' min='1' max='1000'/>
    <label>temperature:</label>
    <input id='temperature' type='number' value='0.7' step='0.1' min='0' max='2'/>
    <button id='send'>Сгенерировать</button>
  </div>

  <h3>Ответ:</h3>
  <pre id='result'></pre>

<script>
document.getElementById('send').addEventListener('click', async () => {
  const prompt = document.getElementById('prompt').value;
  const max_new_tokens = parseInt(document.getElementById('max_new_tokens').value, 10);
  const temperature = parseFloat(document.getElementById('temperature').value);

  document.getElementById('result').textContent = 'Генерируем...';

  const r = await fetch('/generate', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({prompt, max_new_tokens, temperature})
  });

  const data = await r.json();
  document.getElementById('result').textContent = data.result || JSON.stringify(data, null, 2);
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
