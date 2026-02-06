from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import os
import re
import time
import shutil
import uuid

from . import wb_api
from . import ai_responder
from .models import GenerateResponsePayload, Feedback, Question, ReplyPayload
from typing import List, Dict, Optional

app = FastAPI()

# --- Cache Setup ---
# Используем директорию cache для кэша, чтобы избежать проблем с volume mapping
CACHE_DIR = "/app/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "response_cache.json")
response_cache: Dict[str, str] = {}

# --- Job Results Storage ---
# Храним результаты фоновых задач для Алисы
job_results: Dict[str, Dict[str, any]] = {}  # job_id -> {status, result, timestamp}
job_counter: int = 0  # Счётчик для номеров задач (1, 2, 3...)

# --- Current Feedback List for Reading ---
# Храним текущий список отзывов для команды "прочитай отзывы"
current_feedbacks_list: List[Feedback] = []  # Список отзывов для чтения

def load_cache():
    # Убеждаемся, что директория для кэша существует
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Проверяем, существует ли файл кэша
    if os.path.exists(CACHE_FILE):
        # Проверяем, не является ли это директорией (на случай ошибки)
        if os.path.isdir(CACHE_FILE):
            # Если это директория, удаляем её и создаём файл
            print(f"Warning: {CACHE_FILE} is a directory, removing it and creating a file instead.")
            shutil.rmtree(CACHE_FILE)
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            print("Cache file created.")
            return
        
        # Если это файл, пытаемся его прочитать
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    response_cache.update(json.loads(content))
                    print(f"Loaded {len(response_cache)} items from cache.")
                else:
                    print("Cache file is empty, starting with an empty cache.")
        except json.JSONDecodeError:
            print("Cache file is corrupted, starting with an empty cache.")
        except Exception as e:
            print(f"Error reading cache file: {e}, starting with an empty cache.")
    else:
        # Create the file if it doesn't exist
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        print("Cache file created.")


def save_cache():
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(response_cache, f, ensure_ascii=False, indent=4)


def _alice_response(text: str, end_session: bool = False) -> Dict[str, object]:
    return {
        "response": {"text": text, "end_session": end_session},
        "version": "1.0",
    }


def _extract_alice_command_and_intents(body: Dict[str, object]) -> tuple[str, Dict[str, object], str]:
    """
    Возвращает: command_lower, intents_dict, user_id
    """
    try:
        if not isinstance(body, dict):
            print(f"WARNING: body is not dict in _extract_alice_command_and_intents: {type(body)}")
            return "", {}, ""
        
        req = body.get("request")
        if not isinstance(req, dict):
            req = {}
        
        sess = body.get("session")
        if not isinstance(sess, dict):
            sess = {}
        
        command = str(req.get("command", "")).strip().lower()
        
        nlu = req.get("nlu")
        if isinstance(nlu, dict):
            intents = nlu.get("intents") or {}
        else:
            intents = {}
        
        if not isinstance(intents, dict):
            intents = {}
        
        user_id = str(sess.get("user_id", ""))
        
        return command, intents, user_id
    except Exception as e:
        print(f"ERROR in _extract_alice_command_and_intents: {e}")
        return "", {}, ""


def _is_exit_command(command: str, intents: Dict[str, object]) -> bool:
    # Встроенные интенты Яндекса + простые слова-выходы
    if "YANDEX.REJECT" in intents:
        return True
    exit_words = ["хватит", "выйти", "выход", "стоп", "закрой", "закрыть", "отмена"]
    return any(w in command for w in exit_words)


def _help_text() -> str:
    return (
        "Привет! Я помогу тебе управлять отзывами на Wildberries.\n\n"
        "Вот что я умею:\n"
        "• Ответить на отзывы — обработаю все неотвеченные отзывы и отправлю ответы на пятизвёздочные\n"
        "• Статус задачи — скажи номер задачи, чтобы узнать результат\n"
        "• Сколько отзывов — покажу количество неотвеченных отзывов и плохих\n"
        "• Сколько вопросов — покажу количество неотвеченных вопросов\n"
        "• Сколько пять звёзд — покажу статистику по пятизвёздочным отзывам\n"
        "• Какие отзывы остались — покажу список оставшихся отзывов с рейтингами\n"
        "• Прочитай отзывы — прочитаю все неотвеченные отзывы с деталями\n"
        "• Ответь на отзыв [номер] — отвечу на конкретный отзыв по номеру\n"
        "• Ответь на все отзывы — отвечу на все оставшиеся отзывы\n\n"
        "Что хочешь сделать?"
    )


def _run_auto_reply_5_stars_feedbacks_sync() -> Dict[str, object]:
    """
    Синхронная (долго выполняющаяся) обработка:
    - генерирует ответы для всех неотвеченных отзывов (кэширует, если нет в кэше)
    - отправляет ответы только на 5 звезд с rate limit 3 rps
    """
    print("Starting auto-reply flow for unanswered feedbacks (sync job)...")

    feedbacks: List[Feedback] = wb_api.get_unanswered_feedbacks()
    if not feedbacks:
        return {
            "status": "ok",
            "message": "Нет неотвеченных отзывов.",
            "total_feedbacks": 0,
            "generated": 0,
            "cached": 0,
            "replied_5_stars": 0,
            "errors": {},
        }

    generated_count = 0
    cached_count = 0
    replied_count = 0
    errors: Dict[str, str] = {}

    # Сначала генерируем ответы для всех отзывов и кладём в кэш
    # Пропускаем те, для которых уже есть ответ в кэше
    for fb in feedbacks:
        item_id = fb.id

        if item_id in response_cache:
            cached_count += 1
            continue

        response_text = ai_responder.generate_ai_response(
            item_id=item_id,
            text=fb.text or "",
            custom_prompt=None,
            rating=fb.productValuation,
            product_name=fb.productDetails.productName,
            advantages=getattr(fb, "advantages", None),
            pluses=getattr(fb, "pluses", None),
            minuses=getattr(fb, "minuses", None),
        )

        if response_text and not response_text.startswith("Ошибка") and not response_text.startswith("API-ключ") and not response_text.startswith("Не удалось"):
            response_cache[item_id] = response_text
            generated_count += 1
        else:
            errors[item_id] = response_text or "Пустой ответ ИИ"

    save_cache()
    print(f"Generated {generated_count} new responses, {cached_count} already cached, out of {len(feedbacks)} total feedbacks")

    # Теперь отвечаем только на отзывы с оценкой ровно 5 звёзд
    sent_in_current_second = 0
    second_window_start = time.time()

    for fb in feedbacks:
        if fb.productValuation != 5:
            continue

        item_id = fb.id
        reply_text = response_cache.get(item_id)
        if not reply_text:
            continue

        now = time.time()
        if now - second_window_start >= 1:
            second_window_start = now
            sent_in_current_second = 0

        if sent_in_current_second >= 3:
            sleep_time = 1 - (now - second_window_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            second_window_start = time.time()
            sent_in_current_second = 0

        success = wb_api.reply_to_item(
            item_id=item_id,
            text=reply_text,
            item_type="feedbacks",
        )

        if success:
            replied_count += 1
            sent_in_current_second += 1
            if item_id in response_cache:
                del response_cache[item_id]
                save_cache()
        else:
            errors[item_id] = "Не удалось отправить ответ в WB"

    return {
        "status": "ok",
        "message": "Автообработка отзывов завершена.",
        "total_feedbacks": len(feedbacks),
        "generated": generated_count,
        "cached": cached_count,
        "replied_5_stars": replied_count,
        "errors": errors,
    }


def _run_auto_reply_job(job_id: str) -> None:
    """Фоновая задача для Алисы."""
    try:
        print(f"[job {job_id}] started")
        job_results[job_id] = {
            "status": "running",
            "result": None,
            "timestamp": time.time()
        }
        result = _run_auto_reply_5_stars_feedbacks_sync()
        print(f"[job {job_id}] finished: {result}")
        
        # Сохраняем результат
        job_results[job_id] = {
            "status": "completed",
            "result": result,
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"[job {job_id}] failed: {e}")
        job_results[job_id] = {
            "status": "failed",
            "result": {"error": str(e)},
            "timestamp": time.time()
        }
        job_results[job_id] = {
            "status": "failed",
            "result": {"error": str(e)},
            "timestamp": time.time()
        }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    load_cache()
    try:
        # Prune stale entries from cache
        print("Checking for stale cache entries...")
        active_feedbacks = wb_api.get_unanswered_feedbacks()
        active_questions = wb_api.get_unanswered_questions()
        
        active_ids = {fb.id for fb in active_feedbacks} | {q.id for q in active_questions}
        
        initial_cache_size = len(response_cache)
        
        # Create a new cache with only active IDs
        pruned_cache = {id: response for id, response in response_cache.items() if id in active_ids}
        
        if len(pruned_cache) < initial_cache_size:
            response_cache.clear()
            response_cache.update(pruned_cache)
            save_cache()
            removed_count = initial_cache_size - len(pruned_cache)
            print(f"Cache pruned. Removed {removed_count} stale entries.")
        else:
            print("No stale cache entries found. Cache is up to date.")
            
    except Exception as e:
        print(f"Could not prune cache during startup: {e}")
        print("This might happen if the WB API key is not yet available or invalid.")


# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/feedbacks", response_model=List[Feedback])
async def get_feedbacks():
    return wb_api.get_unanswered_feedbacks()

@app.get("/api/questions", response_model=List[Question])
async def get_questions():
    return wb_api.get_unanswered_questions()

@app.post("/api/alice-webhook")
async def alice_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Единый webhook для навыка Алисы.

    Поведение:
    - Launch: приветствие + список команд
    - Help: список команд
    - Команды:
        - "ответь на отзывы" (запускает фоновой job, отвечает сразу)
        - "сколько отзывов"
        - "сколько вопросов"
        - "процент 5 звезд" / "сколько 5 звезд"
    - Неизвестная команда: подсказка
    - Выход: "хватит/стоп/выйти" или интент YANDEX.REJECT
    """
    # Логируем все входящие запросы для отладки
    print(f"=== ALICE WEBHOOK REQUEST ===")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    
    # Проверяем content-type
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        print(f"WARNING: Invalid content-type: {content_type}")
        # Если это не JSON, но запрос от Алисы - всё равно попробуем обработать
        if not content_type:
            print("No content-type, trying to parse as JSON anyway")
    
    # Пытаемся получить тело запроса
    body = None
    try:
        # Проверяем, есть ли тело запроса
        body_bytes = await request.body()
        if not body_bytes:
            print("WARNING: Empty request body")
            return _alice_response("Привет! " + _help_text(), end_session=False)
        
        body = json.loads(body_bytes.decode('utf-8'))
        print(f"Parsed body: {json.dumps(body, ensure_ascii=False, indent=2)}")
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        print(f"Raw body: {body_bytes.decode('utf-8', errors='ignore')[:500]}")
        return _alice_response("Ошибка обработки запроса. Попробуйте ещё раз.", end_session=False)
    except Exception as e:
        print(f"ERROR: Unexpected error parsing request: {e}")
        return _alice_response("Ошибка обработки запроса. Попробуйте ещё раз.", end_session=False)
    
    if not body:
        print("WARNING: Body is None or empty")
        return _alice_response("Привет! " + _help_text(), end_session=False)

    # Проверяем структуру запроса
    if not isinstance(body, dict):
        print(f"ERROR: Body is not a dict: {type(body)}")
        return _alice_response("Ошибка формата запроса. Попробуйте ещё раз.", end_session=False)
    
    # Launch event
    req = body.get("request") or {}
    if not isinstance(req, dict):
        req = {}
    
    if req.get("type") == "Launch":
        print("Launch event detected")
        return _alice_response(_help_text(), end_session=False)

    try:
        command, intents, user_id = _extract_alice_command_and_intents(body)
        print(f"Extracted - command: '{command}', intents: {intents}, user_id: {user_id}")
    except Exception as e:
        print(f"ERROR extracting command/intents: {e}")
        # Если не удалось извлечь - пробуем обработать как есть
        command = req.get("command", "").lower() if isinstance(req, dict) else ""
        intents = {}
        user_id = ""

    # Встроенная помощь
    if "YANDEX.HELP" in intents:
        return _alice_response(_help_text(), end_session=False)

    # Выход
    if _is_exit_command(command, intents):
        return _alice_response("Хорошо, выхожу.", end_session=True)

    # Команда "ответь на отзыв [номер]" или "ответь на отзывы [номера]" - отвечает на конкретные отзывы
    # Проверяем ПЕРЕД автоответом, чтобы не было конфликта
    if "ответь" in command and "отзыв" in command and ("все" in command or any(char.isdigit() for char in command)):
        # Если список отзывов пуст, загружаем заново
        if not current_feedbacks_list:
            current_feedbacks_list.extend(wb_api.get_unanswered_feedbacks())
        
        if not current_feedbacks_list:
            return _alice_response("Нет неотвеченных отзывов для ответа.", end_session=False)
        
        # Извлекаем номера отзывов из команды
        numbers = [int(m) for m in re.findall(r'\b(\d+)\b', command)]
        
        # Если "ответь на все отзывы" или "ответь на все"
        if "все" in command:
            numbers = list(range(1, len(current_feedbacks_list) + 1))
        
        if not numbers:
            return _alice_response("Не указаны номера отзывов. Скажи, например: 'ответь на отзыв 1' или 'ответь на отзывы 1 и 2'.", end_session=False)
        
        # Фильтруем номера - только те, что в диапазоне
        valid_numbers = [n for n in numbers if 1 <= n <= len(current_feedbacks_list)]
        if not valid_numbers:
            return _alice_response(f"Номера должны быть от 1 до {len(current_feedbacks_list)}.", end_session=False)
        
        # Отвечаем на выбранные отзывы
        replied_count = 0
        errors = []
        
        for num in valid_numbers:
            fb = current_feedbacks_list[num - 1]  # Индексация с 0
            item_id = fb.id
            
            # Генерируем ответ, если его нет в кэше
            if item_id not in response_cache:
                response_text = ai_responder.generate_ai_response(
                    item_id=item_id,
                    text=fb.text or "",
                    custom_prompt=None,
                    rating=fb.productValuation,
                    product_name=fb.productDetails.productName,
                    advantages=getattr(fb, "advantages", None),
                    pluses=getattr(fb, "pluses", None),
                    minuses=getattr(fb, "minuses", None),
                )
                
                if response_text and not response_text.startswith("Ошибка") and not response_text.startswith("API-ключ") and not response_text.startswith("Не удалось"):
                    response_cache[item_id] = response_text
                    save_cache()
                else:
                    errors.append(f"отзыв {num}")
                    continue
            
            reply_text = response_cache.get(item_id)
            if not reply_text:
                errors.append(f"отзыв {num}")
                continue
            
            # Отправляем ответ с rate limiting
            time.sleep(0.34)  # ~3 запроса в секунду
            success = wb_api.reply_to_item(
                item_id=item_id,
                text=reply_text,
                item_type="feedbacks",
            )
            
            if success:
                replied_count += 1
                if item_id in response_cache:
                    del response_cache[item_id]
                    save_cache()
            else:
                errors.append(f"отзыв {num}")
        
        # Формируем ответ
        if replied_count == len(valid_numbers):
            if len(valid_numbers) == 1:
                return _alice_response(f"Ответ отправлен на отзыв {valid_numbers[0]}.", end_session=False)
            else:
                return _alice_response(f"Ответы отправлены на {replied_count} отзывов.", end_session=False)
        elif replied_count > 0:
            return _alice_response(f"Отправлено ответов: {replied_count}. Ошибки при отправке: {', '.join(errors)}.", end_session=False)
        else:
            return _alice_response(f"Не удалось отправить ответы. Ошибки: {', '.join(errors)}.", end_session=False)

    # Автоответ (только для этой команды используем фоновые задачи)
    # Проверяем ПОСЛЕ проверки на конкретные номера
    if "ответь на отзыв" in command or "ответь на отзывы" in command or "обработай отзывы" in command:
        global job_counter
        job_counter += 1
        job_id = str(job_counter)
        
        background_tasks.add_task(_run_auto_reply_job, job_id)
        return _alice_response(
            f"Хорошо, начинаю обработку отзывов. Это задача номер {job_id}. "
            f"Когда закончу, спроси меня: 'статус задачи {job_id}' — и я расскажу результат.",
            end_session=False,
        )
    
    # Проверка статуса задачи (только для задач обработки отзывов)
    if "статус задачи" in command or "результат задачи" in command:
        # Ищем номер задачи (просто цифра)
        job_id_match = re.search(r'\b(\d+)\b', command)
        if job_id_match:
            job_id = job_id_match.group(1)
            if job_id in job_results:
                job_data = job_results[job_id]
                status = job_data.get("status", "unknown")
                
                if status == "running":
                    return _alice_response(f"Задача {job_id} ещё выполняется. Подождите немного.", end_session=False)
                elif status == "completed":
                    result = job_data.get("result", {})
                    total = result.get("total_feedbacks", 0)
                    generated = result.get("generated", 0)
                    replied = result.get("replied_5_stars", 0)
                    
                    if total == 0:
                        text = f"Задача {job_id} завершена. Сейчас нет неотвеченных отзывов."
                    elif replied > 0:
                        text = (
                            f"Задача {job_id} завершена! "
                            f"Обработано отзывов: {total}. "
                            f"Отправлено ответов на пятизвёздочные отзывы: {replied}."
                        )
                    else:
                        text = (
                            f"Задача {job_id} завершена. "
                            f"Обработано отзывов: {total}, "
                            f"но среди них не было пятизвёздочных, поэтому ответы не отправлялись."
                        )
                    
                    return _alice_response(text, end_session=False)
                elif status == "failed":
                    return _alice_response(f"Задача {job_id} завершилась с ошибкой.", end_session=False)
            else:
                return _alice_response(f"Задача {job_id} не найдена.", end_session=False)
        else:
            # Если номер задачи не указан, показываем последнюю завершённую
            latest_job = None
            latest_number = 0
            for jid, job_data in job_results.items():
                if job_data.get("status") == "completed":
                    try:
                        jid_num = int(jid)
                        if jid_num > latest_number:
                            latest_job = (jid, job_data)
                            latest_number = jid_num
                    except ValueError:
                        continue
            
            if latest_job:
                jid, job_data = latest_job
                result = job_data.get("result", {})
                total = result.get("total_feedbacks", 0)
                replied = result.get("replied_5_stars", 0)
                
                if total == 0:
                    text = f"Последняя задача {jid} завершена. Неотвеченных отзывов не было."
                elif replied > 0:
                    text = (
                        f"Последняя задача {jid}: обработано {total} отзывов, "
                        f"отправлено ответов на пятизвёздочные: {replied}."
                    )
                else:
                    text = (
                        f"Последняя задача {jid}: обработано {total} отзывов, "
                        f"но среди них не было пятизвёздочных."
                    )
                return _alice_response(text, end_session=False)
            else:
                return _alice_response("Нет завершённых задач. Укажите номер задачи или запустите обработку.", end_session=False)

    # Статистика (выполняется сразу, без фоновых задач)
    if "сколько" in command and "отзыв" in command:
        feedbacks = wb_api.get_unanswered_feedbacks()
        count = len(feedbacks)
        if count == 0:
            return _alice_response("Сейчас нет неотвеченных отзывов. Всё чисто!", end_session=False)
        
        # Считаем плохие отзывы (ниже 5 звезд)
        bad_count = sum(1 for fb in feedbacks if fb.productValuation < 5)
        if bad_count == 0:
            return _alice_response(f"Всего отзывов: {count}, плохих отзывов нет.", end_session=False)
        else:
            return _alice_response(f"Всего отзывов: {count}, плохих отзывов: {bad_count}.", end_session=False)

    if "сколько" in command and "вопрос" in command:
        questions = wb_api.get_unanswered_questions()
        count = len(questions)
        if count == 0:
            return _alice_response("Нет неотвеченных вопросов. Всё отлично!", end_session=False)
        return _alice_response(f"У тебя {count} неотвеченных вопросов.", end_session=False)

    if ("процент" in command or "%" in command or "сколько" in command) and ("5" in command or "пять" in command or "пятизв" in command):
        feedbacks = wb_api.get_unanswered_feedbacks()
        total = len(feedbacks)
        if total == 0:
            return _alice_response("Сейчас нет неотвеченных отзывов, поэтому статистику посчитать нельзя.", end_session=False)
        five_star_count = sum(1 for fb in feedbacks if fb.productValuation == 5)
        return _alice_response(
            f"Среди неотвеченных отзывов: {five_star_count} пятизвёздочных из {total}.",
            end_session=False
        )

    # Команда "какие отзывы остались" - показывает список оставшихся отзывов с рейтингами
    if ("какие" in command or "каких" in command) and "отзыв" in command and ("остал" in command or "есть" in command):
        current_feedbacks_list.clear()
        current_feedbacks_list.extend(wb_api.get_unanswered_feedbacks())
        if not current_feedbacks_list:
            return _alice_response("Нет неотвеченных отзывов.", end_session=False)
        
        # Группируем по рейтингам
        rating_groups: Dict[int, int] = {}
        for fb in current_feedbacks_list:
            rating = fb.productValuation
            rating_groups[rating] = rating_groups.get(rating, 0) + 1
        
        parts = []
        for rating in sorted(rating_groups.keys(), reverse=True):
            count = rating_groups[rating]
            if count == 1:
                parts.append(f"1 отзыв — {rating} звезды" if rating < 5 else "1 отзыв — 5 звёзд")
            else:
                parts.append(f"{count} отзыва — {rating} звезды" if rating < 5 else f"{count} отзывов — 5 звёзд")
        
        text = f"Осталось {len(current_feedbacks_list)} отзывов: " + ", ".join(parts) + "."
        return _alice_response(text, end_session=False)

    # Команда "прочитай отзывы" - читает все отзывы по порядку
    if "прочитай" in command and "отзыв" in command:
        current_feedbacks_list.clear()
        current_feedbacks_list.extend(wb_api.get_unanswered_feedbacks())
        if not current_feedbacks_list:
            return _alice_response("Нет неотвеченных отзывов для чтения.", end_session=False)
        
        # Формируем текст для чтения всех отзывов
        texts = []
        for idx, fb in enumerate(current_feedbacks_list, 1):
            text_parts = [f"Отзыв {idx}."]
            text_parts.append(f"Товар: {fb.productDetails.productName}.")
            text_parts.append(f"Оценка: {fb.productValuation} {'звезды' if fb.productValuation < 5 else 'звёзд'}.")
            
            if fb.pluses:
                text_parts.append(f"Плюсы: {fb.pluses}.")
            if fb.minuses:
                text_parts.append(f"Минусы: {fb.minuses}.")
            if fb.text and fb.text.strip():
                text_parts.append(f"Комментарий: {fb.text}.")
            else:
                text_parts.append("Покупатель не оставил комментария.")
            
            texts.append(" ".join(text_parts))
        
        full_text = " ".join(texts)
        return _alice_response(full_text, end_session=False)

    # Фолбэк - только если команда не пустая (не Launch)
    if command:
        return _alice_response("Извини, не поняла. " + _help_text(), end_session=False)
    else:
        # Если команда пустая (например, пользователь просто запустил навык и ничего не сказал)
        return _alice_response(_help_text(), end_session=False)

@app.post("/api/generate-multiple-responses")
async def generate_multiple_responses(payload: GenerateResponsePayload):
    print(f"Generating multiple responses for {payload.id}")
    
    # Кнопка "Повторить" всегда генерирует новые варианты
    # Проверяем кэш только если это не принудительная регенерация И нет кастомного промпта
    if not payload.force and not payload.prompt and payload.id in response_cache:
        print(f"Returning cached response for {payload.id}")
        cached_response = response_cache[payload.id]
        return {
            "gpt": cached_response,
            "gpt_v2": cached_response,
            "gpt_v3": cached_response
        }
    
    print(f"Generating new responses for {payload.id} (force={payload.force}, prompt={payload.prompt})")
    responses = ai_responder.generate_multiple_ai_responses(
        item_id=payload.id,
        text=payload.text,
        custom_prompt=payload.prompt,
        rating=payload.rating,
        product_name=payload.productName,
        advantages=getattr(payload, 'advantages', None),
        pluses=getattr(payload, 'pluses', None),
        minuses=getattr(payload, 'minuses', None),
    )
    
    # НЕ сохраняем в кэш здесь - сохраним только когда пользователь выберет один из вариантов
    print(f"Generated {len(responses)} responses for {payload.id}, waiting for user selection")
    
    return responses

@app.post("/api/generate-response")
async def generate_response(payload: GenerateResponsePayload):
    # Use cache only if regeneration is not forced and the ID is in the cache.
    # A custom prompt implies regeneration, so we don't need to check for it here.
    if not payload.force and payload.id in response_cache:
        print(f"Returning cached response for {payload.id}")
        return {"response": response_cache[payload.id]}

    # Otherwise, generate a new response
    if payload.force:
        print(f"Forced regeneration for {payload.id}")
    elif payload.prompt:
        print(f"Generating response with custom prompt for {payload.id}")
    else:
        print(f"Generating new response for {payload.id} (not in cache)")

    response_text = ai_responder.generate_ai_response(
        item_id=payload.id,
        text=payload.text,
        custom_prompt=payload.prompt,
        rating=payload.rating,
        product_name=payload.productName,
        advantages=getattr(payload, 'advantages', None),
        pluses=getattr(payload, 'pluses', None),
        minuses=getattr(payload, 'minuses', None),
    )

    # Always cache the newly generated response, regardless of whether a prompt was used.
    # Но не кэшируем ошибки
    if response_text and not response_text.startswith("Ошибка") and not response_text.startswith("API-ключ") and not response_text.startswith("Не удалось"):
        response_cache[payload.id] = response_text
        save_cache()
        print(f"Updated cache for {payload.id}")
    else:
        print(f"Not caching error response for {payload.id}: {response_text}")

    return {"response": response_text}

@app.post("/api/cache-selected-response")
async def cache_selected_response(payload: dict):
    """
    Endpoint для сохранения выбранного пользователем ответа в кэш
    """
    item_id = payload.get("id")
    selected_response = payload.get("response")
    
    if not item_id or not selected_response:
        raise HTTPException(status_code=400, detail="Missing id or response")
    
    # Сохраняем выбранный ответ в кэш
    response_cache[item_id] = selected_response
    save_cache()
    print(f"Cached user-selected response for {item_id}")
    
    return {"status": "success", "message": "Response cached successfully"}

@app.post("/api/reply")
async def send_reply(payload: ReplyPayload):
    """
    Endpoint to send a reply to a feedback or question.
    """
    print(f"Received request to reply to {payload.type} {payload.id}")

    reply_text = payload.text
    reply_state = payload.state

    if payload.type == "questions":
        if payload.answer and isinstance(payload.answer, dict):
            reply_text = payload.answer.get("text") or reply_text
        if not reply_state:
            reply_state = "wbRu"

    if not reply_text:
        raise HTTPException(status_code=400, detail="Reply text is required")

    success = wb_api.reply_to_item(
        item_id=payload.id,
        text=reply_text,
        item_type=payload.type,
        state=reply_state
    )

    if success:
        # If the reply was successful, remove the item from the cache
        if payload.id in response_cache:
            del response_cache[payload.id]
            save_cache()
            print(f"Removed item {payload.id} from cache after successful reply.")
        return {"status": "success", "message": "Reply sent successfully."}
    else:
        raise HTTPException(
            status_code=500, 
            detail="Failed to send reply via Wildberries API."
        )


@app.post("/api/auto-reply-5-stars-feedbacks")
async def auto_reply_5_stars_feedbacks(request: Request):
    """
    Эндпоинт для автоматической обработки всех НЕотвеченных отзывов:
    1) Генерирует ответы ИИ для всех таких отзывов (любой оценки).
    2) Кэширует все успешные ответы в response_cache.
    3) Отправляет ответы в WB ТОЛЬКО для отзывов с оценкой ровно 5 звёзд,
       соблюдая rate limit WB API: не более 3 запросов в секунду.

    Может работать как простой POST (для навыка Алисы) или с телом запроса от Алисы.
    """
    # Этот эндпоинт оставляем для ручного вызова (curl/браузер).
    # Для Алисы используйте /api/alice-webhook.
    return _run_auto_reply_5_stars_feedbacks_sync()

# The Nginx container now handles serving static files and the main index.html.
# These routes are no longer needed in the FastAPI application when containerized.
# app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

# @app.get("/")
# async def read_root():
#     return FileResponse('../frontend/index.html')

# @app.get("/{catchall:path}")
# async def read_index(catchall: str):
#     return FileResponse('../frontend/index.html') 