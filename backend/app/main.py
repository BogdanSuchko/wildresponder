from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import os
import time
import shutil

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
    # Проверяем, это запрос от Алисы или обычный POST
    is_alice_request = False
    try:
        body = await request.json()
        if body and "request" in body and "session" in body:
            is_alice_request = True
    except:
        # Если нет тела запроса или это не JSON - это обычный POST
        pass
    
    print("Starting auto-reply flow for unanswered feedbacks...")

    feedbacks: List[Feedback] = wb_api.get_unanswered_feedbacks()
    if not feedbacks:
        return {
            "status": "ok",
            "message": "Нет неотвеченных отзывов.",
            "total_feedbacks": 0,
            "generated": 0,
            "replied_5_stars": 0,
        }

    generated_count = 0
    cached_count = 0
    replied_count = 0
    errors: Dict[str, str] = {}

    # Сначала генерируем ответы для всех отзывов и кладём в кэш
    # Пропускаем те, для которых уже есть ответ в кэше
    for fb in feedbacks:
        item_id = fb.id
        
        # Проверяем, есть ли уже ответ в кэше
        if item_id in response_cache:
            print(f"Skipping generation for feedback {item_id} (rating={fb.productValuation}): already cached")
            cached_count += 1
            continue
        
        print(f"Generating AI response for feedback {item_id} (rating={fb.productValuation})")

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

        # Повторяем ту же логику, что и в /api/generate-response:
        # не кэшируем явные ошибки.
        if response_text and not response_text.startswith("Ошибка") and not response_text.startswith("API-ключ") and not response_text.startswith("Не удалось"):
            response_cache[item_id] = response_text
            generated_count += 1
        else:
            errors[item_id] = response_text or "Пустой ответ ИИ"

    # Сохраняем кэш со всеми успешно сгенерированными ответами
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
            # Если по какой-то причине текста нет (например, была ошибка генерации) — пропускаем
            print(f"Skip sending reply for {item_id}: no cached response")
            continue

        # rate limit: не более 3 запросов в секунду
        now = time.time()
        # если текущий секундный интервал закончился, сбрасываем счётчик
        if now - second_window_start >= 1:
            second_window_start = now
            sent_in_current_second = 0

        if sent_in_current_second >= 3:
            # Ждём до начала следующего "окна"
            sleep_time = 1 - (now - second_window_start)
            if sleep_time > 0:
                print(f"Rate-limit: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            second_window_start = time.time()
            sent_in_current_second = 0

        print(f"Sending auto reply for 5⭐ feedback {item_id}")
        success = wb_api.reply_to_item(
            item_id=item_id,
            text=reply_text,
            item_type="feedbacks",
        )

        if success:
            replied_count += 1
            sent_in_current_second += 1
            # После успешного ответа удаляем элемент из кэша, как и в ручном эндпоинте
            if item_id in response_cache:
                del response_cache[item_id]
                save_cache()
                print(f"Removed item {item_id} from cache after auto-reply.")
        else:
            errors[item_id] = "Не удалось отправить ответ в WB"

    result = {
        "status": "ok",
        "message": "Автообработка отзывов завершена.",
        "total_feedbacks": len(feedbacks),
        "generated": generated_count,
        "cached": cached_count,
        "replied_5_stars": replied_count,
        "errors": errors,
    }
    
    # Если это запрос от Алисы, верни ответ в формате Алисы
    if is_alice_request:
        # Формируем текст ответа для Алисы
        if result["total_feedbacks"] == 0:
            text = "Нет новых неотвеченных отзывов."
        else:
            text = (
                f"Обработка завершена. "
                f"Всего отзывов: {result['total_feedbacks']}. "
            )
            if result["generated"] > 0:
                text += f"Сгенерировано новых ответов: {result['generated']}. "
            if result["cached"] > 0:
                text += f"Уже было в кэше: {result['cached']}. "
            if result["replied_5_stars"] > 0:
                text += f"Отправлено ответов на 5-звёздочные отзывы: {result['replied_5_stars']}."
            else:
                text += "Нет 5-звёздочных отзывов для ответа."
        
        return {
            "response": {
                "text": text,
                "end_session": False
            },
            "version": "1.0"
        }
    
    # Иначе верни обычный JSON
    return result

# The Nginx container now handles serving static files and the main index.html.
# These routes are no longer needed in the FastAPI application when containerized.
# app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

# @app.get("/")
# async def read_root():
#     return FileResponse('../frontend/index.html')

# @app.get("/{catchall:path}")
# async def read_index(catchall: str):
#     return FileResponse('../frontend/index.html') 