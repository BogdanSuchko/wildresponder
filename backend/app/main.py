from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import os

from . import wb_api
from . import ai_responder
from .models import GenerateResponsePayload, Feedback, Question, ReplyPayload
from typing import List, Dict

app = FastAPI()

# --- Cache Setup ---
CACHE_FILE = "response_cache.json"
response_cache: Dict[str, str] = {}

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                if content:
                    response_cache.update(json.loads(content))
                print(f"Loaded {len(response_cache)} items from cache.")
            except json.JSONDecodeError:
                print("Cache file is empty or corrupted, starting with an empty cache.")
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
        # Возвращаем один кэшированный ответ для всех моделей
        cached_response = response_cache[payload.id]
        return {
            "gemini": cached_response,
            "gemini_v2": cached_response,
            "gemini_v3": cached_response
        }
    
    # Генерируем новые варианты
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
        profile=os.getenv('AI_PROFILE', 'gemini')
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
        profile=os.getenv('AI_PROFILE', 'gemini')
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

# The Nginx container now handles serving static files and the main index.html.
# These routes are no longer needed in the FastAPI application when containerized.
# app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

# @app.get("/")
# async def read_root():
#     return FileResponse('../frontend/index.html')

# @app.get("/{catchall:path}")
# async def read_index(catchall: str):
#     return FileResponse('../frontend/index.html') 