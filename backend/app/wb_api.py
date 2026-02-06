import os
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from .models import Feedback, Question

# Load .env from the 'backend' directory (where uvicorn is run)
load_dotenv()

WB_API_KEY = os.getenv("WB_API_KEY")
API_BASE_URL = "https://feedbacks-api.wildberries.ru/api/v1"

def get_wb_api_headers() -> Dict[str, str]:
    if not WB_API_KEY:
        raise ValueError("Wildberries API Key not found in .env file")
    return {
        "Authorization": f"Bearer {WB_API_KEY}",
        "Content-Type": "application/json"
    }

def _get_photo_url(nmId: int) -> Optional[str]:
    """
    Constructs the URL for the product's first image using the sharding logic.
    Updated with the latest WB basket ranges (as of 2025).
    """
    try:
        vol = nmId // 100000
        part = nmId // 1000
        
        # Актуальные диапазоны basket-серверов WB
        if 0 <= vol <= 143: basket = '01'
        elif 144 <= vol <= 287: basket = '02'
        elif 288 <= vol <= 431: basket = '03'
        elif 432 <= vol <= 719: basket = '04'
        elif 720 <= vol <= 1007: basket = '05'
        elif 1008 <= vol <= 1061: basket = '06'
        elif 1062 <= vol <= 1115: basket = '07'
        elif 1116 <= vol <= 1169: basket = '08'
        elif 1170 <= vol <= 1313: basket = '09'
        elif 1314 <= vol <= 1601: basket = '10'
        elif 1602 <= vol <= 1655: basket = '11'
        elif 1656 <= vol <= 1919: basket = '12'
        elif 1920 <= vol <= 2045: basket = '13'
        elif 2046 <= vol <= 2189: basket = '14'
        elif 2190 <= vol <= 2405: basket = '15'
        elif 2406 <= vol <= 2621: basket = '16'
        elif 2622 <= vol <= 2837: basket = '17'
        elif 2838 <= vol <= 3053: basket = '18'
        elif 3054 <= vol <= 3269: basket = '19'
        elif 3270 <= vol <= 3485: basket = '20'
        else: basket = '21'  # Для новых товаров за пределами известных диапазонов
        
        url = (f"https://basket-{basket}.wbbasket.ru/vol{vol}"
               f"/part{part}/{nmId}/images/tm/1.webp")
        return url
    except Exception as e:
        print(f"Error generating photo URL for {nmId}: {e}")
        return None

def _extract_advantages_from_item(item: Dict[str, Any]) -> Optional[List[str]]:
    """Best-effort извлечение преимуществ из разных вариантов ответов WB.
    Возвращает список строк или None.
    """
    if not isinstance(item, dict):
        return None

    # 1) Прямые кандидаты по ключам
    candidate_keys = [
        "advantages", "advantagesRu", "advantagesRU", "advantages_list",
        "advantagesList", "prosTags", "pros_list", "prosList", "benefits",
        "tags",  # осторожно, проверим содержимое ниже
        "bables",  # поле из вашего примера WB
    ]

    def normalize_list(value: Any) -> List[str]:
        result: List[str] = []
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").replace("\n", ",").split(",")]
            result = [p for p in parts if p]
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, str):
                    result.append(v.strip())
                elif isinstance(v, dict):
                    name = v.get("name") or v.get("title") or v.get("text")
                    if name:
                        result.append(str(name).strip())
        return [x for x in result if x]

    for key in candidate_keys:
        if key in item and item[key]:
            vals = normalize_list(item[key])
            # Для generic 'tags' используем только если это адекватный короткий список строк (< 12)
            if key == "tags" and (not all(isinstance(x, str) for x in item.get(key, [])) or len(vals) > 12):
                pass
            else:
                if vals:
                    # Удаляем дубли, сохраняя порядок
                    seen = set()
                    uniq = []
                    for v in vals:
                        if v.lower() not in seen:
                            seen.add(v.lower())
                            uniq.append(v)
                    return uniq or None

    return None


def get_unanswered_feedbacks() -> List[Feedback]:
    headers = get_wb_api_headers()
    params = {
        "isAnswered": "false",
        "take": 100, # Fetch up to 100
        "skip": 0,
        "order": "dateDesc"
    }
    try:
        response = requests.get(f"{API_BASE_URL}/feedbacks", headers=headers, params=params, timeout=2.0)
        response.raise_for_status()
        data = response.json().get("data", {}).get("feedbacks", [])

        # Нормализуем поля pros/cons -> pluses/minuses для совместимости с моделью
        normalized = []
        for item in data:
            if isinstance(item, dict):
                if item.get("pros") and not item.get("pluses"):
                    item["pluses"] = item.pop("pros")
                if item.get("cons") and not item.get("minuses"):
                    item["minuses"] = item.pop("cons")
                # Универсальный парсинг преимуществ
                adv_list = _extract_advantages_from_item(item)
                if adv_list:
                    item["advantages"] = adv_list
            normalized.append(item)

        feedbacks = [Feedback(**item) for item in normalized]
        
        # Add photo URLs
        for fb in feedbacks:
            fb.productDetails.photo = _get_photo_url(fb.productDetails.nmId)
            
        return feedbacks
    except requests.exceptions.RequestException as e:
        print(f"Error fetching feedbacks from WB API: {e}")
        return []
    except (KeyError, TypeError) as e:
        print(f"Error parsing feedback data: {e}")
        return []

def get_unanswered_questions() -> List[Question]:
    headers = get_wb_api_headers()
    params = {
        "isAnswered": "false",
        "take": 100, # Fetch up to 100
        "skip": 0,
        "order": "dateDesc"
    }
    try:
        response = requests.get(f"{API_BASE_URL}/questions", headers=headers, params=params, timeout=2.0)
        response.raise_for_status()
        data = response.json().get("data", {}).get("questions", [])

        questions = [Question(**item) for item in data]
        
        # Add photo URLs
        for q in questions:
            q.productDetails.photo = _get_photo_url(q.productDetails.nmId)
            
        return questions
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions from WB API: {e}")
        return []
    except (KeyError, TypeError) as e:
        print(f"Error parsing question data: {e}")
        return []

def reply_to_item(item_id: str, text: str, item_type: str, state: Optional[str] = None) -> bool:
    """
    Sends a reply to a feedback or a question.
    'item_type' should be 'feedbacks' or 'questions'.
    """
    if item_type not in ['feedbacks', 'questions']:
        raise ValueError("Invalid item_type specified. Must be 'feedbacks' or 'questions'.")

    headers = get_wb_api_headers()
    
    # Determine the correct URL based on the item type
    if item_type == 'feedbacks':
        url = f"{API_BASE_URL}/feedbacks/answer"
    else: # questions
        url = f"{API_BASE_URL}/questions"

    # Prepare the payload based on the item type
    if item_type == 'feedbacks':
        payload: Dict[str, Any] = {
            "id": item_id,
            "text": text
        }
    else: # questions
        payload: Dict[str, Any] = {
            "id": item_id,
            "answer": {
                "text": text
            },
            "state": state or "wbRu"
        }

    try:
        print(f"Sending reply for {item_type} {item_id}...")
        print(f"URL: {url}")
        print(f"Payload: {payload}")
        # Use POST for feedback answers, PATCH for questions
        if item_type == 'feedbacks':
            response = requests.post(url, headers=headers, json=payload, timeout=3.0)
        else:
            response = requests.patch(url, headers=headers, json=payload, timeout=3.0)
            
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        response.raise_for_status() # Will raise an exception for 4xx/5xx status
        print(f"Successfully replied to {item_type} {item_id}. Status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        # Log the detailed error message from WB API if available
        error_details = e.response.text if hasattr(e, 'response') and e.response else "No response from server"
        print(f"Error sending reply to {item_type} {item_id}: {e}. Details: {error_details}")
        return False 
