import os
import requests
from typing import Dict, Optional, List
from dotenv import load_dotenv
from openai import OpenAI

GEMINI_API_URL = "https://gemini-proxy-worker.turtlelovecode.workers.dev/v1beta/models/gemini-2.5-flash:generateContent"
load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_BASE_URL = "https://api.cometapi.com/v1"

# Ленивый клиент для Comet
_comet_client: Optional[OpenAI] = None

def _get_comet_client() -> OpenAI:
    global _comet_client
    if _comet_client is None:
        _comet_client = OpenAI(base_url=COMET_BASE_URL, api_key=COMET_API_KEY)
    return _comet_client

def _build_prompt(text: str,
                  custom_prompt: Optional[str],
                  rating: Optional[int],
                  product_name: Optional[str],
                  advantages: Optional[List[str]] = None) -> str:
    """Builds the full text prompt for the Gemini API respecting rating/product context.

    Style and persona: reply as the seller of the shop on Wildberries (not WB support).
    Keep it concise and human, avoid excessive newlines and lists.
    """
    rating_context = ""
    if rating is not None:
        if rating < 3:
            rating_context = (
                "Это очень низкая оценка. Внимательно проанализируй текст отзыва. "
                "Если текст позитивный или нейтральный (например, 'все хорошо', 'спасибо'), "
                "вежливо предположи, что пользователь мог ошибиться с оценкой. "
                "Например: 'Благодарим за ваш отзыв! Возможно, вы случайно поставили низкую оценку. "
                "Если это так, будем признательны за её исправление. Если же у вас есть замечания, "
                "пожалуйста, дайте нам знать, и мы поможем'. "
                "Если же текст отзыва явно негативный, извинись и предложи решение проблемы."
            )
        elif rating == 3:
            rating_context += "Это средняя оценка. Будь вежлив, поблагодари за отзыв и спроси, что можно было бы улучшить."
        else:
            rating_context += "Это высокая оценка. Обязательно поблагодари за нее и за положительный отзыв."

    product_name_context = ""
    if product_name:
        product_name_context = (
            f"Контекст: товар, о котором идет речь, называется '{product_name}'. "
            f"Обязательно органично употреби это название в своем ответе."
        )

    style_context = (
        "Ты — ассистент продавца магазина, который продаёт на Wildberries. "
        "Отвечай от лица продавца (мы/наш магазин), не от лица поддержки Wildberries. "
        "Пиши живо и по‑человечески, без канцелярита, понятно и дружелюбно. "
        "Форматирование: 1–2 абзаца максимум, без лишних пустых строк и списков, 3–5 предложений."
    )

    # Advantages context
    advantages_context = ""
    if advantages:
        adv_list = ", ".join(a.strip() for a in advantages if a and a.strip())
        if adv_list:
            advantages_context = f"Покупатель отметил преимущества: {adv_list}. Учти это в ответе."

    if custom_prompt:
        system_content = (
            "Ты помогаешь продавцу отвечать на отзывы/вопросы покупателей на Wildberries. "
            f"{style_context} {rating_context} {product_name_context} {advantages_context} Учти следующий промпт от пользователя."
        )
        user_content = f"Промпт: '{custom_prompt}'.\n\nТекст: '{text}'"
        return f"{system_content}\n{user_content}"
    else:
        system_content = (
            "Ты помогаешь продавцу магазина отвечать на отзывы/вопросы покупателей на Wildberries. "
            f"{style_context} {rating_context} {product_name_context} {advantages_context} Всегда благодари за обратную связь."
        )
        user_content = f"Ответь на следующий отзыв/вопрос: '{text}'"
        return f"{system_content}\n{user_content}"

def _call_gemini(full_prompt: str, temperature: float = 1.0, timeout_seconds: int = 30) -> str:
    payload = {
        "contents": [
            {"parts": [{"text": full_prompt}]}
        ],
        "generationConfig": {"temperature": temperature}
    }
    headers = {"Content-Type": "application/json"}
    api_response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=timeout_seconds)
    api_response.raise_for_status()
    result = api_response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]


def _normalize_text(text: str) -> str:
    """Normalizes AI text to avoid excessive blank lines and trailing spaces."""
    try:
        # Normalize line endings
        normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        # Strip spaces per line
        normalized = '\n'.join(line.strip() for line in normalized.split('\n'))
        # Collapse multiple blank lines to a single blank line
        while '\n\n\n' in normalized:
            normalized = normalized.replace('\n\n\n', '\n\n')
        return normalized.strip()
    except Exception:
        return text


def _build_gpt5_prompt_concise(product_name: Optional[str], text: str, pluses: Optional[str] = None,
                               minuses: Optional[str] = None, advantages: Optional[List[str]] = None) -> str:
    # ЕДИНСТВЕННАЯ строка без системных инструкций, строго в нижнем регистре "ответь"
    # Формат: "ответь на отзыв расширенно <товар> <комментарий> Достоинства: ... Недостатки: ... Преимущества: ..."
    # Удаляем переводы строк из входных значений, чтобы всё было в одну строку
    def _oneline(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return " ".join(value.replace("\r", "\n").split())

    name_part = _oneline(product_name) or ""
    text_part = _oneline(text)
    pluses_part = f"Достоинства: {_oneline(pluses)}" if _oneline(pluses) else None
    minuses_part = f"Недостатки: {_oneline(minuses)}" if _oneline(minuses) else None
    advantages_list = [a for a in (advantages or []) if a and a.strip()]
    advantages_joined = ", ".join(a.strip() for a in advantages_list)
    advantages_part = f"Преимущества: {advantages_joined}" if advantages_joined else None

    parts_inline: List[str] = []
    if name_part:
        parts_inline.append(name_part)
    if text_part:
        parts_inline.append(text_part)
    # Порядок: комментарий → достоинства → недостатки → преимущества
    if pluses_part:
        parts_inline.append(pluses_part)
    if minuses_part:
        parts_inline.append(minuses_part)
    if advantages_part:
        parts_inline.append(advantages_part)

    inline_body = " ".join(parts_inline).strip()
    # Возвращаем ровно одну строку
    return f"ответь на отзыв расширенно {inline_body}".strip()

def _build_gpt5_messages(product_name: Optional[str],
                         text: str,
                         pluses: Optional[str] = None,
                         minuses: Optional[str] = None,
                         advantages: Optional[List[str]] = None,
                         custom_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Строит одиночное сообщение пользователя без system и примеров."""
    # Базовый лаконичный промпт пользователя
    prompt_text = _build_gpt5_prompt_concise(
        product_name=product_name,
        text=text,
        pluses=pluses,
        minuses=minuses,
        advantages=advantages,
    )
    if custom_prompt:
        prompt_text = f"{prompt_text}\n\nДополнительные указания: {custom_prompt}"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "Ты отвечаешь на отзывы клиентов. Пиши только текст ответа — без предисловий, без предложений переписать. Используй эмодзи естественно, как в живом общении 😊"},
        {"role": "user", "content": prompt_text}, # Пиши только ответ на отзыв, без какого-то лишнего текста и вопросов типа 'может переписать по другому?'
    ]
    return messages

def generate_ai_response(item_id: str,
                         text: str,
                         custom_prompt: Optional[str] = None,
                         rating: Optional[int] = None,
                         product_name: Optional[str] = None,
                         advantages: Optional[List[str]] = None,
                         pluses: Optional[str] = None,
                         minuses: Optional[str] = None,
                         profile: str = "gemini") -> str:
    """
    Generates a single response using the Gemini API.
    """
    print(f"Calling Gemini API for item_id: {item_id}")

    try:
        if profile == "gpt":
            # GPT-5 через Comet API, строгий простой промпт
            client = _get_comet_client()
            resp = client.chat.completions.create(
                model="gpt-5-chat-latest",
                messages=_build_gpt5_messages(
                    product_name=product_name,
                    text=text,
                    pluses=pluses,
                    minuses=minuses,
                    advantages=advantages,
                    custom_prompt=custom_prompt,
                ),
                temperature=1,
            )
            return _normalize_text(resp.choices[0].message.content)
        else:
            # Профиль по умолчанию — Gemini
            full_prompt = _build_prompt(
                text=text,
                custom_prompt=custom_prompt,
                rating=rating,
                product_name=product_name,
                advantages=advantages,
            )
            temperature = 1
            generated_text = _call_gemini(full_prompt, temperature=temperature)
            return _normalize_text(generated_text)
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return "Не удалось получить ответ от ИИ. Попробуйте еще раз."
    except (KeyError, IndexError) as e:
        print(f"Error parsing Gemini API response: {e}")
        return "Не удалось обработать ответ от ИИ. Структура ответа изменилась."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"Произошла непредвиденная ошибка: {e}" 


def generate_multiple_ai_responses(item_id: str,
                                   item_id_dup_safe: str = None,
                                   text: str = "",
                                   custom_prompt: Optional[str] = None,
                                   rating: Optional[int] = None,
                                   product_name: Optional[str] = None,
                                   advantages: Optional[List[str]] = None,
                                   pluses: Optional[str] = None,
                                   minuses: Optional[str] = None,
                                   profile: str = "gemini") -> Dict[str, str]:
    """
    Generates several alternative responses (3 variants) to show as choices in UI.
    Returned keys are arbitrary model labels expected by the frontend.
    """
    # item_id_dup_safe оставлен для обратной совместимости, не используется
    try:
        variants = {}
        if profile == "gpt":
            # Для простоты отдаём 3 одинаковых варианта от GPT-5 (можно варьировать температуру при желании)
            client = _get_comet_client()
            for label in ["gpt", "gpt_v2", "gpt_v3"]:
                try:
                    r = client.chat.completions.create(
                        model="gpt-5",
                        messages=_build_gpt5_messages(
                            product_name=product_name,
                            text=text,
                            pluses=pluses,
                            minuses=minuses,
                            advantages=advantages,
                            custom_prompt=custom_prompt,
                        ),
                        temperature=1,
                    )
                    variants[label] = _normalize_text(r.choices[0].message.content)
                except Exception as inner_e:
                    print(f"Variant generation failed for {label}: {inner_e}")
                    variants[label] = "Не удалось сгенерировать этот вариант. Попробуйте снова."
        else:
            base_prompt = _build_prompt(
                text=text,
                custom_prompt=custom_prompt,
                rating=rating,
                product_name=product_name,
                advantages=advantages,
            )
            temps = [0.9, 1.1, 1.3]
            labels = ["gemini", "gemini_v2", "gemini_v3"]
            for label, t in zip(labels, temps):
                try:
                    variants[label] = _normalize_text(_call_gemini(base_prompt, temperature=t))
                except Exception as inner_e:
                    print(f"Variant generation failed for {label}: {inner_e}")
                    variants[label] = "Не удалось сгенерировать этот вариант. Попробуйте снова."

        return variants
    except Exception as e:
        print(f"Failed to generate multiple responses: {e}")
        # Возвращаем один безопасный ответ, чтобы UI не упал
        single = generate_ai_response(
            item_id=item_id,
            text=text,
            custom_prompt=custom_prompt,
            rating=rating,
            product_name=product_name,
        )
        if profile == "gpt":
            return {"gpt": single, "gpt_v2": single, "gpt_v3": single}
        else:
            return {"gemini": single, "gemini_v2": single, "gemini_v3": single}