import os
from typing import Dict, Optional, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_BASE_URL = "https://api.cometapi.com/v1"

# Ленивый клиент для Comet
_comet_client: Optional[OpenAI] = None


def _get_comet_client() -> OpenAI:
    global _comet_client
    if _comet_client is not None:
        return _comet_client

    if not COMET_API_KEY:
        raise RuntimeError("COMET_API_KEY is not configured")

    _comet_client = OpenAI(base_url=COMET_BASE_URL, api_key=COMET_API_KEY)
    return _comet_client

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


def _rating_context(rating: Optional[int]) -> str:
    if rating is None:
        return ""

    if rating < 3:
        return (
            "оценка низкая, проверь текст: если отзыв позитивный — мягко уточни про оценку; "
            "если негативный — извинись и предложи решение"
        )
    if rating == 3:
        return "оценка средняя, поблагодари и спроси, что можно улучшить"
    return "оценка высокая, обязательно поблагодари за позитив"


def _build_gpt5_prompt_concise(product_name: Optional[str], text: str, pluses: Optional[str] = None,
                               minuses: Optional[str] = None, advantages: Optional[List[str]] = None,
                               rating: Optional[int] = None) -> str:
    # ЕДИНСТВЕННАЯ строка без системных инструкций, строго в нижнем регистре "ответь"
    # Формат: "ответь на отзыв расширенно <товар> <комментарий> Достоинства: ... Недостатки: ... Преимущества: ..."
    # Удаляем переводы строк из входных значений, чтобы всё было в одну строку
    def _oneline(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return " ".join(value.replace("\r", "\n").split())

    name_part = _oneline(product_name) or ""
    text_part = _oneline(text)
    rating_part = _rating_context(rating)
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
    if rating_part:
        parts_inline.append(rating_part)
    # Порядок: комментарий → рейтинг → достоинства → недостатки → преимущества
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
                         custom_prompt: Optional[str] = None,
                         rating: Optional[int] = None) -> List[Dict[str, str]]:
    """Строит одиночное сообщение пользователя без system и примеров."""
    # Базовый лаконичный промпт пользователя
    prompt_text = _build_gpt5_prompt_concise(
        product_name=product_name,
        text=text,
        pluses=pluses,
        minuses=minuses,
        advantages=advantages,
        rating=rating,
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
                         minuses: Optional[str] = None) -> str:
    print(f"Calling GPT-5 for item_id: {item_id}")

    try:
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
                rating=rating,
            ),
            temperature=1,
        )
        return _normalize_text(resp.choices[0].message.content)

    except Exception as e:
        print(f"Error calling GPT-5: {e}")
        return "Не удалось получить ответ от ИИ. Попробуйте еще раз."


def generate_multiple_ai_responses(item_id: str,
                                   text: str = "",
                                   custom_prompt: Optional[str] = None,
                                   rating: Optional[int] = None,
                                   product_name: Optional[str] = None,
                                   advantages: Optional[List[str]] = None,
                                   pluses: Optional[str] = None,
                                   minuses: Optional[str] = None) -> Dict[str, str]:
    try:
        client = _get_comet_client()
        variants: Dict[str, str] = {}
        for label in ["gpt", "gpt_v2", "gpt_v3"]:
            try:
                r = client.chat.completions.create(
                    model="gpt-5-chat-latest",
                    messages=_build_gpt5_messages(
                        product_name=product_name,
                        text=text,
                        pluses=pluses,
                        minuses=minuses,
                        advantages=advantages,
                        custom_prompt=custom_prompt,
                        rating=rating,
                    ),
                    temperature=1,
                )
                variants[label] = _normalize_text(r.choices[0].message.content)
            except Exception as inner_e:
                print(f"Variant generation failed for {label}: {inner_e}")
                variants[label] = "Не удалось сгенерировать этот вариант. Попробуйте снова."

        return variants
    except Exception as e:
        print(f"Failed to generate multiple responses: {e}")
        single = generate_ai_response(
            item_id=item_id,
            text=text,
            custom_prompt=custom_prompt,
            rating=rating,
            product_name=product_name,
            advantages=advantages,
            pluses=pluses,
            minuses=minuses,
        )
        return {"gpt": single, "gpt_v2": single, "gpt_v3": single}