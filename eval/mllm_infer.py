from openai import OpenAI
from typing import Any, Dict, List, Optional, Sequence, Tuple

QWEN_BASE_URLS = [
    "http://localhost:8000/v1",
]
QWEN_API_KEY = "EMPTY"
QWEN_MODEL = "qwen3vl"


def _build_qwen_client(base_url: Optional[str] = None) -> OpenAI:
    return OpenAI(
        api_key=QWEN_API_KEY,
        base_url=base_url or QWEN_BASE_URLS[0],
    )


def _extract_content(resp: Any) -> str:
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _parse_stop_state(resp: Any, content: str) -> Tuple[str, bool]:
    choice = resp.choices[0]

    matched_stop = getattr(choice, "matched_stop", None)
    if matched_stop is not None:
        matched_stop = str(matched_stop)
        is_end = "tool_call" not in matched_stop
        text = content if is_end else f"{content}{matched_stop}"
        return text.strip(), is_end

    stop_reason = getattr(choice, "stop_reason", None)
    if stop_reason is None:
        return content.strip(), True

    stop_reason = str(stop_reason)
    is_end = "tool_call" not in stop_reason
    text = content if is_end else f"{content}{stop_reason}"
    return text.strip(), is_end


def chat_qwen3vl_offline(
    messages: List[Dict[str, Any]],
    gt=None,
    times: int = 1,
    base_url: Optional[str] = None,
    model: str = QWEN_MODEL,
    temperature: float = 0.6,
    max_tokens: Optional[int] = None,
) -> Tuple[str, bool]:
    client = _build_qwen_client(base_url)

    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        resp = client.chat.completions.create(**kwargs)
        return _extract_content(resp), True

    except Exception as e:
        print(f"infer error: {e}")
        return "ERROR", False


def chat_qwen3vl_offline_search(
    messages: List[Dict[str, Any]],
    stop: Optional[List[str]] = None,
    base_url: Optional[str] = None,
    model: str = QWEN_MODEL,
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> Tuple[str, bool]:
    client = _build_qwen_client(base_url)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
    except Exception as e:
        print(f"search infer error: {e}")
        return "", True

    content = _extract_content(resp)
    return _parse_stop_state(resp, content)