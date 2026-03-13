import json
import os
import re
import http.client
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI
from serpapi import GoogleSearch


OPENAI_API_KEY = os.getenv("JUDGE_OPENAI_API_KEY", "EMPTY")
OPENAI_API_BASE = os.getenv("JUDGE_OPENAI_API_BASE", "http://10.124.162.81:9000/v1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "judge_model")

SERPAPI_LOG_PATH_LIST = [
    "./search_log/eval_log_serpapi_result.jsonl"
]

LOCAL_RETRIEVE_URL = os.getenv("LOCAL_TEXT_RETRIEVE_URL", "http://localhost:8000/retrieve")
LOCAL_TIMEOUT = 60
WEB_TIMEOUT = 60
MAX_PAGE_CHARS = 262144  # 控制送进 judge/summarize 的文本长度


client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)


def _normalize_query(query: str) -> str:
    return query.strip()


def _cache_key(engine: str, query: str) -> str:
    return f"{engine}::{_normalize_query(query)}"


def _safe_yes_no(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if re.fullmatch(r"\s*yes\s*[\.\!]*\s*", text, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"\s*no\s*[\.\!]*\s*", text, flags=re.IGNORECASE):
        return False

    m = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    return bool(m and m.group(1).lower() == "yes")


def _call_judge_model(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    try:
        chat_response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return (chat_response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[WARN] judge model call failed: {e}")
        return ""


CACHED_QUERY: Dict[str, Dict[str, Any]] = {}
for serpapi_log_path in SERPAPI_LOG_PATH_LIST:
    if not os.path.exists(serpapi_log_path):
        continue
    with open(serpapi_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            query = d.get("query")
            engine = d.get("engine", "serpapi")
            if query:
                CACHED_QUERY[_cache_key(engine, query)] = d


def is_webpage_useful(query: str, webpage_content: str) -> bool:
    system_prompt = """You are an evaluator of webpage usefulness.

Your task is to determine whether the given webpage content is relevant to the query and provides useful information to help answer it.

Evaluation criteria:
1. Relevance: The webpage content matches the topic, intent, or key entities of the query.
2. Usefulness: The webpage contains concrete information that can help answer or resolve the query.

Output rules:
- Output "Yes" if the webpage is relevant and useful.
- Output "No" if the webpage is irrelevant, off-topic, too vague, or not helpful.
- Output only "Yes" or "No". Do not provide explanations."""
    user_prompt = f"""Query:
{query}

Webpage Content:
{webpage_content[:MAX_PAGE_CHARS]}

Does the webpage content relate to the query and provide useful information to address it?"""

    response = _call_judge_model(system_prompt, user_prompt, temperature=0.2)
    return _safe_yes_no(response)


def summarize_webpage(query: str, webpage_content: str) -> str:
    system_prompt = """You are an expert information summarizer.

Your task is to read the webpage content and extract or summarize only the information that is directly relevant to answering the user's query.

Guidelines:
- Focus on content that helps answer the query.
- Exclude irrelevant background, navigation text, ads, or metadata.
- Do not add assumptions, explanations, or commentary.
- Do not include phrases like "according to the webpage" or "the text says".
- Return the relevant information as plain, concise text."""
    user_prompt = f"""Query:
{query}

Webpage Content:
{webpage_content[:MAX_PAGE_CHARS]}

Task:
Extract and summarize the information from the webpage that is relevant to answering the query."""

    return _call_judge_model(system_prompt, user_prompt, temperature=0.2)


def call_serpapi(query: str, topk: int, search_api_key: str) -> Tuple[List[str], List[str]]:
    params = {
        "q": query,
        "google_domain": "google.com",
        "api_key": search_api_key,
    }
    try:
        search = GoogleSearch(params)
        raw_results = search.get_dict()
    except Exception as e:
        print(f"[WARN] call_serpapi failed: {e}")
        return [], []

    organic_results = raw_results.get("organic_results", [])
    if not organic_results:
        print("[WARN] No organic results in SERPAPI response.")
        return [], []

    links = [item.get("link") for item in organic_results if item.get("link")]
    titles = [item.get("title") for item in organic_results if item.get("title")]
    return links[:topk], titles[:topk]


def call_serper(query: str, topk: int, search_api_key: str) -> Tuple[List[str], List[str]]:
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": search_api_key,
        "Content-Type": "application/json",
    }
    try:
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        data_json = json.loads(data.decode("utf-8"))
    except Exception as e:
        print(f"[WARN] call_serper failed: {e}")
        return [], []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    organic = data_json.get("organic", [])
    if not organic:
        print("[WARN] No organic results in SERPER response.")
        return [], []

    links = [item.get("link") for item in organic if item.get("link")]
    titles = [item.get("title") for item in organic if item.get("title")]
    return links[:topk], titles[:topk]


def _fetch_jina_pages(links: List[str], titles: List[str], jina_api_key: Optional[str]) -> List[str]:
    results: List[str] = []

    for title, link in zip(titles, links):
        url = f"https://r.jina.ai/{link}"
        headers = {}
        if jina_api_key:
            headers["Authorization"] = jina_api_key

        try:
            response = requests.get(url, headers=headers, timeout=WEB_TIMEOUT)
        except Exception as e:
            print(f"[WARN] jina fetch failed for {link}: {e}")
            continue

        text = response.text or ""
        if '{"data":null,"code":402' in text:
            print("[WARN] jina returned 402 / quota issue")
            continue

        results.append(f"{title}\n{text}")

    return results


def fetch_webpage(
    query: str,
    topk: int = 5,
    engine: str = "local",
    search_api_key: Optional[str] = None,
    jina_api_key: Optional[str] = None,
) -> List[Any]:
    query = _normalize_query(query)

    if engine == "local":
        payload = {
            "queries": [query],
            "topk": topk,
            "return_scores": True,
        }
        try:
            resp = requests.post(LOCAL_RETRIEVE_URL, json=payload, timeout=LOCAL_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("result", [])
            return results[0] if results else []
        except Exception as e:
            print(f"[WARN] local retrieve failed: {e}")
            return []

    if engine in {"serpapi", "serper"}:
        cache_key = _cache_key(engine, query)
        if cache_key in CACHED_QUERY:
            print(f"[INFO] cached query hit: {cache_key}")
            return CACHED_QUERY[cache_key].get("jina", [])

        if not search_api_key:
            print("[WARN] search_api_key is required for serpapi/serper")
            return []

        if engine == "serpapi":
            links, titles = call_serpapi(query, topk, search_api_key)
        else:
            links, titles = call_serper(query, topk, search_api_key)

        if not links:
            return []

        return _fetch_jina_pages(links, titles, jina_api_key)

    raise NotImplementedError(f"search engine not implemented: {engine}")


def _format_local_doc(doc_item: Dict[str, Any]) -> str:
    content = doc_item["document"]["contents"]
    title = content.split("\n")[0]
    text = "\n".join(content.split("\n")[1:])
    return f"(Title: {title}) {text}"


def _format_web_doc(doc_item: str) -> str:
    doc_item = doc_item[:MAX_PAGE_CHARS]
    title = doc_item.split("\n")[0]
    text = "\n".join(doc_item.split("\n")[1:])
    return f"(Title: {title}) {text}"


def _passages2string(retrieval_result: List[Any], query: str) -> str:
    format_reference = ""
    idx = 1

    for doc_item in retrieval_result:
        try:
            if isinstance(doc_item, dict):
                summarized_doc_item = _format_local_doc(doc_item)
            elif isinstance(doc_item, str):
                title_text = _format_web_doc(doc_item)

                # 先做一个很轻量的过滤
                if len(title_text.strip()) < 50:
                    continue

                if not is_webpage_useful(query, title_text):
                    continue

                summarized_doc_item = summarize_webpage(query, title_text)
                if not summarized_doc_item.strip():
                    continue
            else:
                continue

            format_reference += f"Doc {idx}: {summarized_doc_item}\n"
            idx += 1
        except Exception as e:
            print(f"[WARN] failed to process passage: {e}")
            continue

    return format_reference


def search(
    query_list: List[str],
    engine: str = "local",
    search_api_key: Optional[str] = None,
    jina_api_key: Optional[str] = None,
    topk: int = 5,
) -> Dict[str, str]:
    pretty_results: List[str] = []

    for query in query_list:
        results = fetch_webpage(
            query,
            engine=engine,
            topk=topk,
            search_api_key=search_api_key,
            jina_api_key=jina_api_key,
        )
        formatted = _passages2string(results, query)
        if formatted:
            pretty_results.append(formatted)

    if pretty_results:
        return {"result": "\n---\n".join(pretty_results)}

    return {
        "result": (
            "Unable to retrieve useful information based on this query. "
            "Please rely on your internal capabilities to think about it and provide a direct answer."
        )
    }