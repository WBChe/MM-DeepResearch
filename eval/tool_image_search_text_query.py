import json
import os
import http.client
from typing import Any, Dict, List, Optional, Tuple

import requests
from serpapi import GoogleSearch

from utils import save_image_with_smart_resize


LOG_PATH = "./search_log/serpapi_image_query_log.jsonl"
CACHED_IMG_DIR = "./cache_image_eval/eval_image_query"
LOCAL_RETRIEVE_URL = "http://localhost:8001/retrieve"
DEFAULT_MAX_IMAGE_PIXELS = 224 * 224


def _normalize_data_id(data_id: Any) -> str:
    return str(data_id)


def _normalize_query(query: str) -> str:
    return query.strip().replace(" ", "_")


def _cache_key(data_id: Any, query: str) -> str:
    return f"{_normalize_data_id(data_id)}::{_normalize_query(query)}"


def _fail_result(msg: str) -> Tuple[List[str], Dict[str, str]]:
    return [], {"result": msg}


def _success_result(titles: List[str]) -> Tuple[List[str], Dict[str, str]]:
    titles = [t for t in titles if t]
    titles_str = "; ".join(titles) if titles else "N/A"
    return [], {
        "result": (
            "[Image Search Succeeded] Relevant image(s) have been successfully retrieved. "
            f"The associated title(s) are: {titles_str}. "
            "The retrieved visual evidence can now be used for downstream multimodal reasoning."
        )
    }


def append_jsonl(log_path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_cached_query(log_path: str) -> Dict[str, Dict[str, Any]]:
    cached: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(log_path):
        return cached

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue

            data_id = d.get("data_id")
            query = d.get("query")
            if data_id is None or not query:
                continue

            cached[_cache_key(data_id, query)] = d
    return cached


CACHED_QUERY = load_cached_query(LOG_PATH)


def get_serper_image_response(query: str, search_api_key: str) -> Dict[str, Any]:
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": search_api_key,
        "Content-Type": "application/json",
    }
    conn.request("POST", "/images", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))


def get_serpapi_image_response(query: str, search_api_key: str) -> List[Dict[str, Any]]:
    params = {
        "engine": "google_images_light",
        "q": query,
        "api_key": search_api_key,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("images_results", [])


def _collect_images_from_results(
    data_id: Any,
    query: str,
    retrievals: List[Dict[str, Any]],
    image_keys: Tuple[str, ...],
    topk: int,
) -> List[Dict[str, str]]:
    os.makedirs(CACHED_IMG_DIR, exist_ok=True)

    norm_data_id = _normalize_data_id(data_id)
    norm_query = _normalize_query(query)
    title_img_list: List[Dict[str, str]] = []

    for idx, retrieval in enumerate(retrievals):
        img_path = os.path.join(CACHED_IMG_DIR, f"{norm_data_id}_q{norm_query}_{idx}.jpg")

        # 已存在直接复用
        if os.path.exists(img_path):
            title_img_list.append({
                "cached_title": retrieval.get("title"),
                "cached_images_path": img_path,
            })
            if len(title_img_list) >= topk:
                break
            continue

        saved_path = None
        for key in image_keys:
            image_url = retrieval.get(key)
            if not image_url:
                continue
            try:
                saved_path, _ = save_image_with_smart_resize(
                    image_url,
                    img_path,
                    max_image_pixels=DEFAULT_MAX_IMAGE_PIXELS,
                )
                break
            except Exception:
                continue

        if saved_path is None:
            continue

        title_img_list.append({
            "cached_title": retrieval.get("title"),
            "cached_images_path": saved_path,
        })
        if len(title_img_list) >= topk:
            break

    return title_img_list


def parse_serper_image_response(
    data_id: Any,
    search_results: Dict[str, Any],
    query: str,
    topk: int = 3,
) -> List[Dict[str, str]]:
    retrievals = search_results.get("images", [])
    if not retrievals:
        return []

    return _collect_images_from_results(
        data_id=data_id,
        query=query,
        retrievals=retrievals,
        image_keys=("imageUrl", "thumbnailUrl", "thumbnail"),
        topk=topk,
    )


def parse_serpapi_image_response(
    data_id: Any,
    search_results: List[Dict[str, Any]],
    query: str,
    topk: int = 3,
) -> List[Dict[str, str]]:
    if not search_results:
        return []

    return _collect_images_from_results(
        data_id=data_id,
        query=query,
        retrievals=search_results,
        image_keys=("original", "serpapi_thumbnail", "thumbnail"),
        topk=topk,
    )


def _search_local(query_list: List[str], topk: int) -> Tuple[List[str], List[str]]:
    payload = {
        "queries": query_list,
        "topk": topk,
        "return_scores": True,
    }

    try:
        resp = requests.post(LOCAL_RETRIEVE_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        raw_results = data.get("result", [])
    except Exception as e:
        print(f"[image_search] local request failed: {e}")
        return [], []

    result_image: List[str] = []
    result_title: List[str] = []

    for retrieval in raw_results:
        if not retrieval:
            continue
        for doc_item in retrieval:
            if not isinstance(doc_item, dict):
                continue
            if doc_item.get("score", 0.0) < 0.7:
                continue

            doc = doc_item.get("document", {})
            img_path = doc.get("img_path")
            title = doc.get("text")

            if img_path:
                result_image.append(img_path)
            if title:
                result_title.append(title)

    return result_image[:topk], result_title[:topk]


def _search_online_single_query(
    data_id: Any,
    query: str,
    engine: str,
    search_api_key: str,
    topk: int,
) -> List[Dict[str, str]]:
    key = _cache_key(data_id, query)

    if key in CACHED_QUERY:
        return CACHED_QUERY[key].get("cached_data", [])

    if engine == "serper":
        search_results = get_serper_image_response(query, search_api_key)
        cached_items = parse_serper_image_response(
            data_id=data_id,
            search_results=search_results,
            query=query,
            topk=topk,
        )
    elif engine == "serpapi":
        search_results = get_serpapi_image_response(query, search_api_key)
        cached_items = parse_serpapi_image_response(
            data_id=data_id,
            search_results=search_results,
            query=query,
            topk=topk,
        )
    else:
        raise NotImplementedError(f"engine {engine} not implemented")

    append_jsonl(
        LOG_PATH,
        {
            "data_id": _normalize_data_id(data_id),
            "query": query,
            "cached_data": cached_items,
            "search_response": search_results,
        },
    )

    CACHED_QUERY[key] = {
        "data_id": _normalize_data_id(data_id),
        "query": query,
        "cached_data": cached_items,
        "search_response": search_results,
    }
    return cached_items


def func_image_search_by_text_query(
    data_id: Any,
    query_list: List[str],
    engine: str = "local",
    search_api_key: Optional[str] = None,
    topk: int = 3,
) -> Tuple[List[str], Dict[str, str]]:
    fail_text = "Search request failed or timed out after retries."

    if not isinstance(query_list, list) or len(query_list) == 0:
        return _fail_result(fail_text)

    query_list = [q for q in query_list if isinstance(q, str) and q.strip()]
    if not query_list:
        return _fail_result(fail_text)

    result_image: List[str] = []
    result_title: List[str] = []

    if engine == "local":
        result_image, result_title = _search_local(query_list, topk)
    elif engine in {"serpapi", "serper"}:
        if not search_api_key:
            return _fail_result("Missing search_api_key for online image search.")

        for query in query_list:
            cached_items = _search_online_single_query(
                data_id=data_id,
                query=query,
                engine=engine,
                search_api_key=search_api_key,
                topk=topk,
            )
            for item in cached_items[:topk]:
                img_path = item.get("cached_images_path")
                title = item.get("cached_title")
                if img_path:
                    result_image.append(img_path)
                if title:
                    result_title.append(title)
    else:
        return _fail_result(f"Unsupported engine: {engine}")

    # 去重，保持顺序
    dedup_images = list(dict.fromkeys(result_image))[:topk]
    dedup_titles = list(dict.fromkeys(result_title))[:topk]

    if not dedup_images:
        return _fail_result(
            "Unable to retrieve useful information based on this query. "
            "Please rely on your internal capabilities to think about it and provide a direct answer."
        )

    titles_str = "; ".join(dedup_titles) if dedup_titles else "N/A"
    return dedup_images, {
        "result": (
            "[Image Search Succeeded] Relevant image(s) have been successfully retrieved. "
            f"The associated title(s) are: {titles_str}. "
            "The retrieved visual evidence can now be used for downstream multimodal reasoning."
        )
    }