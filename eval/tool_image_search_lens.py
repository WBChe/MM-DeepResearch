import base64
import http.client
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from serpapi import GoogleSearch

from utils import save_image_with_smart_resize


# =========================
# Config
# =========================

LENS_LOG_PATH = "./search_log/serpapi_lens_log.jsonl"

CACHED_DATA_LIST = [
]

CACHE_DIR = "./cache_image_eval/eval_image_lens"

GITHUB_TOKEN = ""
OWNER = ''
REPO = '' 
BRANCH = "main"

DEFAULT_MAX_IMAGE_PIXELS = 224 * 224
DEFAULT_UPLOAD_RESIZE = 336


# =========================
# Utilities
# =========================

def _normalize_data_id(data_id: Any) -> str:
    return str(data_id)


def _normalize_query(query: str) -> str:
    return query.replace(" ", "_")


def _lens_query_key(data_id: Any, query: str) -> str:
    return f"{_normalize_data_id(data_id)}::{_normalize_query(query)}"


def append_jsonl(log_path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def resize_image_base64(
    image_base64: str,
    max_size: int = DEFAULT_UPLOAD_RESIZE,
    format: str = "JPEG",
) -> str:
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    w, h = image.size
    scale = min(max_size / w, max_size / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)

    if scale < 1.0:
        image = image.resize((new_w, new_h), Image.BICUBIC)

    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _fail_result(msg: str) -> Tuple[List[str], Dict[str, str]]:
    return [], {"result": msg}


def _success_text(titles: List[str]) -> str:
    titles = [t for t in titles if t]
    titles_str = "; ".join(titles) if titles else "N/A"
    return (
        "[Image Search Succeeded] Relevant image(s) have been successfully retrieved. "
        f"The associated title(s) are: {titles_str}. "
        "The retrieved visual evidence can now be used for downstream multimodal reasoning."
    )


def _default_fail_text(data_id: Any) -> str:
    return (
        f"[Image Search Failed] Cached data was found for data_id='{_normalize_data_id(data_id)}', "
        "but no valid image paths could be retrieved. "
        "The deep research agent will proceed without visual evidence."
    )


# =========================
# Cache Loading
# =========================

def load_cached_data(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    cached: Dict[str, Dict[str, Any]] = {}

    for path in paths:
        if not os.path.exists(path):
            print(f"[WARN] cache file not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        cached[_normalize_data_id(k)] = v
                elif isinstance(data, list):
                    for item in data:
                        data_id = item.get("data_id")
                        if data_id is not None:
                            cached[_normalize_data_id(data_id)] = item
            else:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    data_id = item.get("data_id")
                    if data_id is not None:
                        cached[_normalize_data_id(data_id)] = item

    return cached


def load_cached_query(log_path: str) -> Dict[str, Dict[str, Any]]:
    cached_query: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(log_path):
        return cached_query

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            data_id = d.get("data_id")
            query = d.get("query", "")
            if data_id is None or not query:
                continue
            cached_query[_lens_query_key(data_id, query)] = d

    return cached_query


CACHED_DATA = load_cached_data(CACHED_DATA_LIST)
CACHED_QUERY = load_cached_query(LENS_LOG_PATH)


# =========================
# GitHub Upload
# =========================

def upload_to_github(img_file_or_base64: str, file_name: str) -> Optional[str]:
    if not GITHUB_TOKEN:
        print("[WARN] GITHUB_TOKEN is empty, skip github upload.")
        return None

    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{file_name}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        if os.path.exists(img_file_or_base64):
            with open(img_file_or_base64, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        else:
            encoded_string = img_file_or_base64

        encoded_string = resize_image_base64(encoded_string, DEFAULT_UPLOAD_RESIZE)
        # print(1, img_file_or_base64)
        data = {
            "message": f"Upload {file_name} via Python script",
            "content": encoded_string,
            "branch": BRANCH,
        }

        response = requests.put(url, headers=headers, json=data, timeout=60)
        # print(2)
        cdn_url = f"https://cdn.jsdelivr.net/gh/{OWNER}/{REPO}@{BRANCH}/{file_name}"

        if response.status_code in {200, 201, 422}:
            if response.status_code == 422:
                print(f"[INFO] file already exists: {file_name}")
            else:
                print(f"[INFO] upload success: {file_name}")
            print('sucess')
            return cdn_url

        # print(f"[ERROR] upload failed {file_name}: {response.status_code} - {response.text}")
        return None

    except Exception as e:
        print(f"[ERROR] upload_to_github failed: {e}")
        return None


# =========================
# Search APIs
# =========================

def get_serpapi_image_response(public_url: str, query: str, search_api_key: str) -> Dict[str, Any]:
    params = {
        "engine": "google_lens",
        "url": public_url,
        "q": query,
        "api_key": search_api_key,
    }
    search = GoogleSearch(params)
    return search.get_dict()


def get_serper_image_response(public_url: str, query: str, search_api_key: str) -> Dict[str, Any]:
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "url": public_url,
        # "q": query,
    })
    headers = {
        "X-API-KEY": search_api_key,
        "Content-Type": "application/json",
    }
    conn.request("POST", "/lens", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))


# =========================
# Parsing Helpers
# =========================

def _download_search_images(
    data_id: Any,
    query: str,
    output_image_dir: str,
    retrievals: List[Dict[str, Any]],
    image_keys: Tuple[str, ...],
    topk: int,
) -> List[Dict[str, str]]:
    os.makedirs(output_image_dir, exist_ok=True)

    title_img_list: List[Dict[str, str]] = []
    safe_query = _normalize_query(query)
    norm_data_id = _normalize_data_id(data_id)

    for idx, retrieval in enumerate(retrievals):
        img_path = os.path.join(output_image_dir, f"{norm_data_id}_{idx}_{safe_query}.jpg")

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
    response_json: Dict[str, Any],
    output_image_dir: str,
    query: str,
    topk: int = 3,
) -> List[Dict[str, str]]:
    retrievals = response_json.get("organic", [])
    if not retrievals:
        print("[WARN] serper lens organic not found")
        return []

    return _download_search_images(
        data_id=data_id,
        query=query,
        output_image_dir=output_image_dir,
        retrievals=retrievals,
        image_keys=("imageUrl", "thumbnailUrl"),
        topk=topk,
    )


def parse_serpapi_image_response(
    data_id: Any,
    response_json: Dict[str, Any],
    output_image_dir: str,
    query: str,
    topk: int = 3,
) -> List[Dict[str, str]]:
    retrievals = response_json.get("visual_matches", [])
    if not retrievals:
        print("[WARN] serpapi visual_matches not found")
        return []

    return _download_search_images(
        data_id=data_id,
        query=query,
        output_image_dir=output_image_dir,
        retrievals=retrievals,
        image_keys=("image", "thumbnail"),
        topk=topk,
    )


# =========================
# Main Function
# =========================

def func_image_search_by_lens(
    data_id: Any,
    query: str,
    img_base64: Optional[str] = None,
    engine: str = "local",
    search_api_key: Optional[str] = None,
    topk: int = 3,
) -> Tuple[List[str], Dict[str, str]]:
    norm_data_id = _normalize_data_id(data_id)
    query_key = _lens_query_key(norm_data_id, query)

    # 1) local cache
    if engine == "local":
        cache_entry = CACHED_DATA.get(norm_data_id)
        if not cache_entry:
            return _fail_result(
                "[Image Search Failed]. "
                "Please proceed with reasoning using your internal knowledge, "
                "or leverage other available tools if needed."
            )

        cached_items = cache_entry.get("cached_data", [])
        if not cached_items:
            return _fail_result(
                f"[Image Search Failed] Cached entry found for data_id='{norm_data_id}', "
                "but no image data is available. "
                "The image retrieval step did not yield usable results."
            )

    # 2) online engines
    elif engine in {"serpapi", "serper"}:
        cached_items: List[Dict[str, Any]] = []
        search_results: Dict[str, Any] = {}

        if query_key in CACHED_QUERY:
            print("[INFO] lens log hit, load cached query result directly")
            cached_items = CACHED_QUERY[query_key].get("cached_data", [])

        elif norm_data_id in CACHED_DATA:
            cache_entry = CACHED_DATA[norm_data_id]
            cached_items = cache_entry.get("cached_data", [])
            search_results = cache_entry.get("serpapi_image_len_response", {})

        else:
            if not search_api_key:
                return _fail_result(
                    "[Image Search Failed]. Missing search_api_key. "
                    "Please proceed with reasoning using your internal knowledge."
                )
            if not img_base64:
                return _fail_result(
                    "[Image Search Failed]. Missing input image for online lens search."
                )

            try:
                public_url = upload_to_github(img_base64, f"{norm_data_id}.jpg")
                if not public_url:
                    return _fail_result(
                        "[Image Search Failed]. Failed to create public image URL."
                    )

                if engine == "serpapi":
                    search_results = get_serpapi_image_response(public_url, query, search_api_key)
                    cached_items = parse_serpapi_image_response(
                        data_id=norm_data_id,
                        response_json=search_results,
                        output_image_dir=CACHE_DIR,
                        query=query,
                        topk=topk,
                    )
                else:
                    search_results = get_serper_image_response(public_url, query, search_api_key)
                    cached_items = parse_serper_image_response(
                        data_id=norm_data_id,
                        response_json=search_results,
                        output_image_dir=CACHE_DIR,
                        query=query,
                        topk=topk,
                    )

                if "Invalid API key" in json.dumps(search_results, ensure_ascii=False):
                    return _fail_result("[Image Search Failed]. Invalid API key.")

                append_jsonl(
                    LENS_LOG_PATH,
                    {
                        "data_id": norm_data_id,
                        "query": query,
                        "public_url": public_url,
                        "serpapi_image_len_response": search_results,
                        "cached_data": cached_items,
                    },
                )

            except Exception as e:
                print(f"[ERROR] online lens search failed: {e}")
                return _fail_result(
                    "[Image Search Failed]. "
                    "Please proceed with reasoning using your internal knowledge, "
                    "or leverage other available tools if needed."
                )

        # serpapi/serper 回退到本地缓存
        if not cached_items:
            cache_entry = CACHED_DATA.get(norm_data_id)
            if cache_entry:
                cached_items = cache_entry.get("cached_data", [])

    else:
        return _fail_result(f"[Image Search Failed]. Unsupported engine: {engine}")

    result_image = [
        item.get("cached_images_path")
        for item in cached_items
        if item.get("cached_images_path")
    ][:topk]

    result_title = [
        item.get("cached_title")
        for item in cached_items
        if item.get("cached_title")
    ][:topk]

    if not result_image:
        return _fail_result(_default_fail_text(norm_data_id))

    return result_image, {
        "result": _success_text(result_title),
    }