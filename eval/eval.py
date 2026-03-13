from PIL import Image
from qwen_vl_utils import smart_resize
import re
import os
import io
import json
import math
import base64
import argparse
from functools import partial
from multiprocessing import Pool
from collections import Counter

import regex
from pydantic import BaseModel
from tqdm import tqdm

from llm_judge import compute_correctness
from utils import *
from mllm_infer import *
from tool_image_search_lens import func_image_search_by_lens
from tool_image_search_text_query import func_image_search_by_text_query
from tool_text_search import search


def split_corpus_into_chunks(corpus, chunk_num: int):
    assert chunk_num > 0
    n = len(corpus)
    chunk_size = math.ceil(n / chunk_num)

    chunks = []
    for i in range(chunk_num):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        if start >= n:
            break

        if hasattr(corpus, "select"):
            chunks.append(corpus.select(range(start, end)))
        else:
            chunks.append(corpus[start:end])

    return chunks


def resize_and_base64(img_path, max_image_pixels=128 * 128):
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    with Image.open(img_path) as img:
        w, h = img.size

        if w * h > max_image_pixels:
            h2, w2 = smart_resize(h, w, factor=28, max_pixels=max_image_pixels)
            img = img.resize((w2, h2))

        if img.mode != "RGB":
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


class FunctionCall(BaseModel):
    arguments: str
    name: str


def extract_solution(solution_str: str) -> str:
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return solution_str[-4000:]
    return matches[-1].group(1).strip()


def get_tool(text):
    try:
        tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)
        matches = tool_call_regex.findall(text)
        if not matches:
            return None, None

        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name = function_call["name"]
                arguments = function_call["arguments"]
                function_calls.append(
                    FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False))
                )
            except Exception as e:
                print(f"[WARN] Failed to decode tool call: {e}")

        if not function_calls:
            return None, None

        args_json = json.loads(function_calls[0].arguments)
        query_list = args_json.get("query_list")
        tool_name = function_calls[0].name
        return tool_name, query_list
    except Exception as e:
        print(f"[WARN] get_tool error: {e}")
        return None, None


def infer_search(messages, img_base64, data_id, args):
    final_output = ""
    max_turn = 7
    target_sequences = [
        "</tool_call>", " </tool_call>",
        "</tool_call>\n", " </tool_call>\n",
        "</tool_call>\n\n", " </tool_call>\n\n"
    ]
    curr_search_template = "<tool_response>{search_results}</tool_response>"

    while max_turn > 0:
        output_text, is_end = chat_qwen3vl_offline_search(
            messages=messages,
            stop=target_sequences,
        )

        if is_end:
            final_output += output_text
            break

        tool_name, cur_query = get_tool(output_text)

        if tool_name is None:
            result_text = "Search request failed or timed out after retries."
            search_text = curr_search_template.format(search_results=result_text)
            cur_messages = [
                {"role": "assistant", "content": [{"type": "text", "text": output_text}]},
                {"role": "user", "content": [{"type": "text", "text": search_text}]},
            ]

        elif tool_name in {"search", "text_search"}:
            result_text = search(
                cur_query,
                engine=args.engine,
                search_api_key=args.search_api_key,
                jina_api_key=args.jina_api_key,
                topk=args.topk_text_search,
            )
            search_text = curr_search_template.format(search_results=result_text)
            cur_messages = [
                {"role": "assistant", "content": [{"type": "text", "text": output_text}]},
                {"role": "user", "content": [{"type": "text", "text": search_text}]},
            ]

        elif tool_name in {"image_search_by_lens", "image_search_by_text_query"}:
            if tool_name == "image_search_by_lens":
                result_image, result_text = func_image_search_by_lens(
                    data_id,
                    cur_query[0],
                    img_base64=img_base64,
                    engine=args.engine,
                    search_api_key=args.search_api_key,
                    topk=args.topk_image_search_lens,
                )
            else:
                result_image, result_text = func_image_search_by_text_query(
                    data_id,
                    cur_query,
                    engine=args.engine,
                    search_api_key=args.search_api_key,
                    topk=args.topk_image_search,
                )

            img_base64_list = []
            for cur_img_path in result_image:
                try:
                    img_base64_list.append(resize_and_base64(cur_img_path))
                except Exception as e:
                    print(f"[WARN] Failed to process image {cur_img_path}: {e}")

            img_messages = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{cur_img_base64}"}
                }
                for cur_img_base64 in img_base64_list
            ]

            search_text = curr_search_template.format(search_results=result_text)
            cur_messages = [
                {"role": "assistant", "content": [{"type": "text", "text": output_text}]},
                {"role": "user", "content": [{"type": "text", "text": search_text}] + img_messages},
            ]
        else:
            result_text = f"Unsupported tool: {tool_name}"
            search_text = curr_search_template.format(search_results=result_text)
            cur_messages = [
                {"role": "assistant", "content": [{"type": "text", "text": output_text}]},
                {"role": "user", "content": [{"type": "text", "text": search_text}]},
            ]

        messages = messages + cur_messages
        final_output += output_text + search_text
        max_turn -= 1

    return final_output


def infer_and_eval_search(data, times, args):
    messages, answer, img_base64 = get_message_search(data)
    data_id = data["extra_info"]["index"]

    corr_list = []
    output_list = []
    question = messages[1]["content"][0]["text"]
    candidate_answers = data.get("reward_model", {}).get("candidate_answers")

    while len(corr_list) < times:
        final_output = infer_search(messages, img_base64, data_id, args)
        solution = extract_solution(final_output)
        output_list.append(final_output)

        is_correct = compute_correctness(
            solution,
            answer,
            question,
            candidate_answers
        )
        corr_list.append(1 if is_correct else 0)

    return {"output": output_list, "judge_list": corr_list}


def infer_and_eval(data, times, model_func):
    messages, gt = get_message(data)

    corr_list = []
    output_list = []
    question = messages[1]["content"][0]["text"]

    while len(corr_list) < times:
        result, infer_success = model_func(messages, gt, times=times)
        output_list.append(result)

        is_correct = infer_success and compute_correctness(result, gt, question)
        corr_list.append(1 if is_correct else 0)

    return {"output": output_list, "judge_list": corr_list}


def process_line(data, times, enable_search, model_type, args):
    model_func_map = {
        "qwen3vl": chat_qwen3vl_offline,
    }
    model_func = model_func_map[model_type]

    if enable_search:
        res = infer_and_eval_search(data, times, args)
    else:
        res = infer_and_eval(data, times, model_func)

    out = dict(data)
    out["req_func"] = model_func.__name__
    out["req_model"] = args.model_name
    out["output"] = res

    for k in ["image", "images", "image_preview", "rationale_image"]:
        out.pop(k, None)

    return out


def parse_args():
    parser = argparse.ArgumentParser(description="ValleyOmni Infer Server")
    parser.add_argument("--model-name", type=str, default="qwen3vl")
    parser.add_argument("--model-type", type=str, default="qwen3vl")
    parser.add_argument("--pool-size", type=int, default=1)
    parser.add_argument("--idx_key", type=str, default="uniid")
    parser.add_argument("--rollout_times", type=int, default=5)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--enable_search", action="store_true")
    parser.add_argument("--engine", type=str, default="local")
    parser.add_argument("--search_api_key", type=str, default=None)
    parser.add_argument("--jina_api_key", type=str, default=None)
    parser.add_argument("--topk_text_search", type=int, default=5)
    parser.add_argument("--topk_image_search", type=int, default=5)
    parser.add_argument("--topk_image_search_lens", type=int, default=5)
    parser.add_argument("--chunk_num", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, default=0)
    return parser.parse_args()


def cal_acc(data_path):
    distribution = Counter()

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            judge_list = data.get("output", {}).get("judge_list", [])
            one_count = sum(1 for x in judge_list if x == 1)
            distribution[one_count] += 1

    total = sum(distribution.values())
    if total == 0:
        print("No valid samples found.")
        return

    print(distribution)
    print(f"acc: {100 * distribution.get(1, 0) / total:.2f}%")
    print(distribution.get(1, 0), total)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    data = read_data(args.input_path)

    if args.chunk_num > 1:
        chunks = split_corpus_into_chunks(data, args.chunk_num)
        if args.chunk_id >= len(chunks):
            raise ValueError(f"chunk_id {args.chunk_id} out of range, total chunks={len(chunks)}")
        data = chunks[args.chunk_id]

    existid = set()
    if os.path.exists(args.save_path):
        with open(args.save_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    tmp_data = json.loads(line)
                    existid.add(tmp_data["extra_info"]["index"])
                except Exception:
                    continue

    filtered_data = []
    for d in data:
        cur_id = d["extra_info"]["index"]
        if cur_id not in existid:
            filtered_data.append(d)
    data = filtered_data

    print(f"Remaining samples: {len(data)}")

    func = partial(
        process_line,
        times=args.rollout_times,
        enable_search=args.enable_search,
        model_type=args.model_type,
        args=args,
    )

    with open(args.save_path, "a", encoding="utf-8") as fout:
        if args.pool_size <= 1:
            for d in tqdm(data, total=len(data)):
                try:
                    result = func(d)
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                except Exception as e:
                    print(f"[ERROR] Failed on sample {d.get('extra_info', {}).get('index')}: {e}")
        else:
            with Pool(args.pool_size) as pool:
                for result in tqdm(pool.imap_unordered(func, data), total=len(data)):
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()

    cal_acc(args.save_path)