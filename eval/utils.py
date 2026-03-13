from datasets import load_dataset
import json
import pdb
import pandas as pd
import base64

from qwen_vl_utils import process_vision_info, smart_resize
import subprocess
import copy
from multiprocessing import Pool
from functools import partial
from PIL import Image
from typing import Optional, Tuple
from io import BytesIO
import os
from prompt import SEARCH_SYS_PROMPT, SEARCH_USER_PROMPT

SPATIAL_MERGE_SIZE = 2

def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def save_image_with_smart_resize(
    image_url: str,
    save_path: str,
    image_patch_size: int = 14,
    max_image_pixels: int = 512 * 512,
) -> Optional[Tuple[int, int]]:
    """
    Download image from URL, smart resize if needed (max 512x512), and save.

    Returns:
        (width, height) if success, else None
    """
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)

    try:

        proc = subprocess.run(
            ["wget", "-q", "-O", "-", image_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=True,
        )

        with BytesIO(proc.stdout) as bio:
            image_obj = copy.deepcopy(Image.open(bio))

        image = to_rgb(image_obj)

        # ---------- smart resize ----------
        width, height = image.size
        cur_pixels = width * height

        if cur_pixels > max_image_pixels:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=patch_factor,
                max_pixels=max_image_pixels,
            )
            image = image.resize((resized_width, resized_height))

        # ---------- save ----------
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

        return save_path, image.size  # (w, h)

    except Exception as e:
        print(e)
        return None, None

def get_message(data):
    question = data['extra_info']['question']
    answer = data['reward_model']['ground_truth']['target'][0]
    img_base64 = base64.b64encode(data["images"][0]["bytes"]).decode("utf-8")
    img_messages = [{"type": "image_url", "image_url": {"url": f'data:image/jpeg;base64,{img_base64}'}}]
    messages = [
            {"role": "system", "content": [{"type": "text", "text": 'You are a helpful and harmless assistant.'}]},
            {"role": "user", "content": [{"type": "text", "text": question}] + img_messages}
        ]
    return messages, answer

def get_message_search(data):
    question = data['prompt'][1]['content']
    answer = data['reward_model']['ground_truth']['target'][0]

    img_base64 = base64.b64encode(data["images"][0]["bytes"]).decode("utf-8")
    img_messages = [{"type": "image_url", "image_url": {"url": f'data:image/jpeg;base64,{img_base64}'}}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SEARCH_SYS_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": question}] + img_messages}
    ]
    return messages, answer, img_base64

def read_data(data_path):
    return load_dataset('parquet', data_files=data_path, split='train')

if __name__ == '__main__':
    read_data(data_path)