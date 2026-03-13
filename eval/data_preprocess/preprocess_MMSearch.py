
import pandas as pd
import os
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import base64

from prompt import DEFAULT_SYSTEM_CONTENT, DEFAULT_USER_CONTENT_PREFIX


def process_single_row(row, current_split_name, row_index):
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    if row.get("query_image") == None:
        return None
        images = []
        question = row['query']
    else:
        tmp_image = row.get("query_image", '')
        tmp_image['path'] = None
        images = [tmp_image]
        question = '<image>'+ row['query']


    # Build prompt structure
    user_content = DEFAULT_USER_CONTENT_PREFIX.rstrip("\n") + question
    prompt = [{"role": "system", "content": DEFAULT_SYSTEM_CONTENT}, {"role": "user", "content": user_content}]

    ground_truth = row.get("gt_answer")
    candidate_answers = str(row['alternative_gt_answers'])
    reward_model_data = {'ground_truth': {'target': [ground_truth]}, 'style': 'rule', 'candidate_answers': candidate_answers}

    # Process data source
    data_source_tagged = "searchR1_MMSearch"

    # Build tools kwargs structure
    tools_kwargs = {
        "text_search": {
            "create_kwargs": {"ground_truth": {'target': [ground_truth]}, "question": question, "data_source": "searchR1_MMSearch", 'candidate_answers': candidate_answers, 'data_id': f'MMSearch_{row_index}'}
        },
        "image_search_by_lens": {
            "create_kwargs": {"ground_truth": {'target': [ground_truth]}, "question": question, "data_source": "searchR1_MMSearch", 'candidate_answers': candidate_answers, 'data_id': f'MMSearch_{row_index}'}
        },
        "image_search_by_text_query": {
            "create_kwargs": {"ground_truth": {'target': [ground_truth]}, "question": question, "data_source": "searchR1_MMSearch", 'candidate_answers': candidate_answers, 'data_id': f'MMSearch_{row_index}'}
        },
        "model_search": {
            "create_kwargs": {"ground_truth": {'target': [ground_truth]}, "question": question, "data_source": "searchR1_MMSearch", 'candidate_answers': candidate_answers, 'data_id': f'MMSearch_{row_index}'}
        },
    }

    # Build complete extra_info structure
    extra_info = {
        "index": f'MMSearch_{row_index}',
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "images": images,
            "agent_name": "tool_agent",
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": None,
        }
    )

if __name__ == "__main__":
    df_test = pd.read_parquet("CaraJ/MMSearch")

    # 3. 定义处理函数
    def apply_process_row(row, split_name="train"):
        return process_single_row(row, current_split_name=split_name, row_index=row.name)

    # 4. 分别处理 train 和 test
    df_test_processed = df_test.apply(lambda row: apply_process_row(row, split_name="test"), axis=1)

    print('ori image len', len(df_test_processed))
    df_test_processed = df_test_processed.dropna(how="all")
    print('processed image len', len(df_test_processed))

    test_path = "./data/MMSearch_test.parquet"

    df_test_processed.to_parquet(test_path, index=False)

    # print(f"✅ Saved {len(df_train_processed)} train samples to {train_path}")
    print(f"✅ Saved {len(df_test_processed)} test samples to {test_path}")