from openai import OpenAI
import pdb
import re
from utils import *
import re
from prompt import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT

openai_api_key = "EMPTY"
openai_api_base = "http://10.124.162.81:9000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def compute_correctness(solution_str, ground_truth, question, candidate_answers=[]):
    try:
        chat_response = client.chat.completions.create(
            model='judge_model',
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_PROMPT.format(question=question, ground_truth_answer=ground_truth, candidate_answers=candidate_answers, model_response=solution_str)},
            ],
            temperature=0.01,  # Lower temperature for more deterministic judgement
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response = chat_response.choices[0].message.content.strip()
    except Exception as e:
        print('judge error')
        pdb.set_trace()
    # Parse LLM judge response
    if re.search(r"Yes", response, re.IGNORECASE):
        return True
    elif re.search(r"No", response, re.IGNORECASE):
        return False
    else:
        return False

