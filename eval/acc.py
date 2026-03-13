import argparse
from collections import Counter
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Infer Server")
    parser.add_argument("--save_path", type=str, default='images')
    args = parser.parse_args()
    return args

def cal_acc(data_path):
    distribution = Counter()
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'judge_list' in data['output']:
                output_score = data.get("output", []).get('judge_list', '')
            else:
                output_score = data.get("output", []).get('corr_list', '')
            output_text = data.get("output", []).get('output', '')
            
            one_count = 0
            for text, score in zip(output_text, output_score):
                if text != '' and score == 1:
                    one_count += 1
            
            distribution[one_count] += 1
    print(distribution)
    print(f'acc: {100 * distribution[1] / sum(distribution.values())}%')
    print(distribution[1], sum(distribution.values()))

if __name__ == '__main__':
    args = parse_args()
    cal_acc(args.save_path)

