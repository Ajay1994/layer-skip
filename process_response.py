import os
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
from typing import List, Optional
import pandas as pd
import numpy as np

import json
def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list
    
questions = get_json_list("table/question.jsonl")
categories = [questions[i]["category"] for i in range(0, len(questions))]
response_pruned = get_json_list("commit_logs/judge_result000000_1111111111.jsonl")

score_unpruned = []
score_pruned = [] 
for item in response_pruned:
    score = item["score"]
    score_unpruned.append(score[0])
    score_pruned.append(score[1])

result = pd.DataFrame()
result["category"] = categories
result["pruned"] = score_pruned
result["unpruned"] = score_unpruned
unique_cat = np.unique(categories)

mask = result['pruned'] == -1.0
result = result[~mask]

for cat in unique_cat:
    cat_res = result.loc[result['category'] == cat]
    # print(cat_res)
    print(f"{cat}:\t\t\tPruned:{np.mean(cat_res['pruned']):.3f}\t\t\tUnpruned:{np.mean(cat_res['unpruned']):.3f}\t\t\tDifference: {np.mean(cat_res['pruned']) - np.mean(cat_res['unpruned'])}")