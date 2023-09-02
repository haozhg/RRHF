#### The code is modified from trlX
import json
import math
import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from human_eval.data import read_problems, write_jsonl, stream_jsonl
from execution_rrhf import check_correctness
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
from collections import defaultdict, Counter
from typing import List
import re

problems = read_problems()


def reward_fn(
    samples: List[Dict],
    n_workers: int = os.cpu_count(),
    timeout: float = 3.0,
) -> List[Dict]:
    """
    input: 
    {
        "task_id": str,
        "prompt_w_instruction": str,
        "completion": str (cleaned),
    }
    
    output:
    {
        "task_id"
        "query": str (prompt_with_instruction),
        "responses": List[str],
        "scores": List[float],
    }
    """

    all_args = []
    completion_id = Counter()
    results = defaultdict(list)

    print("Reading samples...")
    for sample in tqdm(samples):
        task_id = sample["task_id"]
        completion = sample["completion"]
        args = (problems[task_id], completion, timeout, completion_id[task_id])
        completion_id[task_id] += 1
        all_args.append(args)

    assert len(completion_id) == len(problems), "Some problems are not attempted."

    results_by_task_id = defaultdict(list)
    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(check_correctness, *zip(*all_args)))
    
    assert len(samples) == len(results)
    
    for sample, result in zip(samples, results):
        """
        {
            "task_id": str,
            "passed": bool,
            "result": str,
            "completion_id": int
        }
        """
        results_by_task_id[result["task_id"]].append(
            {
                "task_id": result["task_id"],
                "query": sample["prompt_w_instruction"],
                "response": sample["completion"],
                "score": result["n_passed"],   
            }
        )
    
    # now merge by task_id
    final_results = []
    for task_id, ls in results_by_task_id.items():
        final_results.append(
            {
                "task_id": ls[0]["task_id"],
                "query": ls[0]["prompt_w_instruction"],
                "responses": [i["response"] for i in ls],
                "scores": [i["score"] for i in ls],
            }
        )

    return final_results


import sys

with open(sys.argv[1], 'r') as f:
    candidates = [json.loads(item.strip()) for item in f.readlines()]

finals = reward_fn(candidates)


with open(sys.argv[2], 'w') as f:
    for res in finals:
        f.write(json.dumps(res).strip() + '\n')