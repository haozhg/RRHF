set -e
set -x

# step 3, RRHF/scoring_responses.py
# input = single jsonl
# output = query + responses + scores
# first group by for each response
# check #test cases passed, evaluate_functional_correctness()
# ```
# {
#     "task_id"
#     "query": str (prompt_with_instruction),
#     "responses": List[str],
#     "scores": List[float],
# }
# ```

time python data_generation/scoring_responses.py \
    ../ninjatech-ai/preds/open_llama_3b_v2_code_alpaca/checkpoint-800/T0.8_N4_rrhf_gen.jsonl \
    ../ninjatech-ai/preds/open_llama_3b_v2_code_alpaca/checkpoint-800/T0.8_N4_rrhf_gen_train.jsonl \
    2>&1 | tee -a scoring_responses.py.log
