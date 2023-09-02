set -e
set -x

time python data_generation/scoring_responses.py \
    ../ninjatech-ai/preds/open_llama_3b_v2_code_alpaca/checkpoint-800/T0.8_N4_rrhf_gen.jsonl \
    ../ninjatech-ai/preds/open_llama_3b_v2_code_alpaca/checkpoint-800/T0.8_N4_rrhf_gen_train.jsonl \
    2>&1 | tee -a scoring_responses.py.log
