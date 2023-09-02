set -e
set -x

bash train.sh \
    ../ninjatech-ai/open_llama_3b_v2_code_alpaca/checkpoint-800 \
    ../ninjatech-ai/open_llama_3b_v2_code_alpaca/checkpoint-800-rrhf \
    ../ninjatech-ai/preds/open_llama_3b_v2_code_alpaca/checkpoint-800/T0.8_N4_rrhf_gen_train.jsonl
