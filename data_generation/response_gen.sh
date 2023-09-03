export MASTER_ADDR=localhost
export MASTER_PORT=7834
export CUDA_VISIBLE_DEVICES=0,1
MODEL_DIR=$1
OUT_DIR=$2
mkdir -p $OUT_DIR
OMP_NUM_THREADS=12
torchrun --nproc_per_node 2 --master_port 7834 response_gen.py \
                        --base_model $MODEL_DIR \
                        --data_path "openai_humaneval" \
                        --out_path $OUT_DIR \
                        --batch_size 4

python ./split_files.py $OUT_DIR $OUT_DIR
# bash ./scoring_responses.sh $OUT_DIR
# python make_data.py $OUT_DIR
