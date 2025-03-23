



source ~/venv/cu124/bin/activate


port=54182
timeout=600000
reasoning_parser=deepseek_r1
# support list:
#     deepseek_r1
#     simplescaling
#     openthinker


device_ids=0


task=ifeval
sample=-1
decode_type=query
query_keys=input
response_keys=response
reasoning_keys=reasoning


model_path_list[0]=YOUR_PATH/DeepSeek-R1-Distill-Llama-8B
tensor_parallel_size=1
enable_reasoning=True
max_model_len=15504
max_new_tokens=8192
threads=128
model_num=1
reasoning_max_retry=3

SRC=YOUR_PATH_TO_FreeEvalLM

for model_path in ${model_path_list[@]}
do
    save_path=$SRC/results/DeepSeek-R1-Distill-Llama-8B/ifeval
    python -u $SRC/freeEvalLM/src/decode.py \
        --port $port \
        --timeout $timeout \
        --task $task \
        --sample $sample \
        --file_path None \
        --save_path $save_path \
        --decode_type $decode_type \
        --query_keys $query_keys \
        --response_keys $response_keys \
        --reasoning_keys $reasoning_keys \
        --model_path $model_path \
        --tensor_parallel_size $tensor_parallel_size \
        --max_model_len $max_model_len \
        --enable_reasoning $enable_reasoning \
        --reasoning_parser $reasoning_parser \
        --model_num $model_num \
        --threads $threads \
        --device_ids $device_ids \
        --max_new_tokens $max_new_tokens \
        --show_log True \
        --add_reasoning_prompt True \
        --reasoning_max_retry $reasoning_max_retry
    echo "finished $model_path"
done
model_path_list=()

