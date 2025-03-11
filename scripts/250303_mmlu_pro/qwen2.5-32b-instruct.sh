



source ~/gjh_ws/venv/cu124/bin/activate


port=50014
timeout=600000
reasoning_parser=deepseek_r1

device_ids=4,5,6,7


task=mmlu_pro_0shots
sample=-1
decode_type=query
query_keys=input
response_keys=response
reasoning_keys=reasoning



model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/Qwen2.5-32B-Instruct/
tensor_parallel_size=2
enable_reasoning=False
max_model_len=15504
max_new_tokens=8192
threads=128
model_num=1


for model_path in ${model_path_list[@]}
do
    save_path=/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/results/250303_mmlu_pro/$model_path
    python -u /share/home/wxzhao/gjh_ws/Code/FreeEvalLM/freeEvalLM/src/decode.py \
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
        --show_log False
    echo "finished $model_path"
done
model_path_list=()

