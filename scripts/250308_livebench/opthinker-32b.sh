



source ~/gjh_ws/venv/cu124/bin/activate


port=54482
timeout=600000
reasoning_parser=openthinker

device_ids=4,5,6,7


task=livebench
sample=-1
decode_type=query
query_keys=input
response_keys=response
reasoning_keys=reasoning


sys_path=/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/system_prompt/openthinker.txt
model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/OpenThinker-32B
tensor_parallel_size=4
enable_reasoning=True
max_model_len=15504
max_new_tokens=8192
threads=64
model_num=1


for model_path in ${model_path_list[@]}
do
    save_path=/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/results/250308_livebench_new/$model_path
    python /share/home/wxzhao/gjh_ws/Code/FreeEvalLM/freeEvalLM/src/decode.py \
        --port $port \
        --timeout $timeout \
        --task $task \
        --sample $sample \
        --file_path None \
        --save_path $save_path \
        --decode_type $decode_type \
        --query_keys $query_keys \
        --system_prompt_file $sys_path \
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

