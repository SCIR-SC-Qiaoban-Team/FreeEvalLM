

get_random_port() {
    echo $((RANDOM % 10001 + 50000))
}

is_port_available() {
    local port=$1
    ! ss -tuln | awk '{print $4}' | grep -q ":$port$"
}

find_available_port() {
    while true; do
        port=$(get_random_port)
        if is_port_available "$port"; then
            echo "$port"
            return
        fi
    done
}




source ~/gjh_ws/venv/cu124/bin/activate


port=54512
timeout=600000
reasoning_parser=deepseek_r1

device_ids=2,3


task=livebench
sample=-1
decode_type=query_force_reasoning_content
query_keys=input
response_keys=response
reasoning_keys=reasoning



model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/QwQ
tensor_parallel_size=2
enable_reasoning=True
max_model_len=15504
max_new_tokens=8192
threads=128
model_num=1
reasoning_max_retry=3
cut_by_sentence=False

reasoning_scale_list[0]=0.1
reasoning_scale_list[1]=0.2
reasoning_scale_list[2]=0.5
reasoning_scale_list[3]=0.6
reasoning_scale_list[4]=0.8
reasoning_scale_list[5]=0.9
reasoning_scale_list[6]=0.0

for reasoning_scale in ${reasoning_scale_list[@]}
do
    for model_path in ${model_path_list[@]}
    do
        port=$(find_available_port)
        echo $port
        file_path=/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/results/250311_livebench/share/home/wxzhao/gjh_ws/Downloads/LLMs/QwQ
        save_path=/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/results/250312_livebench_less/$model_path/scale_$reasoning_scale$([ "$cut_by_sentence" == "True" ] && echo "_cut_sentence" || echo "")/$(basename "$model_path").json
        python -u /share/home/wxzhao/gjh_ws/Code/FreeEvalLM/freeEvalLM/src/decode.py \
            --port $port \
            --timeout $timeout \
            --task $task \
            --sample $sample \
            --file_path $file_path \
            --save_path $save_path \
            --decode_type $decode_type \
            --query_keys $query_keys \
            --response_keys $response_keys \
            --force_reasoning_content_keys $reasoning_keys \
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
            --reasoning_max_retry $reasoning_max_retry \
            --reasoning_scale $reasoning_scale \
            --cut_by_sentence $cut_by_sentence \
            --overwrite True
        echo "finished $model_path"
    done
done



