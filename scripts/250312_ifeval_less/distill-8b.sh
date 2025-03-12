



source ~/gjh_ws/venv/cu124/bin/activate


port=54561
timeout=600000
reasoning_parser=deepseek_r1

device_ids=0


task=ifeval
sample=4
decode_type=query_force_reasoning_content
query_keys=input
response_keys=response
reasoning_keys=reasoning


model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/DeepSeek-R1-Distill-Llama-8B
tensor_parallel_size=1
enable_reasoning=True
max_model_len=15504
max_new_tokens=8192
threads=128
model_num=1
reasoning_max_retry=3
cut_by_sentence=False
reasoning_scale_list[0]=0.9

for model_path in ${model_path_list[@]}
do
    file_path=/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/results/250311_ifeval/share/home/wxzhao/gjh_ws/Downloads/LLMs/DeepSeek-R1-Distill-Llama-8B/ifeval.json
    save_path=/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/results/250312_ifeval_less/$model_path/scale_$reasoning_scale$([ "$cut_by_sentence" == "True" ] && echo "_cut_sentence" || echo "")/$(basename "$model_path").json
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
model_path_list=()

