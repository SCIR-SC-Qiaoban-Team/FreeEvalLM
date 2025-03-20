

source ~/venv/cu124/bin/activate


port=54512
timeout=600000
reasoning_parser=deepseek_r1

device_ids=0,1


task=ifeval
sample=-1
decode_type=query_force_reasoning_content
query_keys=input
response_keys=response
reasoning_keys=reasoning



model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/DeepSeek-R1-Distill-Llama-8B
tensor_parallel_size=2
enable_reasoning=True
max_model_len=15504
max_new_tokens=1024
threads=128
model_num=1
reasoning_max_retry=3
cut_by_sentence=False

reasoning_scale_list[0]=0.1


SRC=YOUR_PATH_TO_FreeEvalLM


for reasoning_scale in ${reasoning_scale_list[@]}
do
    for model_path in ${model_path_list[@]}
    do
        file_path=$SRC/results/DeepSeek-R1-Distill-Llama-8B/ifeval
        save_path=$SRC/results/DeepSeek-R1-Distill-Llama-8B/ifeval/scale_$reasoning_scale$([ "$cut_by_sentence" == "True" ] && echo "_cut_sentence" || echo "")
        python -u $SRC/freeEvalLM/src/decode.py \
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



