



source ~/gjh_ws/venv/cu124/bin/activate


port=50000
timeout=600000
reasoning_parser=deepseek_r1

device_ids=2,3


save_name=wildjailbreak_harmful
file_path=/share/home/wxzhao/decode/my_decode/query/wildjailbreak_harmful.csv
decode_type=query
query_keys=adversarial
response_keys=response
reasoning_keys=reasoning



model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/Llama-3.3-70B-Instruct
tensor_parallel_size=2
enable_reasoning=False
max_model_len=4096
max_new_tokens=2048
threads=128
model_num=1
for model_path in ${model_path_list[@]}
do
    save_path=/share/home/wxzhao/decode/my_decode/results/$save_name/$(basename "$model_path").json
    python /share/home/wxzhao/decode/my_decode/decode.py \
        --port $port \
        --timeout $timeout \
        --file_path $file_path \
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
        # --show_log False
    echo "finished $model_path"
done
model_path_list=()


model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/DeepSeek-R1-Distill-Llama-70B
reasoning_parser=deepseek_r1
tensor_parallel_size=2
max_model_len=15504
enable_reasoning=True
threads=128
model_num=1
for model_path in ${model_path_list[@]}
do
    save_path=/share/home/wxzhao/decode/my_decode/results/$save_name/$(basename "$model_path").json
    python /share/home/wxzhao/decode/my_decode/decode.py \
        --port $port \
        --timeout $timeout \
        --file_path $file_path \
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
        # --show_log False
    echo "finished $model_path"
done
model_path_list=()
