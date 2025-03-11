



source ~/gjh_ws/venv/cu124/bin/activate


port=50023
timeout=600000
# device_ids=6,7
# device_ids=6,7
device_ids=2,3


# save_name=strongreject_small
# file_path=/share/home/wxzhao/decode/my_decode/query/strongreject_small_dataset.csv
# decode_type=query_reasoning_ctrl
# query_keys=forbidden_prompt
# response_keys=response
# reasoning_keys=reasoning

# save_name=XSTest
# file_path=/share/home/wxzhao/decode/my_decode/query/xstest_prompts_safe.csv
# decode_type=query_reasoning_ctrl
# query_keys=prompt
# response_keys=response
# reasoning_keys=reasoning

save_name=wildjailbreak_harmful
file_path=/share/home/wxzhao/decode/my_decode/query/wildjailbreak_harmful.csv
decode_type=query_reasoning_ctrl
query_keys=adversarial
response_keys=response
reasoning_keys=reasoning

# # model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/DeepSeek-R1-Distill-Llama-8B
# model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/DeepSeek-R1-Distill-Llama-70B
# reasoning_parser=deepseek_r1
# tensor_parallel_size=2
# max_model_len=15504
# enable_reasoning=True
# threads=256
# model_num=1
# reasoning_max_retry=10
# reasoning_max_len=0
# for model_path in ${model_path_list[@]}
# do
#     save_path=/share/home/wxzhao/decode/my_decode/results/$save_name/zero_thinking/$(basename "$model_path").json
#     python /share/home/wxzhao/decode/my_decode/decode.py \
#         --port $port \
#         --timeout $timeout \
#         --file_path $file_path \
#         --save_path $save_path \
#         --decode_type $decode_type \
#         --query_keys $query_keys \
#         --response_keys $response_keys \
#         --reasoning_keys $reasoning_keys \
#         --model_path $model_path \
#         --tensor_parallel_size $tensor_parallel_size \
#         --max_model_len $max_model_len \
#         --enable_reasoning $enable_reasoning \
#         --reasoning_parser $reasoning_parser \
#         --model_num $model_num \
#         --threads $threads \
#         --device_ids $device_ids \
#         --reasoning_max_retry $reasoning_max_retry \
#         --add_reasoning_prompt True \
#         --enable_length_ctrl True \
#         --reasoning_max_len $reasoning_max_len \
#         # --show_log False
#     echo "finished $model_path"
# done
# model_path_list=()


# model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/OpenThinker-7B
# model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/OpenThinker-32B
# reasoning_parser=openthinker
# tensor_parallel_size=2
# max_model_len=15504
# enable_reasoning=True
# system_prompt_file=/share/home/wxzhao/decode/my_decode/system_prompt/openthinker.txt
# threads=256
# model_num=1
# reasoning_max_retry=10
# reasoning_max_len=0
# for model_path in ${model_path_list[@]}
# do
#     save_path=/share/home/wxzhao/decode/my_decode/results/$save_name/zero_thinking/$(basename "$model_path")_wsys.json
#     python /share/home/wxzhao/decode/my_decode/decode.py \
#         --port $port \
#         --timeout $timeout \
#         --file_path $file_path \
#         --save_path $save_path \
#         --decode_type $decode_type \
#         --query_keys $query_keys \
#         --response_keys $response_keys \
#         --reasoning_keys $reasoning_keys \
#         --model_path $model_path \
#         --tensor_parallel_size $tensor_parallel_size \
#         --max_model_len $max_model_len \
#         --enable_reasoning $enable_reasoning \
#         --reasoning_parser $reasoning_parser \
#         --model_num $model_num \
#         --system_prompt_file $system_prompt_file \
#         --threads $threads \
#         --device_ids $device_ids \
#         --reasoning_max_retry $reasoning_max_retry \
#         --add_reasoning_prompt True \
#         --enable_length_ctrl True \
#         --reasoning_max_len $reasoning_max_len \
#         # --show_log False
#     echo "finished $model_path"
# done
# model_path_list=()


model_path_list[0]=/share/home/wxzhao/gjh_ws/Downloads/LLMs/s1.1-32B
reasoning_parser=simplescaling
tensor_parallel_size=2
max_model_len=15504
enable_reasoning=True
threads=128
model_num=1
reasoning_max_retry=10
reasoning_max_len=0
for model_path in ${model_path_list[@]}
do
    save_path=/share/home/wxzhao/decode/my_decode/results/$save_name/zero_thinking/$(basename "$model_path").json
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
        --reasoning_max_retry $reasoning_max_retry \
        --add_reasoning_prompt True \
        --enable_length_ctrl True \
        --reasoning_max_len $reasoning_max_len \
        # --show_log False
    echo "finished $model_path"
done
model_path_list=()


