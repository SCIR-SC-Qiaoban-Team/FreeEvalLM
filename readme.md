<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities </h1>

<!-- Authors -->

<p align="center">
    <!-- Use &nbsp; for spacing, <sup> for affiliations, and style="text-decoration: none;" for link decoration -->
    <a href="https://circle-hit.github.io/" target="_blank" style="text-decoration: none;">Weixiang Zhao<sup>1</sup><sup>*</sup></a>&nbsp;,&nbsp;
    <a href="https://github.com/XingYuSSS" target="_blank" style="text-decoration: none;">Xingyu Sui<sup>1</sup><sup>*</sup></a>&nbsp;,&nbsp;
    <a href="https://github.com/MuyuenLP" target="_blank" style="text-decoration: none;">Jiahe Guo<sup>1</sup><sup>*</sup></a>&nbsp;,&nbsp;
    <a href="https://github.com/yulinlp" target="_blank" style="text-decoration: none;">Yulin Hu<sup>1</sup><sup>*</sup></a>&nbsp;,&nbsp;
    <a href="https://dengyang17.github.io/" target="_blank" style="text-decoration: none;">Yang Deng<sup>2</sup></a>&nbsp;,&nbsp;
    <a href="https://homepage.hit.edu.cn/yanyan" target="_blank" style="text-decoration: none;">Yanyan Zhao<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://homepage.hit.edu.cn/qinbing" target="_blank" style="text-decoration: none;">Bing Qin<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://chewanxiang.com/" target="_blank" style="text-decoration: none;">Wanxiang Che<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://www.chuatatseng.com/" target="_blank" style="text-decoration: none;">Tat-Seng Chua<sup>3</sup></a>&nbsp;,&nbsp;
    <a href="https://homepage.hit.edu.cn/liuting" target="_blank" style="text-decoration: none;">Ting Liu<sup>1</sup></a>&nbsp;,&nbsp;
    <br/><br/>
    <sup>1</sup>Harbin Institute of Technology&nbsp;&nbsp;&nbsp;
    <sup>2</sup>Singapore Management University&nbsp;&nbsp;&nbsp;
    <sup>3</sup>National University of Singapore&nbsp;&nbsp;&nbsp;
</p>

<!-- Warning -->

<p align="center" style="color: red;">
    <b><em>Warning: This paper contains model outputs that may be considered offensive.</em></b>
</p>



<!-- Links (Paper, GitHub, Dataset) -->

<p align="center" style="font-size: 1.2em;">
    <b>
        <a href="https://arxiv.org" target="_blank" style="text-decoration: none;">[Paper]</a>
    </b>
    &nbsp;&nbsp;
    <b>
        <a href="https://github.com/SCIR-SC-Qiaoban-Team/FreeEvalLM" target="_blank" style="text-decoration: none;">[Project Page]</a>
    </b>
    &nbsp;&nbsp;

</p>


## News

- [2025/03/20] We released our code source.


## Install and Run


#### Installation

You need to insatll `easyvllm` for the deocoding of LRMs.

```shell
git clone https://github.com/SCIR-SC-Qiaoban-Team/easyvllm
cd easyvllm
pip install -e .
```
`easyvllm` depends on `VLLM` and requires additional installation steps to support LRMs. Please refer to the installation guide at `https://github.com/SCIR-SC-Qiaoban-Team/easyvllm`.


```shell
git clone https://github.com/SCIR-SC-Qiaoban-Team/FreeEvalLM
cd FreeEvalLM
pip install -e .
```

Additionally, you need to add your OpenAI token for the evaluation of certain benchmarks.

#### Run our code

```shell
cd FreeEvalLM
python freeEvalLM/src/decode.py
```

Our pipeline consists of two steps: `generate` and `evaluate`. If you need to use them separately, please call the corresponding evaluator in the `tasks` module individually.


##  Arguments

| Parameter                  | Type                                                       | Default       | Description |
|----------------------------|------------------------------------------------------------|---------------|-------------|
| `model_path`                | -                                                          | -             | Path to the model |
| `save_path`                 | -                                                          | -             | Path to save results |
| `decode_type`               | `Literal['query', 'query_reasoning_ctrl', 'query_force_reasoning_content']` | - | Type of decoding |
| `file_path`                 | `str`                                                      | `None`        | Path to the input file |
| `task`                      | `str`                                                      | `None`        | Task name, when `file_path` is set to `None`, the data and evaluator will be loaded according to the predefined task type, which support `livebench`, `ifeval`, `mmlu_pro`, `strong_reject`, `wild_jailbreak` and `XSTest_S`  |
| `sample`                    | `int`                                                      | `-1`          | Number of sampled data. When set to -1, all data will be loaded|
| `query_keys`                | `str`                                                      | `None`        | Keys for query extraction |
| `response_keys`             | `str`                                                      | `None`        | Keys for response storage |
| `reasoning_keys`            | `str`                                                      | `None`        | Keys for reasoning extraction and storage|
| `tensor_parallel_size`       | `int`                                                      | `1`           | Number of tensor parallel units in `vllm` |
| `model_num`                 | `int`                                                      | `None`        | Number of models to be loaded in `vllm` |
| `port`                      | `int`                                                      | `50000`       | Port number for `vllm serve`. |
| `max_model_len`             | `int`                                                      | `None`        | Maximum model length in `vllm` |
| `show_log`                  | `bool`                                                     | `True`        | Whether to display logs of `vllm` |
| `timeout`                   | `int`                                                      | `30`          | Timeout duration in seconds |
| `threads`                   | `int`                                                      | `20`          | Number of threads |
| `enable_reasoning`           | `bool`                                                     | `False`       | Enable reasoning when using LRMs |
| `reasoning_parser`          | `str`                                                      | `'deepseek_r1'` | Reasoning parser type, support `deepseek_r1`, `openthinker` and `simplescaling` |
| `system_prompt_file`         | `str`                                                      | `None`        | Path to the system prompt file |
| `chat_template_file`        | `str`                                                      | `None`        | Path to the chat template file |
| `max_new_tokens`            | `int`                                                      | `8192`        | Maximum number of new tokens |
| `device_ids`                | `str`                                                      | `None`        | Device IDs to use |
| `reasoning_max_retry`       | `int`                                                      | `10`          | Maximum number of retries when the model's output does not conform to the expected format  |
| `add_reasoning_prompt`      | `bool`                                                     | `False`       | Manually add the reasoning token.|
| `enable_length_ctrl`        | `bool`                                                     | `False`       | Enable length control |
| `reasoning_max_len`         | `int`                                                      | `None`        | Maximum length for reasoning |
| `reasoning_min_len`         | `int`                                                      | `0`           | Minimum length for reasoning |
| `reasoning_scale`           | `float`                                                    | `None`        | Scaling factor for reasoning |
| `cut_by_sentence`           | `bool`                                                     | `False`       | Cut content by sentence for length control|
| `force_reasoning_content_keys` | `str`                                                  | `None`        | Keys for forced reasoning content, usually consistent with the previously saved `reasoning_keys` |
| `overwrite`                 | `bool`                                                     | `False`       | Overwrite the responses and reasoning keys of the input files |


##  Tasks

We now support testing for `MMLU-Pro`, `IFEval`, `Live-Bench`, `StrongReject`, `WildJailbreak`, and `XSTest`.
If you need to add a custom task, please refer to `src/task.py` to add the dataset and evaluator.

##  Example

```shell
cd FreeEvalLM
bash scripts/distill-8b_ifeval.sh
```

If you want to control the thinking length and try multi-GPUs inference, please refer to `scripts/distill-8b_ifeval_length-control.sh`


## Acknowledgments

We would like to express sincere gratitude to the following open-source projects and their development teams. 

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)
- [LiveBench](https://github.com/LiveBench/LiveBench)
- [IFEval](https://github.com/Rohan2002/IFEval)
- [XSTest](https://github.com/paul-rottger/xstest)
- [wildteaming](https://github.com/allenai/wildteaming)
- [strong_reject](https://github.com/dsbowen/strong_reject)
- [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)



## Citation

If you find our work useful, please consider citing our paper:

```
bib

```