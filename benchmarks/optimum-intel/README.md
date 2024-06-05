## How to do benchmark for optimum-intel models on MMLU dataset?
This repository contains code to benchmark optimum-intel models against tensorrt-llm, huggingface and ipex models on the MMLU dataset. MMLU is a massive multitask test consisting of multiple-choice questions from 57 various branches of knowledge such as elementary mathematics, US history, computer science, law, and medicine. For more details about MMLU, pls check out the original paper [here](https://arxiv.org/pdf/2009.03300).

### Installation 
1. Get the repositories
```bash
mkdir workspace && cd workspace
git clone https://github.com/intel-sandbox/HuggingFace.git
cd HuggingFace/benchmarks/optimum-intel
```

2. Build docker image
```bash
bash build_image.sh
# for tensorrt-llm models, please explicitly pass the `trt-llm` flag
bash build_image.sh trt-llm
```
3. Run docker container 
```bash 
bash run_docker.sh
# for tensorrt-llm models
bash run_docker.sh trt-llm 
```

### Run Benchmark 
```bash
# Run the benchmark script
python3 HuggingFace/benchmarks/optimum-intel/mmlu.py --model_name meta-llama/Llama-2-7b-chat-hf --device xpu --max_input_length 2048 --max_new_tokens 1 --eval_mode optimum-intel --batch_size 1 --num_beams 1 --save_dir .
```
The benchmark script will save both accuracy and performance of the model to a csv file in the current directory. The accuracy is measured by the exact-match score of the predictions and labels. The performance is measured by average output latency. To compute the average TTFT(Time To First Token) and TPOT(Time Per Output Token), you will need to run the above benchmark twice by specifying 2 different max_new_tokens with one of them to be 1. For example, you pass max_new_tokens=1 and max_new_tokens=20, then 
$$avg\_ttft = avg\_latency_{max\_new\_tokens=1}$$
$$avg\_tpot = (avg\_latency_{max\_new\_tokens=20}-avg\_latency_{max\_new\_tokens=1})/(20-1)$$


For more detailed usage, please use the `help` option:
```bash
python3 mmlu.py -h
```

To run the benchmark with multiple variable combinations, you can use the `run_benchmark.sh` script 
```bash 
bash run_benchmark.sh
``` 
It will output a csv file in the folder `/workspace/mmlu-benchmark-log`. 
