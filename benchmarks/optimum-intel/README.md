## How to do benchmark for optimum-intel models on MMLU dataset?
This repository contains code to benchmark optimum-intel models against tensorrt-llm, huggingface and ipex models on the MMLU dataset. MMLU is a massive multitask test consisting of multiple-choice questions from 57 various branches of knowledge such as elementary mathematics, US history, computer science, law, and medicine. For more details about MMLU, pls check out the original paper [here](https://arxiv.org/pdf/2009.03300).

### Installation 
1. Get the repositories
```bash
mkdir workspace && cd workspace
git clone <HuggingFace GitHub Web URL>
# only needed if you want to benchmark tensorrt-llm models as well  
git clone https://github.com/NVIDIA/TensorRT-LLM.git tensorrt-llm
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
export HF_CACHE_DIR=<your local huggingface cache directory>
bash run_docker.sh $HF_CACHE_DIR
# for tensorrt-llm models
bash run_docker.sh $HF_CACHE_DIR trt-llm 
```

### Run Benchmark 
```bash
# Download the data
mkdir data; wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
tar -xf data/mmlu.tar -C data && mv data/data data/mmlu
# Run the benchmark script
python3 mmlu.py --model_name meta-llama/Llama-2-7b-chat-hf --device xpu --max_input_length 2048 --max_new_tokens 1 --eval_mode optimum-intel --batch_size 1 --num_beams 1 --save_dir results
```
For more detailed usage, please use the `help` option:
```bash
python3 mmlu.py -h
```

To run the benchmark with multiple variable combinations, you can use the `run_benchmark.sh` script 
```bash 
bash run_benchmark.sh
``` 



