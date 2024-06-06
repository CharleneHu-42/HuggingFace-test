## How to do benchmark for optimum-intel models on MMLU dataset?
This repository contains code to benchmark optimum-intel models against tensorrt-llm, huggingface and ipex models on the MMLU dataset. MMLU is a massive multitask test consisting of multiple-choice questions from 57 various branches of knowledge such as elementary mathematics, US history, computer science, law, and medicine. For more details about MMLU, pls check out the original paper [here](https://arxiv.org/pdf/2009.03300).

### Installation 
1. Clone the repository
```bash
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
# for tensorrt-llm 
bash run_docker.sh trt-llm 
```
You can also mount your existing huggingface cache directory inside the docker container to avoid re-downloading the model:
```bash
export HF_CACHE_DIR=<your local huggingface cache dir, e.g. /root/.cache/huggingface>
bash run-docker.sh $HF_CACHE_DIR
# for tensorrt-llm 
bash run_docker.sh trt-llm $HF_CACHE_DIR
``` 


### Run Benchmark 
```bash
cd benchmark
# optimum-intel 
bash run_benchmark.sh optimum-intel xpu
# tensorrt-llm
bash run_benchmark.sh trt-llm cuda 
```
For more input options, pls checkout the script `run_benchmark.sh`. 

The benchmark script will save both accuracy and performance of the model to a csv file in the folder `/workspace/mmlu-benchmark-log`. The accuracy is measured by the exact-match score of the predictions and labels. The performance is measured by average latency. To compute the average TTFT(Time To First Token) and TPOT(Time Per Output Token), you will need to specify 2 different max_new_tokens with one of them to be 1. For example, you pass max_new_tokens=1 and max_new_tokens=20, then 
$$avg\_ttft = avg\_latency_{max\_new\_tokens=1}$$
$$avg\_tpot = (avg\_latency_{max\_new\_tokens=20}-avg\_latency_{max\_new\_tokens=1})/(20-1)$$
