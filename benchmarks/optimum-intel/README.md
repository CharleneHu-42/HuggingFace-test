## How to do benchmark for optimum-intel models on MMLU dataset?
This repository contains code to benchmark optimum-intel models against tensorrt-llm, huggingface and ipex models on the MMLU dataset. MMLU is a massive multitask test consisting of multiple-choice questions from 57 various branches of knowledge such as elementary mathematics, US history, computer science, law, and medicine. For more details about MMLU, pls check out the original paper [here](https://arxiv.org/pdf/2009.03300).

### Installation 
1. Build docker image
```bash
bash build_image.sh
# for tensorrt-llm models, please explicitly pass the `trt-llm` flag
bash build_image.sh trt-llm
```
2. Run docker container 
```bash 
bash run_docker.sh
# for tensorrt-llm models
bash run_docker.sh trt-llm
```

### Run Benchmark 
```bash 
bash run_benchmark.sh --eval_mode optimum-intel --batch_size 1 --models llama2 --precision
```
For more detailed usage, please use `help` option
```bash
bash run_benchmark.sh -h
```



