This repository contains code to benchmark optimum-intel models against tensorrt-llm, huggingface, ipex, tgi and vllm models. There are 2 benchmark tasks available: mmlu and simple_bench.  
- MMLU is a public benchmark for LLM and consists of multiple-choice questions from 57 various branches of knowledge such as elementary mathematics, US history, computer science, law, and medicine. Accuracy is calculated from the model predicted answers and the groudtruth answers. For more details about MMLU, pls check out the original paper [here](https://arxiv.org/pdf/2009.03300).
- Simple_bench is a local benchmark that uses local prompt dataset as input data and measures the model generation latency to compare the various backends. No accuracy is calculated.

You can speficy the different task by using the `task_name` flag of the `main.py`, e.g. 

```bash
python main.py --task_name simple_bench --backend hf --model_name meta-llama/Llama-2-7b-chat-hf
```

## Env Set-Up 
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
By default, it will mount your local directory `/root/.cache/huggingface` inside the docker container to avoid re-downloading the model. 
But you can also specify your own huggingface cache directory as follows:
```bash
export HF_CACHE_DIR=<your local huggingface cache dir that has the folder hub inside>
bash run-docker.sh $HF_CACHE_DIR
# for tensorrt-llm 
bash run_docker.sh trt-llm $HF_CACHE_DIR
``` 

4. [Optional] Start TGI Server 
If you want to benchmark tgi, you need to start a tgi service. First, open another terminal and build the TGI Docker image for XPU
```bash
git clone https://github.com/huggingface/text-generation-inference.git && cd text-generation-inference
docker build \
	-f Dockerfile_intel . \
	-t tgi/intel \
	--build-arg http_proxy=${http_proxy} \
	--build-arg https_proxy=${https_proxy} \
	--build-arg no_proxy=${no_proxy}
```

Then start the TGI server
```bash
model=meta-llama/Llama-2-7b-hf
volume=/workspace1/huggingface/hub
HF_TOKEN=<your huggingface token>

docker run \
    --privileged  \
    -p 8080:80 \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    -e no_proxy=${no_proxy} \
    -e HF_TOKEN=${HF_TOKEN} \
    -v $volume:/data \
    --device=/dev/dri \
    --ipc=host \
    --name tgi-ipex \
    tgi/intel:latest \
    --model-id $model \
    --sharded false \
    --dtype float16 \
    --max-input-tokens 1024 \
    --max-total-tokens 2048 \
    --cuda-graphs 0
```

Now your tgi server is up. You can test it by running the following code:
```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":100}}' \
-H 'Content-Type: application/json'
```

## Run Benchmark 
```bash
cd benchmark
# Log in to huggingface-cli if you need to download the llama model
# You can get your token from huggingface.co/settings/token
huggingface-cli login --token <your huggingface token>
# optimum-intel 
bash run_benchmark.sh mmlu optimum-intel xpu
# tensorrt-llm
bash run_benchmark.sh mmlu trt-llm cuda 
```
The benchmark script will save the benchmark results to a csv file in the folder `/workspace/mmlu-benchmark-log`. For MMLU task, the accuracy is measured by the exact-match score of the predictions and labels. The performance is measured by average latency. To compute the average TTFT(Time To First Token) and TPOT(Time Per Output Token), you will need to specify 2 different max_new_tokens with one of them to be 1. For example, you pass max_new_tokens=1 and max_new_tokens=20, then 
$$avg\_ttft = avg\_latency_{max\_new\_tokens=1}$$
$$avg\_tpot = (avg\_latency_{max\_new\_tokens=20}-avg\_latency_{max\_new\_tokens=1})/(20-1)$$
By default, the `run_benchmark.sh` script will run max_new_tokens=1 and max_new_tokens=20. For more input options, pls checkout the script `run_benchmark.sh`. 

Please note that for ipex mode you have to first downgrade the transformers version with `pip install transformers==4.31.0` and than run `bash run_benchmark.sh ipex xpu`.