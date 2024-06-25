This repository contains a simple benchmark script to compare the performance of TGI, VLLM, Optimum-Intel and HuggingFace transformers. 

## How to run benchmark for TGI?
1. Build the TGI Docker image for XPU
```bash
git clone https://github.com/huggingface/text-generation-inference.git && cd text-generation-inference
docker build \
	-f Dockerfile_intel . \
	-t tgi/intel \
	--build-arg http_proxy=${http_proxy} \
	--build-arg https_proxy=${https_proxy} \
	--build-arg no_proxy=${no_proxy}
```

2. Get yout huggingface token 
- Go to https://huggingface.co/settings/tokens
- Copy your cli READ token
- Export HF_TOKEN=<your cli READ token>


3. Run TGI server
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
    --cuda-graphs 0 \
    --trust-remote-code
```

4. Run benchmark
Once your tgi server is up, you can run the benchmark script to get the token latency
```
python benchmark.py --backend tgi --input_tokens 32 --output_tokens 1 --batch_size 1 --tgi_endpoint "http://127.0.0.1:8080" 
```


## How to run benchmark for others?
To run benchamrk for optimum-intel/vllm/hf, you don't need to start an inference endpoint. You just need to run like blew:
```bash
python benchmark.py --backend optimum-intel --input_tokens 32 --output_tokens 1 --batch_size 1
```
You can run `python benchmark.py --help` to checkout more script options. 