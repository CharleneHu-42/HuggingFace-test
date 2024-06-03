#!/bin/bash

hf_cache="$1"
eval_mode="$2"

SCRIPT=$(realpath "$0")
PROJECT_ROOT=$(dirname $(dirname $(dirname $(dirname $SCRIPT))))

if [[ "$eval_mode" == "trt-llm" ]]; then 
	tag=trt-llm
else 
	tag=opt-intel
fi 

docker run -it \
	-e http_proxy=${http_proxy} \
	-e https_proxy=${https_proxy} \
	-e no_proxy=${no_proxy} \
	-v $hf_cache:/root/.cache \
	-v $PROJECT_ROOT:/mnt/code \
	-w /mnt/code/HuggingFace/benchmarks/optimum-intel \
	--runtime=nvidia \
	--gpus all \
	--entrypoint /bin/bash \
	--name $tag \
	benchmark/mmlu:$tag