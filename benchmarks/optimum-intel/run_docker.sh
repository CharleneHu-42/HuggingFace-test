#!/bin/bash

if [ "$1" == "trt-llm" ]; then
	EVAL_MODE=trt-llm
	if [ -d "$2" ]; then 
		HF_CACHE="$2" 
	else
		HF_CACHE=/root/.cache/huggingface
	fi
else 
	EVAL_MODE=opt
	if [ -d "$1" ]; then 
		HF_CACHE="$1" 
	else
		HF_CACHE=/root/.cache/huggingface
	fi 
fi 

SCRIPT=$(realpath "$0")
PROJECT_ROOT=$(dirname $(dirname $(dirname $SCRIPT)))

if [ $EVAL_MODE == "trt-llm" ]; then 
	tag=cuda
	docker run -it \
		-e http_proxy=${http_proxy} \
		-e https_proxy=${https_proxy} \
		-e no_proxy=${no_proxy} \
		-v $HF_CACHE:/root/.cache/huggingface \
		-w /workspace \
		--runtime=nvidia \
		--gpus all \
		--entrypoint /bin/bash \
		--name bench-${tag} \
		benchmark/optimum-intel:${tag}
else 
	tag=xpu
	docker run -it \
		--privileged  \
		-e http_proxy=${http_proxy} \
		-e https_proxy=${https_proxy} \
		-e no_proxy=${no_proxy} \
		-v $HF_CACHE:/root/.cache/huggingface \
		-w /workspace \
		--device=/dev/dri \
		--ipc=host \
		--entrypoint /bin/bash \
		--name bench-${tag} \
		benchmark/optimum-intel:${tag}
fi 
