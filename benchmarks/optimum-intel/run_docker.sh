#!/bin/bash
eval_mode="$1"

HF_CACHE=/root/.cache/huggingface
SCRIPT=$(realpath "$0")
PROJECT_ROOT=$(dirname $(dirname $(dirname $(dirname $SCRIPT))))

if [[ "$eval_mode" == "trt-llm" ]]; then 
	tag=trt-llm
	docker run -it \
		-e http_proxy=${http_proxy} \
		-e https_proxy=${https_proxy} \
		-e no_proxy=${no_proxy} \
		-v $HF_CACHE:/root/.cache/huggingface \
		-w /workspace \
		--runtime=nvidia \
		--gpus all \
		--entrypoint /bin/bash \
		--name $tag \
		benchmark/mmlu:$tag
else 
	tag=opt-intel
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
		--name $tag \
		benchmark/mmlu:$tag
fi 
