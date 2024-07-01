#!/bin/bash

tag="$1"
HF_CACHE="$2"

if [ -d "$2" ]; then 
	HF_CACHE="$2" 
else
	HF_CACHE=/root/.cache/huggingface
fi

SCRIPT=$(realpath "$0")
PROJECT_ROOT=$(dirname $(dirname $(dirname $SCRIPT)))

if [ $tag == "nvidia" ]; then 
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
		huggingface/benchmarks:${tag}
else 
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
		appliedmlwf/benchmark:xpu-ww25
fi 
