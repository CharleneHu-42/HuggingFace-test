docker run -it \
	-e http_proxy=${http_proxy} \
    	-e https_proxy=${https_proxy} \
    	-e no_proxy=${no_proxy} \
	-v /mnt/disk4/fanlilin/HF_CACHE:/root/.cache \
	-v ${PWD}:/mnt/code \
	-w /mnt/code \
	--runtime=nvidia \
	--gpus all \
	--entrypoint /bin/bash \
	--name trt_llm \
	mmlu/tensorrt_llm:latest
