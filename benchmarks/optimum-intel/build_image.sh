docker build \
	-t mmlu/tensorrt_llm \
	--build-arg http_proxy=${http_proxy} \
    	--build-arg https_proxy=${https_proxy} \
    	--build-arg no_proxy=${no_proxy} \
	-f docker/Dockerfile.trt
	.
