#! /bin/bash 

dockerfile=Dockerfile

if [[ $1 = "trt-llm" ]]; then 
	dockerfile=Dockerfile.trt
	tag=trt-llm
else
	dockerfile=Dockerfile.opt
	tag=opt-intel
fi

docker build \
	-f ${dockerfile} . \
	-t benchmark/mmlu:${tag} \
	--build-arg http_proxy=${http_proxy} \
	--build-arg https_proxy=${https_proxy} \
	--build-arg no_proxy=${no_proxy}