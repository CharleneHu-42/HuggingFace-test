#!/bin/bash 

tag="$1" 
dockerfile=Dockerfile.${tag}

cp ../../tests/workloads/datasets/prompt.json .

docker build \
	-f ${dockerfile} . \
	-t huggingface/benchmarks:${tag} \
	--build-arg http_proxy=${http_proxy} \
	--build-arg https_proxy=${https_proxy} \
	--build-arg no_proxy=${no_proxy}

rm prompt.json