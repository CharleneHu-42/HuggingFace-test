# Run Guide

## 1. Build dockerfile: 

These two dockerfiles are internal and external respectively.
```bash
for internal: docker build -f df_internal --build-arg imageVersion=2022_ww26 -t bert_qa:internal .

for external: docker build -f df_external --build-arg http_proxy=http://proxy-chain.intel.com:911 --build-arg https_proxy=http://proxy-chain.intel.com:911 --build-arg no_proxy=*.intel.com -t bert_qa:external .
```
notes:

The imageVersion is the version of base image, it may be changed.

If the container based on external image is disconnected with Internet, please change the proxy to your proxy in the dockerfile.

## 2. Start the docker container test:

notes:
```bash
help: ./host_qa_test.sh -h

for external: ./host_qa_test.sh -c 0 -i bert_qa:external #2DDP in 1 container(node1)
              ./host_qa_test.sh -c 1 -i bert_qa:external #4DPP in 2 containers single node(node1 and node2)

for internal: ./host_qa_test.sh -c 0 -i bert_qa:internal #2DDP in 1 container(node1)
              ./host_qa_test.sh -c 1 -i bert_qa:internal #4DPP in 2 containers single node(node1 and node2)
```
the output is /tmp/debug_squad, you could change it in host_qa_test.sh

The shell named host_qa_test.sh is a startup script

please stop and rm existing container named node1 and node2 before start the test

```bash
docker stop node1
docker rm node1
docker stop node2
docker rm node2
```
you could remove hg-bridge by "docker network rm hg-bridge"

