# Run Guide

## 1. Build dockerfile: 

These two dockerfiles are internal and external respectively.
```bash
for internal: docker build -f df_internal --build-arg imageVersion=2022_ww26 -t bert_qa:internal .

for external: docker build -f df_external -t bert_qa:external .
```
notes:

The imageVersion is the version of base image, it may be changed.

If the container based on external image is disconnected with Internet, please change the proxy to your proxy in the dockerfile.

## 2. Start the docker container test:

notes:
```bash
help: ./host_qa_test.sh -h

for external: ./host_qa_test.sh -i bert_qa:external

for internal: ./host_qa_test.sh -i bert_qa:internal
```
the output is /tmp/debug_squad, you could change it in host_qa_test.sh

The shell named host_qa_test.sh is a startup script

