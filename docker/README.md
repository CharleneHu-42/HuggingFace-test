#These two dockerfiles are internal and external respectively. The shell file is for install mpi.

##1. Build dockerfile: 

for internal: docker build -f df_internal --build-arg imageVersion=2022_ww26 -t bert_qa:internal .

for external: docker build -f df_external -t bert_qa:external .

notes:

The imageVersion is the version of base image, it may be changed.

If the container based on external image is disconnected with Internet, please change the proxy to your proxy in the dockerfile.

##2. Start the docker container test: ./host_qa_test.sh image_id

notes:

The shell named host_qa_test.sh is a startup script 
image_id is the docker image id.

