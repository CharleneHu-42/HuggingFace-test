These two dockerfiles are internal and external respectively. The shell file is for install mpi.

1. Build dockerfile: 

for internal: docker build -f df_internal --build-arg imageVersion=2022_ww26 .

for external: docker build -f df_external .

notes:

The imageVersion is the version of base image, it may be changed.

If the container based on external image is disconnected with Internet, please change the proxy to your proxy in the dockerfile.

2. Start the image: docker run -v /localdisk/fengjiqing/:/usr/tmp/ --privileged --shm-size 800g image_id /bin/bash

notes:

The shell named start_bert.sh is a startup script. /localdisk/fengjiqing/ is a local path which save the output files of this task and /usr/tmp/ is the container's task output path which is shown in start_bert.sh (output_dir).

