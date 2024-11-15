#!/bin/bash

# Default variable values
device="xpu"
target="transformers"
local_dir=${PWD}
name="hf-ut"

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help            Display this help message"
 echo " -d, --device          Hardware Device[xpu, cuda]"
 echo " -t, --target          Target Name[transformers, peft, accelerate, diffusers, trl]"
 echo " -l, --local_dir       The local directory to be mounted inside the container"
 echo " -n, --name            The container name"
}

has_argument() {
    [[ ("$1" == *=* && -n ${1#*=}) || ( ! -z "$2" && "$2" != -*)  ]];
}

extract_argument() {
  echo "${2:-${1#*=}}"
}

# Function to handle options and arguments
handle_options() {
  while [ $# -gt 0 ]; do
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;
      -d | --device*)
        if ! has_argument $@; then
          echo "Device name not specified." >&2
          usage
          exit 1
        fi

        device=$(extract_argument $@)

        shift
        ;;
      -t | --target*)
        if ! has_argument $@; then
          echo "Target library name not specified." >&2
          usage
          exit 1
        fi

        target=$(extract_argument $@)

        shift
        ;;
      -l | --local_dir)
        local_dir=$(extract_argument $@)
        shift
        ;;
      -n | --name)
        name=$(extract_argument $@)
        shift
        ;;
      *)
        echo "Invalid option: $1" >&2
        usage
        exit 1
        ;;
    esac
    shift
  done
}

# Main script execution
handle_options "$@"

mkdir -p $local_dir/${target}

if [[ $device == "cuda" ]]; then
	docker run -it \
		--privileged \
		-e http_proxy=${http_proxy} \
		-e https_proxy=${https_proxy} \
		-e no_proxy=${no_proxy} \
    -e report=/mnt/${target}/ut.xlsx \
    -e log=/mnt/${target}/ut.log \
		-v ${HF_HOME}:/root/.cache/huggingface \
		-v ${local_dir}:/mnt \
    -v /dev/shm:/dev/shm \
		-w /tests/${target} \
		--runtime=nvidia \
		--gpus all \
		--entrypoint /bin/bash \
		--name ${name} \
		huggingface-ut/${device}
elif [[ $device == "xpu" ]]; then
	docker run -it \
		--privileged  \
		-e http_proxy=${http_proxy} \
		-e https_proxy=${https_proxy} \
		-e no_proxy=${no_proxy} \
    -e report=/mnt/${target}/ut.xlsx \
    -e log=/mnt/${target}/ut.log \
    -e OCL_ICD_VENDORS=/etc/OpenCL/vendors \
    -v /dev/dri/by-path:/dev/dri/by-path \
		-v ${HF_HOME}:/root/.cache/huggingface \
		-v ${local_dir}:/mnt \
		-w /tests/${target} \
		--device=/dev/dri \
		--ipc=host \
		--entrypoint /bin/bash \
		--name ${name} \
		huggingface-ut/${device}

else
	echo "the given device is not supported."
fi
