#!/bin/bash

# Default variable values
device=xpu
library=transformers
mount_dir=${PWD}
name=hf-${device}-ut

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help            Display this help message"
 echo " -d, --device          Hardware Device[xpu, cuda]"
 echo " -l, --library         Target Name[optimum-quanto, transformers, accelerate, peft, diffusers, trl]"
 echo " -v, --mount_dir       Local directory to be mounted inside the container; default value is the current directory"
 echo " -n, --name            Container name"
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
      -d | --device)
        if ! has_argument $@; then
          echo "Device name not specified." >&2
          usage
          exit 1
        fi

        device=$(extract_argument $@)

        shift
        ;;
      -l | --library)
        if ! has_argument $@; then
          echo "Target library name not specified." >&2
          usage
          exit 1
        fi

        library=$(extract_argument $@)

        shift
        ;;
      -v | --mount-dir)
        mount_dir=$(extract_argument $@)
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

mkdir -p ${mount_dir}/${library}

if [[ -z "${HF_HOME}" ]]; then
  HF_HOME=$HOME/.cache/huggingface
fi

if [[ $device == "cuda" ]]; then
	docker run -it \
		--privileged \
		-e http_proxy=${http_proxy} \
		-e https_proxy=${https_proxy} \
		-e no_proxy=${no_proxy} \
		-e report=/mnt/${library}/ut.xlsx \
		-e log=/mnt/${library}/ut.log \
		-v ${HF_HOME}:/root/.cache/huggingface \
		-v ${mount_dir}:/mnt \
		-v /dev/shm:/dev/shm \
		-w /.tests/${library} \
		--runtime=nvidia \
		--gpus all \
		--name ${name} \
		appliedml/huggingface:${device}-ut
elif [[ $device == "xpu" ]]; then
	docker run -it \
		--privileged  \
		-e http_proxy=${http_proxy} \
		-e https_proxy=${https_proxy} \
		-e no_proxy=${no_proxy} \
		-e report=/mnt/${library}/ut.xlsx \
		-e log=/mnt/${library}/ut.log \
		-v /dev/dri/by-path:/dev/dri/by-path \
		-v ${HF_HOME}:/root/.cache/huggingface \
		-v ${mount_dir}:/mnt \
		-w /.tests/${library} \
		--device=/dev/dri \
		--ipc=host \
		--name ${name} \
		appliedml/huggingface:${device}-ut
else
	echo "the specified device is not supported"
fi
