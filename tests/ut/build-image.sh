#!/bin/bash

# Default variable values
device="xpu"
target="transformers"

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help            Display this help message"
 echo " -d, --device          Hardware Device[xpu, cuda]"
 echo " -t, --target          Target Name[transformers, peft, accelerate, diffusers, trl]"
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
    esac
    shift
  done
}

# Main script execution
handle_options "$@"

cd $target

docker build \
	-f Dockerfile.${device} . \
	--build-arg http_proxy=${http_proxy} \
	--build-arg https_proxy=${https_proxy} \
	--build-arg no_proxy=${no_proxy} \
	-t huggingface-ut/${target}:${device}
