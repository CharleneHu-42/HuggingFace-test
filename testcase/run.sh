#!/bin/bash

# Default variable values
use_bf16=False
use_ipex_optimize=False
use_jit=False
use_torch_compile=False
task_name=""
model_id=""
torch_dtype="float32"
backend="ipex"

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help            Display this help message"
 echo " -t, --task            Specify task name"
 echo " -m, --model           Specify model ID "
 echo " -i, --ipex            Use ipex optimize "
 echo " -j, --jit             Use jit "
 echo " -c, --torch_compile   Use torch compile"
 echo " -b, --bf16            Use amp bf16"
 echo " --torch_dtype         Indicate the model dtype[float32, bfloat16]"
 echo " --backend             Indicate the torch compile backend[ipex, inductor]"
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
      -t | --task*)
        if ! has_argument $@; then
          echo "Script name not specified." >&2
          usage
          exit 1
        fi

        task_name=$(extract_argument $@)

        shift
        ;;
      -m | --model*)
        if ! has_argument $@; then
          echo "Model ID not specified." >&2
          usage
          exit 1
        fi

        model_id=$(extract_argument $@)

        shift
        ;;
      -b | --bf16)
	      use_bf16=$(extract_argument $@)
        shift
        ;;
      -i | --ipex_optimize)
        use_ipex_optimize=$(extract_argument $@)
        shift
        ;;
      -j | --jit)
        use_jit=$(extract_argument $@)
        shift
        ;;
      -c | --torch_compile)
        use_torch_compile=$(extract_argument $@)
        shift
        ;;
      --torch_dtype)
        torch_dtype=$(extract_argument $@)
        shift
        ;;
      --backend)
        backend=$(extract_argument $@)
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


# Setup environment variables for performance on Xeon
export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
export KMP_BLOCKTIME=INF
export KMP_TPAUSE=0
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_FORJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export TORCHINDUCTOR_FREEZING=1
export OMP_NUM_THREADS=56

# Perform the desired actions based on the provided flags and arguments
numactl -C 0-55 --membind 0 python $task_name/run_$task_name.py --model_id $model_id --bf16 $use_bf16 --jit $use_jit --ipex_optimize $use_ipex_optimize --torch_dtype $torch_dtype --torch_compile $use_torch_compile --backend $backend


