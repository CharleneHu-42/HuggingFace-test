#!/bin/bash

# Default variable values
use_bf16=false
use_ipex_optimize=false 
use_jit=false 
use_torch_compile=false
task_name=""
model_id=""

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help      Display this help message"
 echo " -t, --task      Specify task name"
 echo " -m, --model     Specify model ID "
 echo " -i, --ipex      Use ipex optimize "
 echo " -m, --jit       Use iit "
 echo " -c, --compile   Use torch compile"
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
        use_bf16=true
        ;;
      -i | --ipex_optimize)
        use_ipex_optimize=true
        ;;
      -j | --jit)
        use_jit=true
        ;;
      -c | --compile)
        use_torch_compile=true
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

export OMP_NUM_THREADS=56

# Perform the desired actions based on the provided flags and arguments
if [[ "$use_bf16" = false ]]; then
 numactl -C 0-55 --membind 0 python $task_name/run_$task_name.py --model_id $model_id
 exit 0
fi 


if [[ "$use_ipex_optimize" = false ]] && [[ "$use_jit" = false ]] && [[ "$use_torch_compile" = false ]]; then
 numactl -C 0-55 --membind 0 python $task_name/run_$task_name.py --model_id $model_id --bf16
elif [[ "$use_ipex_optimize" = true ]] && [[ "$use_jit" = false ]] && [[ "$use_torch_compile" = false ]]; then 
 numactl -C 0-55 --membind 0 python $task_name/run_$task_name.py --model_id $model_id --bf16 --ipex_optimize
elif [[ "$use_ipex_optimize" = true ]] && [[ "$use_jit" = true ]] && [[ "$use_torch_compile" = false ]]; then 
 numactl -C 0-55 --membind 0 python $task_name/run_$task_name.py --model_id $model_id --bf16 --ipex_optimize --jit
elif [[ "$use_ipex_optimize" = false ]] && [[ "$use_jit" = false ]] && [[ "$use_torch_compile" = true ]]; then 
 numactl -C 0-55 --membind 0 python $task_name/run_$task_name.py --model_id $model_id --bf16 --torch_compile
else 
 echo "Invalid test case."
fi 
