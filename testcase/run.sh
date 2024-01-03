#!/bin/bash

# Default variable values
use_ipex_optimize=False
use_jit=False
use_torch_compile=False
task_name=""
model_id=""
model_dtype="float32"
compute_dtype="float32"
backend="inductor"
device="cpu"

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help            Display this help message"
 echo " -t, --task            Specify task name"
 echo " -m, --model_id           Specify model ID "
 echo " -i, --ipex            Use ipex optimize "
 echo " -j, --jit             Use jit "
 echo " -c, --torch_compile   Use torch compile"
 echo " --model_dtype         Indicate the model dtype[float32, bfloat16, float16]"
 echo " --compute_dtype       Indicate the compute dtype[float32, bfloat16, float16]"
 echo " --backend             Indicate the torch compile backend[ipex, inductor]"
 echo " --device              Indicate the computation device[cpu, cuda, xpu]"
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
      -m | --model_id*)
        if ! has_argument $@; then
          echo "Model ID not specified." >&2
          usage
          exit 1
        fi

        model_id=$(extract_argument $@)

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
      --model_dtype)
        model_dtype=$(extract_argument $@)
        shift
        ;;
      --compute_dtype)
        compute_dtype=$(extract_argument $@)
        shift
        ;;
      --backend)
        backend=$(extract_argument $@)
        shift
        ;;
      --device)
        device=$(extract_argument $@)
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


if [[ "$device" = "cpu" ]]; then
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
fi 
export TORCHINDUCTOR_FREEZING=1
export TRITON_CODEGEN_INTEL_XPU_BACKEND=1
export OMP_NUM_THREADS=56

# Perform the desired actions based on the provided flags and arguments
numactl -C 0-55 --membind 0 python $task_name/run_$task_name.py --model_id $model_id --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device