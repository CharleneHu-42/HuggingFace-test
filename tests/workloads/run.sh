#!/bin/bash

# Default variable values

## test case configurations
task_name=""
model_id=""
model_dtype="float32"
autocast_dtype="float32"
batch_size=1
num_beams=1
input_tokens=32
output_tokens=32
gradient_checkpointing=False
quant_algo="None"
quant_dtype="None"

## runtime configurations
backend="inductor"
device="cpu"
num_processes=4
warm_up_steps=10
run_steps=10

## optimization options
ipex_optimize=False
ipex_optimize_transformers=False
jit=False
torch_compile=False
optimum_intel=False

# arg parser
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help            Display this help message"
 echo " -t, --task            Specify task name"
 echo " -m, --model_id        Specify model id"
 echo " --model_dtype         Indicate the model dtype[float32, bfloat16, float16]"
 echo " --autocast_dtype      Indicate the compute dtype[float32, bfloat16, float16]"
 echo " --batch_size          Input batch size for text-generation"
 echo " --num_beams           The num_beams for text-generation"
 echo " --input_tokens        The input token length for text-generation[32, 64, 128, 256, 512, 1024]"
 echo " --output_tokens       The output token length for text-generation"
 echo " --gradient_checkpointing  Whether to run fine-tuning with gradient checkpoint to save memory, only used for fine-tune task"
 echo " --quant_algo          Use quant_algo to decide quantization method, options are ["bitsandbytes", "autoawq"]"
 echo " --quant_dtype         Use quant_dtype to decide quantization data type, like ["int8", "nf4", "fp4"] in bitsandbytes, ["int4"] in autoawq"
 echo " --backend             Indicate the torch compile backend[ipex, inductor]"
 echo " --device              Indicate the computation device[cpu, cuda, xpu]"
 echo " --num_processes       The number of data parallelism, only used for CPU fine-tune task"
 echo " --warm_up_steps       The benchmark warm up steps for all tasks"
 echo " --run_steps           The benchmark run steps for all tasks"
 echo " -i, --ipex            Use ipex optimize"
 echo " -j, --jit             Use jit"
 echo " -c, --torch_compile   Use torch compile"
 echo " --ipex_optimize_transformers              Ipex optimize_transformers for text-generation"
 echo " --optimum_intel       Use optimum-intel optimization"
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
        ipex_optimize=$(extract_argument $@)
        shift
        ;;
      -j | --jit)
        jit=$(extract_argument $@)
        shift
        ;;
      -c | --torch_compile)
        torch_compile=$(extract_argument $@)
        shift
        ;;
      --model_dtype)
        model_dtype=$(extract_argument $@)
        shift
        ;;
      --autocast_dtype)
        autocast_dtype=$(extract_argument $@)
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
      --batch_size)
        batch_size=$(extract_argument $@)
        shift
        ;;
      --num_beams)
        num_beams=$(extract_argument $@)
        shift
        ;;
      --input_tokens)
        input_tokens=$(extract_argument $@)
        shift
        ;;
      --output_tokens)
        output_tokens=$(extract_argument $@)
        shift
        ;;
      --ipex_optimize_transformers)
        ipex_optimize_transformers=$(extract_argument $@)
        shift
        ;;
      --gradient_checkpointing)
        gradient_checkpointing=$(extract_argument $@)
        shift
        ;;
      --num_processes)
        num_processes=$(extract_argument $@)
        shift
        ;;
      --warm_up_steps)
        warm_up_steps=$(extract_argument $@)
        shift
        ;;
      --run_steps)
        run_steps=$(extract_argument $@)
        shift
        ;;
      --optimum_intel)
        optimum_intel=$(extract_argument $@)
        shift
        ;;
      --quant_algo)
        quant_algo=$(extract_argument $@)
        shift
        ;;
      --quant_dtype)
        quant_dtype=$(extract_argument $@)
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

CORES=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
export TORCHINDUCTOR_FREEZING=1
export TORCHINDUCTOR_CPP_WRAPPER=1
export TRITON_CODEGEN_INTEL_XPU_BACKEND=1
export OMP_NUM_THREADS=${CORES}
export TORCHINDUCTOR_CPP_MIN_CHUNK_SIZE=${CORES}

small_model_list=("Helsinki-NLP/opus-mt-mul-en" "google-t5/t5-small")
for item in "${small_model_list[@]}"; do
  if [[ "$item" == "$model_id" ]]; then
    CORES=4
    break
  fi
done

export OMP_NUM_THREADS=${CORES}
export TORCHINDUCTOR_CPP_MIN_CHUNK_SIZE=${CORES}

# Perform the desired actions based on the provided flags and arguments
if [[ "$task_name" == "fine-tune" ]]; then
  if [[ "$device" == "cpu" ]]; then
    export CCL_WORKER_COUNT=1
    accelerate launch --config_file $task_name/"$device"_config.yaml $task_name/run_$task_name.py --base_model $model_id --use_ipex $ipex_optimize --quant_algo $quant_algo --quant_dtype $quant_dtype --device $device
  else
    accelerate launch --config_file $task_name/"$device"_config_ddp.yaml $task_name/run_$task_name.py --base_model $model_id --quant_algo $quant_algo --quant_dtype $quant_dtype --device $device
  fi
else
  numactl -C '0-'${CORES} --membind 0 python $task_name/run_$task_name.py --model_id $model_id --model_dtype $model_dtype --quant_algo $quant_algo --quant_dtype $quant_dtype --jit $jit --ipex_optimize $ipex_optimize --autocast_dtype $autocast_dtype --torch_compile $torch_compile --backend $backend --device $device --batch_size $batch_size --num_beams $num_beams --input_tokens $input_tokens --output_tokens $output_tokens --ipex_optimize_transformers $ipex_optimize_transformers --warm_up_steps $warm_up_steps --run_steps $run_steps --optimum_intel $optimum_intel
fi
