#!/bin/bash
set -x

# Default variable values
ipex_optimize=False
jit=False
torch_compile=False
task_name=""
model_dtype="bfloat16"
autocast_dtype="float32"
backend="inductor"
device="cpu"
batch_size=1
num_beams=1
input_tokens=512
output_tokens=128
ipex_optimize_transformers="False"
warm_up_steps=10
run_steps=10
quant_algo="None"
quant_dtype="None"
optimum_intel="False"

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help               Display this help message"
 echo " -i, --ipex_optimize      Use ipex optimize "
 echo " -j, --jit                Use jit "
 echo " -c, --torch_compile      Use torch compile"
 echo " --model_dtype            Indicate the model dtype[float32, bfloat16, float16]"
 echo " --autocast_dtype         Indicate the compute dtype[float32, bfloat16, float16]"
 echo " --backend                Indicate the torch compile backend[ipex, inductor]"
 echo " --device                 Indicate the computation device[cpu, cuda, xpu]"
 echo " --batch_size             Input batch size for text-generation"
 echo " --num_beams              The num_beams for text-generation"
 echo " --input_tokens           The input token length for text-generation[32, 64, 128, 256, 512, 1024]"
 echo " --output_tokens          The output token length for text-generation"
 echo " --ipex_optimize_transformers              Ipex optimize_transformers for text-generation"
 echo " --warm_up_steps          The benchmark warm up steps for all tasks"
 echo " --run_steps              The benchmark run steps for all tasks"
 echo " --quant_algo          Use quant_algo to decide quantization method, options are ["bitsandbytes", "autoawq"]"
 echo " --quant_dtype         Use quant_dtype to decide quantization data type, like ["int8", "nf4", "fp4"] in bitsandbytes, ["int4"] in autoawq"
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
      --warm_up_steps)
        warm_up_steps=$(extract_argument $@)
        shift
        ;;
      --run_steps)
        run_steps=$(extract_argument $@)
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
      --optimum_intel)
        optimum_intel=$(extract_argument $@)
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

huggingface-cli login --token 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

base_log_dir=$(pwd)/logs/$(date +"%Y%m%d")

echo "test text-generation"
log_dir=$base_log_dir/text-generation/
# if [[ -n $OI_PAGED_ATTN_BLOCK_SIZE ]]; then
#   log_dir=$log_dir/blocksize_$OI_PAGED_ATTN_BLOCK_SIZE
# fi
mkdir -p $log_dir
model_list=("Qwen/Qwen2.5-1.5B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "openai-community/gpt2")
extended_model_list=("mistralai/Mistral-7B-Instruct-v0.3" "microsoft/Phi-3-mini-4k-instruct" "EleutherAI/gpt-j-6b" "baichuan-inc/Baichuan2-7B-Chat" "meta-llama/Llama-2-13b-hf" "facebook/opt-6.7b")
# model_list=("${model_list[@]}" "${extended_model_list[@]}")

# model_list=("Qwen/Qwen2.5-1.5B-Instruct")
# input_ouput_tokens=(
#   # "32,32"
#   "128,128"
#   "1024,128"
#   # "2016,128"
# )
# bs, beam, do_sample
generation_configs=(
  "1,1,False"
  "4,1,False"
  "1,4,False"
  "4,4,False"
  "1,1,True"
  "4,1,True"
)

for model in "${model_list[@]}"
do
  model_name=$(echo $model | awk -F'/' '{print $NF}' | tr '_' '-')
  if [[ $model_name =~ "gpt-j" ]]; then
    input_ouput_tokens=(
      "128,128"
      "1024,128"
    )
    if [[ $ipex_optimize == "False" ]]; then
      source /opt/conda/bin/activate gptj-fix
    fi
  elif [[ $model_name =~ "gpt2" ]]; then
    input_ouput_tokens=(
        "128,128"
        "512,128"
      )
  else
    input_ouput_tokens=(
      "128,128"
      "1024,128"
    )
  fi
  for gen_config in "${generation_configs[@]}"
  do
    IFS=',' read -r batch_size num_beams do_sample <<< "${gen_config}"
    for input_output in "${input_ouput_tokens[@]}"
    do
      IFS=',' read -r input_tokens output_tokens <<< "${input_output}"
      if [[ $optimum_intel == "True" ]]; then
        log_file=${log_dir}/${model_name}_${model_dtype}_optimum-intel_ipex_${ipex_optimize}_inductor_${torch_compile}_bs_${batch_size}_beam_${num_beams}_sample_${do_sample}_${input_tokens}_${output_tokens}.log
      else
        log_file=${log_dir}/${model_name}_${model_dtype}_ipex_${ipex_optimize}_inductor_${torch_compile}_bs_${batch_size}_beam_${num_beams}_sample_${do_sample}_${input_tokens}_${output_tokens}.log
      fi
      # log_file_names="${log_file_names},$log_file"

      /usr/bin/time -v ./run.sh --task text-generation --model_id $model --model_dtype $model_dtype --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --batch_size $batch_size --num_beams $num_beams --input_tokens $input_tokens --output_tokens $output_tokens --do_sample $do_sample --ipex_optimize_transformers $ipex_optimize_transformers --warm_up_steps $warm_up_steps --run_steps $run_steps --optimum_intel $optimum_intel 2>&1 | tee -a $log_file
    done
  done

  echo "----------------------------"

  if [[ $model_name =~ "gpt-j" ]] && [[ $ipex_optimize == "False" ]] && [[ $optimum_intel == "False" ]]; then
    source /opt/conda/bin/activate idp
  fi
  # log_file_names="${log_file_names:1}"
  # if [[ $bitsandbytes != "None" ]]; then
  #   output_log=$log_dir/gnr_${model_name}_bnb_benchmark.log
  # elif [[ $autoawq != "None" ]]; then
  #   output_log=$log_dir/gnr_${model_name}_autoawq_benchmark.log
  # else
  #   output_log=$log_dir/gnr_${model_name}_stock_eager_benchmark.log
  # fi
  # python analyse_logs.py --file_names $log_file_names --out_name $output_log
done
python collect_results.py -l $log_dir


echo "test question-answering"
log_dir=$base_log_dir/question-answering/
# if [[ -n $OI_PAGED_ATTN_BLOCK_SIZE ]]; then
#   log_dir=$log_dir/blocksize_$OI_PAGED_ATTN_BLOCK_SIZE
# fi
mkdir -p $log_dir
model_list=("distilbert/distilbert-base-cased-distilled-squad" "deepset/bert-large-uncased-whole-word-masking-squad2")

for model in "${model_list[@]}"; do
  if [[ $optimum_intel == "True" ]]; then
    log_file=${log_dir}/${model_name}_${model_dtype}_optimum-intel_ipex_${ipex_optimize}_inductor_${torch_compile}.log
  else
    log_file=${log_dir}/${model_name}_${model_dtype}_ipex_${ipex_optimize}_inductor_${torch_compile}.log
    fi
  bash ./run.sh --task question-answering --model_id $model --model_dtype $model_dtype --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps 2>&1 | tee -a $log_file
  
  echo "----------------------------"
done

python collect_results.py -l $log_dir

