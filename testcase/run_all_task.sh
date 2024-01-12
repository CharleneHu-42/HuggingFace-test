#!/bin/bash

# Default variable values
use_ipex_optimize=False
use_jit=False
use_torch_compile=False
task_name=""
model_dtype="float32"
compute_dtype="float32"
backend="inductor"
device="cpu"
batch_size=1
num_beams=4
input_tokens=32
output_tokens=32
ipex_optimize_transformers="False"
distributed=False

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help               Display this help message"
 echo " -i, --ipex_optimize      Use ipex optimize "
 echo " -j, --jit                Use jit "
 echo " -c, --compile            Use torch compile"
 echo " --model_dtype            Indicate the model dtype[float32, bfloat16, float16]"
 echo " --compute_dtype          Indicate the compute dtype[float32, bfloat16, float16]"
 echo " --backend                Indicate the torch compile backend[ipex, inductor]"
 echo " --device                 Indicate the computation device[cpu, cuda, xpu]"
 echo " --batch_size             Input batch size for text-generation"
 echo " --num_beams              The num_beams for text-generation"
 echo " --input_tokens           The input token length for text-generation[32, 64, 128, 256, 512, 1024]"
 echo " --output_tokens          The output token length for text-generation"
 echo " --ipex_optimize_transformers              Ipex optimize_transformers for text-generation"
 echo " --distributed         Whether to run fine-tuning in distributed mode, only used for fine-tune task"
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
        use_ipex_optimize=$(extract_argument $@)
        shift
        ;;
      -j | --jit)
        use_jit=$(extract_argument $@)
        shift
        ;;
      -c | --compile)
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
      --distributed)
        distributed=$(extract_argument $@)
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

echo "test image to image"
declare -a model_list
model_list=("stabilityai/stable-diffusion-xl-refiner-1.0" "timbrooks/instruct-pix2pix" "lambdalabs/sd-image-variations-diffusers")

for model in "${model_list[@]}"
do
    ./run.sh --task image-to-image --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device
    echo "----------------------------"
done


echo "test zero-shot-image-classification"
model_list=("openai/clip-vit-large-patch14" "openai/clip-vit-base-patch16" "openai/clip-vit-base-patch32")

for model in "${model_list[@]}"
do
    ./run.sh --task zero-shot-image-classification --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device
    echo "----------------------------"
done

echo "test sentence similarity"
model_list=("sentence-transformers/all-mpnet-base-v2" "sentence-transformers/all-MiniLM-L6-v2" "shibing624/text2vec-base-chinese")

for model in "${model_list[@]}"
do
    ./run.sh --task sentence-similarity --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device
    echo "----------------------------"
done

echo "test text-to-image"
model_list=("stabilityai/stable-diffusion-xl-base-1.0" "runwayml/stable-diffusion-v1-5" "stabilityai/stable-diffusion-2-1")

for model in "${model_list[@]}"
do
    ./run.sh --task text-to-image --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device
    echo "----------------------------"
done


echo "test text-generation"
model_list=("gpt2" "tiiuae/falcon-7b-instruct" "distilgpt2" "meta-llama/Llama-2-7b-chat-hf")

for model in "${model_list[@]}"
do
    ./run.sh --task text-generation --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device --batch_size $batch_size --num_beams $num_beams --input_tokens $input_tokens --output_tokens $output_tokens --ipex_optimize_transformers $ipex_optimize_transformers
    echo "----------------------------"
done

echo "test summarization"
model_list=("facebook/bart-large-cnn" "sshleifer/distilbart-cnn-12-6" "philschmid/bart-large-cnn-samsum")

for model in "${model_list[@]}"
do
    ./run.sh --task summarization --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device
    echo "----------------------------"
done

echo "test automatic-speech-recognition"
model_list=("jonatasgrosman/wav2vec2-large-xlsr-53-english" "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese" "facebook/wav2vec2-base-960h")

for model in "${model_list[@]}"
do
    ./run.sh --task automatic-speech-recognition --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device
    echo "----------------------------"
done

echo "test text-to-speech"
model_list=("microsoft/speecht5_tts" "suno/bark-small" "suno/bark")

for model in "${model_list[@]}"
do
    ./run.sh --task text-to-speech --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device
    echo "----------------------------"
done

echo "test image-to-text"
model_list=("nlpconnect/vit-gpt2-image-captioning" "Salesforce/blip-image-captioning-large" "Salesforce/blip-image-captioning-base")

for model in "${model_list[@]}"
do
    ./run.sh --task image-to-text --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device
    echo "----------------------------"
done

echo "test visual-question-answering"
model_list=("Salesforce/blip-vqa-base" "dandelin/vilt-b32-finetuned-vqa" "Salesforce/blip-vqa-capfilt-large")

for model in "${model_list[@]}"
do
    ./run.sh --task visual-question-answering --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device
    echo "----------------------------"
done

echo "test question-answering"
model_list=("bert-large-uncased-whole-word-masking-finetuned-squad")

for model in "${model_list[@]}"
do
    ./run.sh --task question-answering --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device
    echo "----------------------------"
done

echo "test image-classification"
model_list=("google/vit-base-patch16-224")

for model in "${model_list[@]}"
do
    ./run.sh --task image-classification --model_id $model --model_dtype $model_dtype --jit $use_jit --ipex_optimize $use_ipex_optimize --compute_dtype $compute_dtype --torch_compile $use_torch_compile --backend $backend --device $device
    echo "----------------------------"
done

echo "test fine-tune"
cd fine-tune
./run.sh --task fine-tune --device $device --distributed $distributed 
