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
 echo " -h, --help               Display this help message"
 echo " -m, --model              Model ID"
 echo " -i, --ipex_optimize      Use ipex optimize "
 echo " -j, --jit                Use jit "
 echo " -c, --compile            Use torch compile"
 echo " -b, --bf16               Use amp bf16"
 echo " --torch_dtype            Indicate the model dtype[float32, bfloat16]"
 echo " --backend                Indicate the torch compile backend[ipex, inductor]"
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
      -c | --compile)
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

echo "test image to image"
declare -a model_list
model_list=("stabilityai/stable-diffusion-xl-refiner-1.0" "timbrooks/instruct-pix2pix" "lambdalabs/sd-image-variations-diffusers")

for model in "${model_list[@]}"
do
    ./run.sh --task image-to-image --model $model --bf16 $use_bf16 --jit $use_jit --ipex_optimize $use_ipex_optimize --torch_dtype $torch_dtype --torch_compile $use_torch_compile --backend $backend
done


echo "test zero-shot-image-classification"
model_list=("openai/clip-vit-large-patch14" "openai/clip-vit-base-patch16" "openai/clip-vit-base-patch32")

for model in "${model_list[@]}"
do
    ./run.sh --task zero-shot-image-classification --model $model --bf16 $use_bf16 --jit $use_jit --ipex_optimize $use_ipex_optimize --torch_dtype $torch_dtype --torch_compile $use_torch_compile --backend $backend
done

echo "test sentence similarity"
model_list=("sentence-transformers/all-mpnet-base-v2" "sentence-transformers/all-MiniLM-L6-v2" "shibing624/text2vec-base-chinese")

for model in "${model_list[@]}"
do
    ./run.sh --task sentence-similarity --model $model --bf16 $use_bf16 --jit $use_jit --ipex_optimize $use_ipex_optimize --torch_dtype $torch_dtype --torch_compile $use_torch_compile --backend $backend
done

echo "test text-to-image"
model_list=("stabilityai/stable-diffusion-xl-base-1.0" "runwayml/stable-diffusion-v1-5" "stabilityai/stable-diffusion-2-1")

for model in "${model_list[@]}"
do
    ./run.sh --task text-to-image --model $model --bf16 $use_bf16 --jit $use_jit --ipex_optimize $use_ipex_optimize --torch_dtype $torch_dtype --torch_compile $use_torch_compile --backend $backend
done


echo "test text-generation"
model_list=("gpt2" "tiiuae/falcon-7b-instruct" "distilgpt2")

for model in "${model_list[@]}"
do
    ./run.sh --task text-generation --model $model --bf16 $use_bf16 --jit $use_jit --ipex_optimize $use_ipex_optimize --torch_dtype $torch_dtype --torch_compile $use_torch_compile --backend $backend
done

echo "test summarization"
model_list=("facebook/bart-large-cnn" "sshleifer/distilbart-cnn-12-6" "philschmid/bart-large-cnn-samsum")

for model in "${model_list[@]}"
do
    ./run.sh --task summarization --model $model --bf16 $use_bf16 --jit $use_jit --ipex_optimize $use_ipex_optimize --torch_dtype $torch_dtype --torch_compile $use_torch_compile --backend $backend
done

echo "test automatic-speech-recognition"
model_list=("jonatasgrosman/wav2vec2-large-xlsr-53-english" "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese" "pyannote/speaker-diarization-3.0")

for model in "${model_list[@]}"
do
    ./run.sh --task automatic-speech-recognition --model $model --bf16 $use_bf16 --jit $use_jit --ipex_optimize $use_ipex_optimize --torch_dtype $torch_dtype --torch_compile $use_torch_compile --backend $backend
done

echo "test text-to-speech"
model_list=("microsoft/speecht5_tts" "suno/bark-small" "suno/bark")

for model in "${model_list[@]}"
do
    ./run.sh --task text-to-speech --model $model --bf16 $use_bf16 --jit $use_jit --ipex_optimize $use_ipex_optimize --torch_dtype $torch_dtype --torch_compile $use_torch_compile --backend $backend
done

echo "test image-to-text"
model_list=("nlpconnect/vit-gpt2-image-captioning" "Salesforce/blip-image-captioning-large" "Salesforce/blip-image-captioning-base")

for model in "${model_list[@]}"
do
    ./run.sh --task image-to-text --model $model --bf16 $use_bf16 --jit $use_jit --ipex_optimize $use_ipex_optimize --torch_dtype $torch_dtype --torch_compile $use_torch_compile --backend $backend
done

echo "test visual-question-answering"
model_list=("Salesforce/blip-vqa-base" "dandelin/vilt-b32-finetuned-vqa" "Salesforce/blip-vqa-capfilt-large")

for model in "${model_list[@]}"
do
    ./run.sh --task visual-question-answering --model $model --bf16 $use_bf16 --jit $use_jit --ipex_optimize $use_ipex_optimize --torch_dtype $torch_dtype --torch_compile $use_torch_compile --backend $backend
done


