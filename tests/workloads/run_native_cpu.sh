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

tasks=(
  "image-to-image"
  "image-to-text"
  "image-feature-extraction"
  "text-to-image"
  "text-to-video"
  "zero-shot-image-classification"
  "sentence-similarity"
  "question-answering"
  "text-generation"
  "summarization"
  "translation"
  "automatic-speech-recognition"
  "text-to-speech"
  "audio-classification"
  "visual-question-answering"
  "document-question-answering"
  "image-text-to-text"
  "video-text-to-text"
)
declare -A model_list
model_list["image-to-image"]="stabilityai/stable-diffusion-2-inpainting stabilityai/stable-diffusion-xl-refiner-1.0 lllyasviel/sd-controlnet-canny timbrooks/instruct-pix2pix"
model_list["image-to-text"]="nlpconnect/vit-gpt2-image-captioning Salesforce/blip-image-captioning-large Salesforce/blip-image-captioning-base"
model_list["image-feature-extraction"]="google/vit-base-patch16-224-in21k facebook/dinov2-base facebook/dinov2-small"
model_list["text-to-image"]="stable-diffusion-v1-5/stable-diffusion-v1-5 stable-diffusion-v1-5/stable-diffusion-inpainting stabilityai/stable-diffusion-xl-base-1.0"
model_list["text-to-video"]="THUDM/CogVideoX-5b ByteDance/AnimateDiff-Lightning THUDM/CogVideoX-2b"
model_list["zero-shot-image-classification"]="openai/clip-vit-large-patch14 openai/clip-vit-base-patch16 openai/clip-vit-base-patch32"
model_list["sentence-similarity"]="sentence-transformers/all-mpnet-base-v2 sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_list["question-answering"]="deepset/roberta-base-squad2"
model_list["text-generation"]="facebook/opt-1.3b"
model_list["summarization"]="facebook/bart-large-cnn sshleifer/distilbart-cnn-12-6 google/pegasus-xsum"
model_list["translation"]="google-t5/t5-small google-t5/t5-base Helsinki-NLP/opus-mt-mul-en"
model_list["automatic-speech-recognition"]="openai/whisper-large-v2 jonatasgrosman/wav2vec2-large-xlsr-53-english openai/whisper-small"
model_list["text-to-speech"]="microsoft/speecht5_tts suno/bark facebook/mms-tts-eng"
model_list["audio-classification"]="facebook/mms-lid-256 MIT/ast-finetuned-audioset-10-10-0.4593 alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
model_list["visual-question-answering"]="Salesforce/blip-vqa-capfilt-large Salesforce/blip-vqa-base dandelin/vilt-b32-finetuned-vqa"
model_list["document-question-answering"]="impira/layoutlm-document-qa naver-clova-ix/donut-base-finetuned-docvqa impira/layoutlm-invoices"
model_list["image-text-to-text"]="meta-llama/Llama-3.2-11B-Vision-Instruct llava-hf/llava-v1.6-mistral-7b-hf Qwen/Qwen2-VL-7B-Instruct"
model_list["video-text-to-text"]="llava-hf/LLaVA-NeXT-Video-7B-hf KangarooGroup/kangaroo"

base_log_dir=$(pwd)/logs/$(date +"%Y%m%d")
for task in "${tasks[@]}"
do
  echo "test $task ......"
  log_dir=$base_log_dir/$task
  mkdir -p $log_dir
  IFS=' ' read -r -a model_array <<< "${model_list[$task]}"
  for model in "${model_array[@]}"
  do
    echo "test $model ......"
    model_name=$(echo $model | awk -F'/' '{print $NF}' | tr '_' '-')
    if [[ $task == "text-generation" || $task == "summarization" || $task == "translation" ]]; then
      if [[ $task == "summarization" ]]; then
        beam_list=(1)
        input_ouput_tokens=(
          "512,128"
        )
      elif [[ $task == "translation" ]]; then
        beam_list=(1)
        input_ouput_tokens=(
          "128,128"
        )
      else
        beam_list=(1 4)
        input_ouput_tokens=(
          "128,128"
          "1024,128"
        )
      fi
      for num_beams in "${beam_list[@]}"
      do
        do_sample=False
        for input_output in "${input_ouput_tokens[@]}"
        do
          IFS=',' read -r input_tokens output_tokens <<< "${input_output}"
          log_file=${log_dir}/${model_name}_${model_dtype}_ipex_${ipex_optimize}_inductor_${torch_compile}_bs_${batch_size}_beam_${num_beams}_sample_${do_sample}_${input_tokens}_${output_tokens}.log
          ./run.sh --task $task --model_id $model --model_dtype $model_dtype --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --batch_size $batch_size --num_beams $num_beams --input_tokens $input_tokens --output_tokens $output_tokens --do_sample $do_sample 2>&1 | tee -a $log_file
        done
      done
    else
      # Replace separate execution with a loop, assuming autocast_dtype is default value float32
      log_file=${log_dir}/${model_name}_${model_dtype}_ipex_${ipex_optimize}_inductor_${torch_compile}_bs_${batch_size}.log
      bash ./run.sh --task $task --model_id $model --model_dtype $model_dtype --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps 2>&1 | tee -a $log_file
    fi
    echo "----------------------------"
  done
  python collect_results.py -l $log_dir
done
