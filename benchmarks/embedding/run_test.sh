#!/bin/bash
set -x

PORT=8080
ENDPOINT=/embed
NUM_PROMPTS=5120
MAX_SEQ_LEN=512
MAX_BS=512
RESULTS_DIR=logs

SEED=42
huggingface-cli login --token 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

usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help            Display this help message"
 echo " -m, --model_id        Model id"
 echo " --port                Port number of TEI service"
 echo " --max_batch_size      Maximum batch size for benchmark"
 echo " --max_seq_len         Maximum sequence length  for benchmark"
 echo " --num_prompts         Number of prompts for benchmark"
 echo " --results_dir         Directory to save results"
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
      -m | --model_id*)
        if ! has_argument $@; then
          echo "Model ID not specified." >&2
          usage
          exit 1
        fi
        MODEL=$(extract_argument $@)
        shift
        ;;
      --port)
        PORT=$(extract_argument $@)
        shift
        ;;
      --max_batch_size)
        MAX_BS=$(extract_argument $@)
        shift
        ;;
      --max_seq_len)
        MAX_SEQ_LEN=$(extract_argument $@)
        shift
        ;;
      --num_prompts)
        NUM_PROMPTS=$(extract_argument $@)
        shift
        ;;
      --results_dir)
        RESULTS_DIR=$(extract_argument $@)
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

mkdir -p $RESULTS_DIR/results
echo "Starting benchmarking suite for $MODEL to https://127.0.0.1:$PORT$ENDPOINT with $NUM_PROMPTS prompts."

# Use current dir as temp cache to avoid permission issue
export HF_HOME=$(pwd)/hf_cache
mkdir -p $HF_HOME

echo "python test_sync.py --model $MODEL --port $PORT --endpoint $ENDPOINT --max_length=$MAX_SEQ_LEN --save-result --result-dir $RESULTS_DIR --num-prompts $NUM_PROMPTS "
python test_sync.py --model $MODEL --port $PORT --endpoint $ENDPOINT --max_length=$MAX_SEQ_LEN --save-result \
        --result-dir $RESULTS_DIR \
        --num-prompts $NUM_PROMPTS --max-bs $MAX_BS --seed $SEED

for ((i = 0; i < 10; i++)); do
    batch_size=$((2 ** $i))
    if [[ $batch_size -gt $MAX_BS ]]; then
        break
    fi
    echo "python test_async.py --model $MODEL --port $PORT --endpoint $ENDPOINT --max_length=$MAX_SEQ_LEN --save-result --result-dir $RESULTS_DIR --num-prompts $NUM_PROMPTS --batch_size $batch_size"
    python test_async.py --model $MODEL --port $PORT --endpoint $ENDPOINT \
            --max_length=$MAX_SEQ_LEN --save-result \
            --result-dir $RESULTS_DIR \
            --num-prompts $NUM_PROMPTS \
            --batch_size $batch_size --seed $SEED
done
