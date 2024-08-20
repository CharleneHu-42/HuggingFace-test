#!/bin/bash

if [[ $# -lt 1 ]]; then
	echo "Usage: ./run_test.sh <MODEL ID> [benchmark-opts]"
	exit 1
fi

MODEL=$1
shift
PORT=8081
ENDPOINT=/embed
NUM_PROMPTS=5120

mkdir -p $RESULTS_DIR/results
echo "Starting benchmarking suite for $MODEL to https://127.0.0.1:$PORT$ENDPOINT with $NUM_PROMPTS prompts."

echo "Ensuring ownership of HF Cache"
echo "sudo chown -R $USER ~/.cache/huggingface/hub"
sudo chown -R $USER ~/.cache/huggingface/hub

echo "python test_sync.py --model $MODEL --port $PORT --endpoint $ENDPOINT --max_length=512 --save-result --result-dir $RESULTS_DIR --num-prompts $NUM_PROMPTS "
python test_sync.py --model $MODEL --port $PORT --endpoint $ENDPOINT --max_length=512 --save-result \
	--result-dir $RESULTS_DIR \
	--num-prompts $NUM_PROMPTS $@

for ((i = 0; i < 10; i++)); do
	batch_size=$((2 ** $i))
	echo "python test_async.py --model $MODEL --port $PORT --endpoint $ENDPOINT --max_length=512 --save-result --result-dir $RESULTS_DIR --num-prompts $NUM_PROMPTS --batch_size $batch_size"
	python test_async.py --model $MODEL --port $PORT --endpoint $ENDPOINT \
		--max_length=512 --save-result \
		--result-dir $RESULTS_DIR \
		--num-prompts $NUM_PROMPTS \
		--batch_size $batch_size $@
done
