#!/bin/bash

if [[ $# -ne 1 ]]; then
	echo "Usage: ./run_test.sh <MODEL ID>"
	exit 1
fi
MODEL=$1

echo "python test_sync.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 "
python test_sync.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120

for ((i = 0; i < 10; i++)); do
	batch_size=$((2 ** $i))
	echo "python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size $batch_size"
	python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size $batch_size
done
