if [[ $# -ne 1 ]]; then
	echo "Usage: ./run_test.sh <MODEL ID>"
fi
MODEL=$1

python test_sync.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120
python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size 1
python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size 2
python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size 4
python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size 8
python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size 16
python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size 32
python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size 64
python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size 128
python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size 256
python test_async.py --model $MODEL --port 8081 --endpoint /embed --max_length=512 --save-result --result-dir ../../../hf-benchmarks --num-prompts 5120 --batch_size 512
