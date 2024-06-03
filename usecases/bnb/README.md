## Bitsandbytes

1. Go to the tests/workloads folder
2. Install bitsandbytes with CPU backend:
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/ && git checkout multi-backend-refactor
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cpu -S .
make
pip install .
```


## Inference
Running inference by `sh run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16` and you can add the flag `--quant_type` to set quantization type, including `int8`, `nf4` and `fp4`.

## Finetune
Running lora finetune by `sh run.sh -t fine-tune --model_id meta-llama/Llama-2-7b-hf --num_processes 1`, also use `--quant_type` to set quantization type
