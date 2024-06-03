## Bitsandbytes

1. Go to the tests/workloads folder
2. Install bitsandbytes with CPU backend:
```bash
git clone --branch multi-backend-refactor https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cpu -S .
make
pip install .
```


## Inference
Running inference by `sh run.sh -t text-generation --model_dtype bfloat16` and you can add the flag `--quant_type` to set quantization type, including `int8`, `nf4` and `fp4`.

## Finetune
Running lora finetune by `sh run.sh -t fine-tune --num_processes 1`, also use `--quant_type` to set quantization type
