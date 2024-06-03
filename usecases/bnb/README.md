## Bitsandbytes

1. Install bitsandbytes with CPU backend in a proper folder:
```bash
git clone --branch multi-backend-refactor https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cpu -S .
make
pip install .
```
2. Go to the tests/workloads folder

## Inference
Running inference by `sh run.sh -t text-generation --model_dtype bfloat16` and you can add the flag `--quant_type` to set quantization type, including `int8`, `nf4` and `fp4`, for example:
Run int8 inference: `sh run.sh -t text-generation --model_dtype bfloat16 --quant_type int8`
Run nf4 inference: `sh run.sh -t text-generation --model_dtype bfloat16 --quant_type nf4`

## Finetune
Running lora finetune by `sh run.sh -t fine-tune`, also use `--quant_type` to set quantization type, for example:
Run int8 lora: `sh run.sh -t fine-tune --quant_type int8`
