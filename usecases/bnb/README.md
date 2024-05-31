## Bitsandbytes

1. Get ipex docker image by: `docker pull intel/intel-extension-for-pytorch:2.3.0-pip-base`
2. Get into the container run the following command:
```bash
apy update && apt-get install -y python3-dev git build-essential cmake
pip install transformers accelerate datasets
```
3. Install bitsandbytes with CPU backend:
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/ && git checkout multi-backend-refactor
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cpu -S .
make
pip install .
```


## Inference
Running inference by `python bnb_inference` and you can add the flag `--quant_type` to set quantization type, including `int8`, `nf4` and `fp4`.

## Finetune
Running lora finetune by `python bnb_lora`, also use `--quant_type` to set quantization type, for example:
Int8 lora: `python bnb_lora --quant_type int8`
NF4 lora: `python bnb_lora --quant_type nf4`
FP4 lora: `python bnb_lora --quant_type fp4`

