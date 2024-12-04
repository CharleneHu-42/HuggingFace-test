# HF Workload Test Guide

**All tested are and should be validated in docker container**

## Prepare env

```bash
pip install -r requirements.txt
```
**___Note: if the installation of transformers and accelerate fail, you will have to install them from source.___**


## Inference

### CPU
#### Native DX
By default, We use `BF16` and `BF16 + torch.compile` on CPU for all inference tasks, run the following command:

```bash
sh run_cpu.sh
```

After running this command, you can find the data in the `cpu_benmark.log`. Make sure you read the instruction at the beginning of the log.

#### Advanced DX
For advanced DX, `optimum-intel` is required and can be installed with the following commands:

```bash
git clone https://github.com/huggingface/optimum-intel.git && cd optimum-intel
pip install .
```

Use `--optimum_intel` in `image-classification`, `question-answering`, and `text-generation` can enable optimum-intel optimization, for example:

```bash
sh run.sh -t image-classification -m google/vit-base-patch16-224 --model_dtype bfloat16 --optimum_intel True
```

### XPU
Before running the test cases, you need to follow [the IPEX official documentation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu) to set up the correct environment.

#### Native DX
Run below command:

```bash
./run_all_task_xpu.sh --model_dtype float16 --warm_up_steps 10 --run_steps 10 2>&1 | tee xpu_benchmark_raw.log 
```

If you want to compare the performance with NV GPU, just add the flag `--device cuda` to the command above.

When the test finishes, you can use the following command to extract the performance data from the log:

```bash
python analyse_logs.py --file_names xpu_benchmark_raw.log --out_name xpu_benchmark.log
```
#### Advanced DX
<to be filled>

## Finetune
### CPU
We defaultly use amp bf16 to train [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) in [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset with 4 DDP across 4 instances. Please change the [fine-tune/hostfile](https://github.com/intel-sandbox/HuggingFace/blob/main/tests/workloads/fine-tune/hostfile) to your instances ip and run the following command:

```bash
./run.sh -t fine-tune -m meta-llama/Llama-2-7b-hf --device cpu
```

### XPU 

```bash
./run.sh -t fine-tune -m meta-llama/Llama-2-7b-hf --device xpu
```

### CUDA 

```bash
./run.sh -t fine-tune -m meta-llama/Llama-2-7b-hf --device cuda
```

## Notes

### Connection Error
If you cannot connect to huggingface model hub, please try `export HF_ENDPOINT=https://hf-mirror.com`

### Batch Size
The actual batch size is calculated as:
```math
micro\_batch\_size \times gradient\_accumulation\_steps \times world\_size
```

where the `gradient_accumulation_steps` is calculated using the following formula:
```math
gradient\_accumulation\_steps = batch\_size // micro\_batch\_size // world\_size
```
Please make sure that the input batch size is divisible by `micro_batch_size` and `world_size`, otherwise the actual batch size will differ from the the given batch size, e.g. with `batch_size=128`, `micro_batch_size=4` and `world_size=6`, the actual batch size is equal to 120, rather than 128.
