# HF Test Guide

## Inference

### CPU
We defaultly use BF16 OOB and BF16 + torch.compile in CPU, run the following command:
```bash
sh run_cpu.sh
```
After running this command, you can find the data in the `cpu_benmark.log`. Make sure you read the instruuction at the beginning of the log.


### XPU
Before running the testcases, please use the following command to first verify whether you are in the right XPU test environment:
```bash
source {DPCPPROOT}/env/vars.sh
source {MKLROOT}/env/vars.sh
source {CCLROOT}/env/vars.sh
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```
The command should return PyTorch* and Intel® Extension for PyTorch* versions installed, as well as GPU card(s) information detected. If it fails, you will need follow [the IPEX official documentation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu) to set-up the correct environment. Please note that the first 3 source commands are needed in order to use IPEX. But you only need to run it once for one termial session. For your reference, we have `env.sh` in the test folder. You can use `source env.sh` to activate the required oneAPI environment, if your oneAPI basekit is installed under `/opt/intel/oneapi`.

For Intel Native Experience on XPU, first install the testcase software dependencies:
```bash
pip install -r requirements_xpu.txt
```
**___Note: if the installation of transformers and accelerate fail, you will have to install them from source.___**

Then run the command:
```bash
./run_all_task_xpu.sh --model_dtype float16 --warm_up_steps 10 --run_steps 10 2>&1 | tee xpu_benchmark_raw.log 
```
If you want to compare the performance with NV GPU, just add the flag `--device cuda` to the command above.

When the test finishes, you can use the following command to extract the performance data from the log:
```bash
python analyse_logs.py --file_names xpu_benchmark_raw.log --out_name xpu_benchmark.log
```

## Fine-tune
### CPU
We defaultly use bf16 training with 4 DDP in a single instance, run the following command:
```bash
./run.sh --task fine-tune --device cpu
```

### XPU 
```bash
./run.sh --task fine-tune --device xpu
```

### CUDA 
```bash
./run.sh --task fine-tune --device cuda
```


## Notes
### Connection Error
If you cannot connect to huggingface model hub, please try `export HF_ENDPOINT=https://hf-mirror.com`
