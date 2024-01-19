# HF Test Guide

## Inference

### CPU
We defaultly running BF16 OOB and BF16 + torch.compile in CPU, run the following command:
```bash
sh run_cpu.sh
```
After running this command, you can find the data in the `cpu_benmark.log`. Make sure you read the instruuction at the beginning of the log.


### XPU
Before running the testcases, please run the following command to first verify whether you are in the right XPU environment:
```bash
source {DPCPPROOT}/env/vars.sh
source {MKLROOT}/env/vars.sh
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```
The command should return PyTorch* and Intel® Extension for PyTorch* versions installed, as well as GPU card(s) information detected. If it fails, you will need follow [the IPEX official documentation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu) to set-up the correct environment. Please note that the first 2 source commands are needed in order to use IPEX. But you only need to run it once for one termial session. For your reference, we attached `env.sh` in the test folder. You can run `source env.sh` to activate the oneAPI environment.  

```bash
./run_all_task_xpu.sh --model_dtype float16 --warm_up_steps 10 -- run_steps 10
```
If you want to compare the performance with NV GPU, just add the flag `--device cuda` to the command above.  



## Fine-tune
### CPU
We defaultly use bf16 training with 4 DDP in a single instance, run the following command.
```bash
./run.sh --task fine-tune --device cpu
```

### XPU 
```bash
./run.sh --task fine-tune --device xpu
```


## Notes
### Connection Error
If you cannot connect to huggingface model hub, please try `export HF_ENDPOINT=https://hf-mirror.com`
