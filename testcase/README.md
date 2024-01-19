# HF Test Guide

## Inference

### CPU

### XPU
Before running the testcases, please run the following command to first verify whether you are in the right environment:
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

### XPU 
```bash
./run.sh --task fine-tune --device xpu
```


## Notes
### Connection Error
If you cannot connect to huggingface model hub, please try `export HF_ENDPOINT=https://hf-mirror.com`

### Command Usage 
To run individual task:
```bash 
sh run.sh --task task_name --model_id model_name
```
`task_name` should be the same with folders name. For example: `text-generation`

To accelerate inference with bfloat16, ipex_optimize and jit, use the following command
```
sh run.sh --task task_name --model_id model_name --model_dtype bfloat16 --compute_dtype bfloat16 --ipex_optimize True --jit True
```
**___Note: The ipex and jit optimizations may failed in some tasks.___**


### Text-Generation
For text-genetation task, you can control the input and output and use greedy search by add the follwing flags:
```
--batch_size 1 --num_beams 1 --input_tokens 1024 --output_tokens 32
```
**___Note: Default values are batch_size=1, num_beams=4, input_tokens=32, output_tokens=32.___**

You can also use ipex optimize transformers by adding the flag `--ipex_optimize_transformers True`, but it doesn't work for now.

### Finetune
We use `accelerate launch` to start finetune on XPU and GPU, use `mpirun python` on CPU to enable distributed finetune(need to install one-ccl on CPU). 
The finetune task defaultly use `fp16` and `ipex` on XPU and GPU, `bf16` on CPU. If you want to run finetune separately and use `ipex` on CPU, please run the follwing script:
```
sh run.sh --task_name fine-tune --ipex_optimize True --device cpu 
``` 
You can also add `--gradient_checkpointing True` to use gradient checkpointing.


### Test Data 
For text prompts and speech demos can be found [here](https://drive.google.com/drive/folders/1PbGjFGuPgSxTqK3tC1UKP7sF0cib1pyd?usp=drive_link).



