# TRL on XPU&GPU
This repository includes examples and training recipes to fine-tune large language modles using the ORPO and KTO algorithm with TRL.


## ORPO

### Installtion 
To install the necessary dependencies, run the following command:
```bash
pip install -U transformers datasets accelerate peft trl wandb
```

### Usage 
#### 1. Specify the visible devices

For single-card usage:
```bash
# on XPU
export ZE_AFFINITY_MASK=0
# on CUDA
export CUDA_VISIBLE_DEVICES=0
```

#### 2. Run training 
You can run the ORPO training script with custom arguments. Here is an example:
```bash
python run_orpo.py --base_model meta-llama/Meta-Llama-3-8B --model_save_dir OrpoLlama-3-8B --attn_type eager
```
To find out more options, run:
```bash
python run_orpo.py -h
```

## KTO 



