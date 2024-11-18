## HuggingFace UT
This repository provides utilities to help you quickly and easily run unit tests of various HuggingFace libraries.  

### 1. Build docker image

```bash
./build_image.sh -d <device> 
```

`-d` options: "xpu" and "cuda"

### 2. Launch Docker Container

```bash 
./run_docker.sh -d xpu -t transformers 
```

Run `./run_docker.sh -h` for more options. By default, current directory will be mounted to `/mnt` directory of the container. You can specify your own mount directory.

### 3. Prepare test
In container, follow the below steps:

#### 3.1 Set up IPEX XPU environment
This step is only needed for XPU. 
```bash
source /opt/intel/oneapi/pti/latest/env/vars.sh
```

#### 3.2 Copy the helper scripts over
This step is only needed for transformers
```bash
cd transformers
# copy the helper scripts over 
cp /mnt/transformers/xpu/* .
```

### 4. Start testing
#### 4.1 optimum-quanto UT
```bash
cd optimum-quanto
pytest -rA test --excelreport $report | tee $log
```

#### 4.2 transformers UT
```bash
./run-ut.sh xpu /mnt/test_results
```
This script will create a folder `test_results` in `/mnt`, which organizes the test results in excel files for later analysis. 

#### 4.3 accelerate/peft/diffusers/trl UT

```bash
# take accelerate as example
cd accelerate  
RUN_SLOW=1 python -m pytest tests -sv --excelreport $report --timeout 600 2>&1 | tee $log
```
For TRL, you need to specify `export CUDA_VISIBLE_DEVICES=0,1` for tests requiring multi-gpu. 

#### 5. Analyze test results 
```bash 
python analyze_test_results.py --excel_dir /mnt/test_results --output_dir /mnt/test_stats
```
