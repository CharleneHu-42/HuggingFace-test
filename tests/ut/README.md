## HuggingFace UT
This repository provides utilities to help you quickly and easily run unit tests of various HuggingFace libraries.  

1. Build docker image
```bash
./build-image.sh -d xpu 
```
`-d` stands for device. Use `-d cuda` to build docker image for cuda.

2. Start Docker Container
```bash 
./run-docker.sh -d xpu -t transformers 
```
Run `./run-docker.sh -h` for more options. By default, the current directory will be mounted to `/mnt` inside the container. You can specify your own local directory

3. Prepare for test
Once inside the container, follow the below steps:

3.1 Set up IPEX XPU environment 
This step is only needed for XPU. 
```bash
source tools/env_activate.sh fine-tuning
```

3.2 Copy the helper scripts over
This step is only needed for transformers
```bash
cd transformers
# copy the helper scripts over 
cp /mnt/transformers/xpu/* .
```

4. Start testing
4.1 For transformers' UT
```bash
# run tests
./run-ut.sh xpu /mnt/test_results
```
This script will create a folder `test_results` in `/mnt`, which organizes the test results in excel files for later analysis. 

4.2 For peft/accelerate/diffusers/trl
```bash
# take accelerate as example
cd accelerate  
RUN_SLOW=1 python -m pytest tests -sv --excelreport $report --timeout 600 2>&1 | tee $log
```
For TRL, you need to specify `export CUDA_VISIBLE_DEVICES=0,1` for tests that require multi-gpu set-up. 

5. Analyze test results 
```bash 
python analyze_test_results.py --excel_dir /mnt/test_results --output_dir /mnt/test_stats
```


