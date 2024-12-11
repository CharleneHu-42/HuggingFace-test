## HuggingFace UT
This repository provides utilities to help you quickly and easily run unit tests of various HuggingFace libraries.  

### 1. Build docker image

```bash
./build_image.sh -d <device> 
```

`-d` options: "xpu" and "cuda"

### 2. Launch docker container

```bash 
./run_docker.sh -d xpu -t <test_case>
```

Run `./run_docker.sh -h` for more options. By default, current directory will be mounted to `/mnt` directory of the container. You can specify your own mount directory.

test_case options:
  - optimum-quanto
  - transformers
  - accelerate
  - peft
  - diffusers
  - trl

### 3. Run test in container
#### 3.1 optimum-quanto UT
```bash
cd optimum-quanto
pytest -rA test --excelreport $report | tee $log
```

#### 3.2 transformers UT
```bash
# first dry run to detect anomalies in advanced
./run-ut.sh xpu transformers 1
# run all UTs
./run-ut.sh xpu transformers
```
This script will create a folder `test_results` in `/mnt/transformers`, where test results are stored in excel files for later analysis. 


#### 3.3 accelerate/peft/diffusers/trl UT
```bash
cd <taget_case_folder>  # e.g. accelerate, peft, diffusers, trl

RUN_SLOW=1 python -m pytest tests -sv --excelreport $report --timeout 600 2>&1 | tee $log
```
For `trl`, you need to set `export CUDA_VISIBLE_DEVICES=0,1` for multi-card tests. 

### 4. Analyze test results 
Depending on the library, different steps might be needed for the analysis. You can take the `playgound.py` template as a guidance to conduct the analysis.