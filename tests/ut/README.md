## HuggingFace UT

This folder provides utilities to run unit tests of various HuggingFace libraries easily.

Assume you are running command in the directory of where this README is.

### 1. build docker image

```bash
$ cd ../../HuggingFace/docker
$ bash ./build_image.sh -d <device> -t ut
```

`-d` options: "xpu" and "cuda"

### 2. launch docker container

```bash 
$ cd -
$ bash ./run_docker.sh -d xpu -l <target_test_library>
```
You can run `./run_docker.sh -h` for more options. By default, current directory will be mounted to `/mnt` directory of the container. You can specify your own mount directory.

target test library options:
  - optimum-quanto
  - transformers
  - accelerate
  - peft
  - diffusers
  - trl

### 3. run test in container

#### 3.1 optimum-quanto UT

```bash
$ pytest -rA test --excelreport $report 2>&1 | tee $log
```

#### 3.2 transformers UT
```bash
# first copy over helper scripts
cp /mnt/spec_*.py /mnt/run_ut.sh .
# first dry run to detect anomalies in advance
./run-ut.sh xpu transformers 1
# run all UTs
./run-ut.sh xpu transformers
```
This script will create a folder `test_results` in `/mnt/transformers`, where test results are stored in excel files for later analysis. 

#### 3.3 accelerate UT
```bash
./run-ut.sh xpu accelerate 1
./run-ut.sh xpu accelerate
```

#### 3.4 peft/diffusers/trl UT
```bash
$ RUN_SLOW=1 python -m pytest tests -sv --excelreport $report --timeout 600 2>&1 | tee $log
```
For `trl`, you need to set `export CUDA_VISIBLE_DEVICES=0,1` for multi-card tests. 

### 4. Analyze test results 
Depending on the library, different steps might be needed for the analysis. You can take the `playgound.py` template as a guidance to conduct the analysis.