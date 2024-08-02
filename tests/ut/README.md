## HuggingFace UT
This repository provides utilities to help you quickly and easily run unit tests of various HuggingFace libraries.  

1. Build docker image
To run transformers' UT on XPU, you need to first build the docker image.
```bash
./build-image.sh -d xpu -t transformers 
```
Run `./build-image.sh -h` for more options. 

2. Start Docker Container
```bash 
./run-docker.sh -d xpu -t transformers 
```
Run `./run-docker.sh -h` for more options. By default, the current directory will be mounted to `/mnt` inside the container.

3. Run tests
Once inside the container, follow the below steps:

3.1 Set up IPEX XPU environment 
This step is only needed for XPU. Pls follow the installation guide [here](https://github.com/intel/intel-extension-for-pytorch). 

3.2 Copy the helper scripts over
```bash
cd transformers
# copy the helper scripts over 
cp /mnt/transformers/xpu/* .
```
3.3 Start testing
```bash
# run tests
./run-ut.sh xpu /mnt/test_results
```
This script will create a folder `test_results` in `/mnt`, which organizes the test results in excel files for later analysis. 

4. Analyze test results 
```bash 
python analyze_test_results.py --excel_dir /mnt/test_results --output_dir /mnt/test_stats
```
