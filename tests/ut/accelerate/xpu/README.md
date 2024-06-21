## How to run XPU-relevant tests in Accelerate?
1. Set up IPEX XPU environment
To install IPEX for XPU, pls follow the installation guide [here](https://github.com/intel/intel-extension-for-pytorch). 

2. Install XPU backend for Triton
```bash
pip install "git+https://github.com/intel/intel-xpu-backend-for-triton@d72e1e65c78b7b79405023e684338697579ea391#subdirectory=python"
```

3. Set up UT environment
```bash 
git clone https://github.com/huggingface/accelerate.git
cd accelerate
pip install -e ".[testing]"
pip install -e ".[test_trackers]"
pip install deepspeed pytest-excel openpyxl   
```

4. Start testing 
```bash 
RUN_SLOW=1 python -m pytest tests -sv --excelreport="ut_results.xlsx" 2>&1 | tee ut_results.log
```
