## How to run XPU-relevant tests in Transformers?
1. Clone the transformers' repository
```bash 
git clone https://github.com/huggingface/transformers.git
cd transformers
```

2. Copy all files except the markdown file to the transformers' root directory
```bash 
export huggingface_repo=<your local path of the HuggingFace repository>
find $huggingface_repo/tests/ut/transformers/xpu -type f ! -name "*.md" -exec cp {} . \;
```

3. Create a new test environment 
```bash
conda create -n pytest-xpu python=3.9
conda activate pytest-xpu
pip install -r requirements.txt
pip install -e ".[testing]"
```
To install IPEX for XPU, you need to follow the instruction guide [here](https://github.com/intel/intel-extension-for-pytorch). 

4. Apply the xpu test marker patch and the not-device-tests patch
```bash 
cp $huggingface_repo/hot-patches/0001-add-more-not_device-tests.patch $huggingface_repo/hot-patches/0001-add-new-test-markers.patch .
git apply 0001-add-new-test-markers.patch
git apply 0001-add-more-not_device-tests.patch
```

5. Start Testing
```bash
bash run_xpu_ut.sh tests_results
```
This bash script will create 2 additional folders for test results:
- `reports` saves all test details in txt files, which is good for in-depth analysis   
- `tests_results` organizes the test results in excel files, which is good for statistics and summary report   

6. Analyze test results 
```bash 
python analyze_test_results.py --excel_dir test_results --output_dir test_stats
```




