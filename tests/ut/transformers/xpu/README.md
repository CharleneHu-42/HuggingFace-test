## How to run XPU-relevant tests in Transformers?
1. Set up IPEX XPU environment
To install IPEX for XPU, pls follow the installation guide [here](https://github.com/intel/intel-extension-for-pytorch). 

2. Set up UT environment
```bash 
apt-get update & apt-get install tesseract-ocr ffmpeg espeak
git clone <the HuggingFace repository URL>
cd HuggingFace/tests/ut/transformers/xpu
pip install -r requirements.txt
```

3. Clone the transformers' repository
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e ".[testing]"
```

4. Start testing 
```bash 
# copy over the test helper scripts
export huggingface_repo=<your local path of the HuggingFace repository>
find $huggingface_repo/tests/ut/transformers/xpu -type f ! -name "*.md" -exec cp {} . \;
# run pytest
bash run_xpu_ut.sh test_results
```
This bash script will create 2 additional folders for test results:
- `reports` saves all test details in txt files, which is good for in-depth analysis   
- `test_results` organizes the test results in excel files, which is good for statistics and summary report 

5. Analyze test results 
```bash 
python analyze_test_results.py --excel_dir test_results --output_dir test_stats
```




