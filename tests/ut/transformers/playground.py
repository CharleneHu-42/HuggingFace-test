import pandas as pd 
from utils import print_ut_stats

cuda_file = "cuda_to_analyze.xlsx"
xpu_file = "xpu_to_analyze.xlsx"

cuda = pd.read_excel(cuda_file)
xpu = pd.read_excel(xpu_file)

print("++++++++++++++++++++++++CUDA++++++++++++++++++++++++")
print_ut_stats(cuda)
print("++++++++++++++++++++++++XPU++++++++++++++++++++++++")
print_ut_stats(xpu)
