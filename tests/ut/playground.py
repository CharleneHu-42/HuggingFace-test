
import pandas as pd 
from utils import *

cuda_file = "cuda_ut.xlsx"
xpu_file = "xpu_ut.xlsx"

cuda = pd.read_excel(cuda_file)
xpu = pd.read_excel(xpu_file)

print("++++++++++++++++++++++++CUDA++++++++++++++++++++++++")
print_ut_stats(cuda)
print("++++++++++++++++++++++++XPU++++++++++++++++++++++++")
print_ut_stats(xpu)

# save_skipped_stats_to_excel(cuda, "cuda_skipped.xlsx")
# save_failed_stats_to_excel(cuda, "cuda_failed.xlsx")

# save_cases_to_txt(cuda, "cuda_all.txt")
# save_cases_to_txt(xpu, "xpu_all.txt")
