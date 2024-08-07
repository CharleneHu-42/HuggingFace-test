import pandas as pd 
from utils import print_ut_stats

# cuda_file = "cuda_to_analyze.xlsx"
# xpu_file = "xpu_to_analyze.xlsx"

# cuda = pd.read_excel(cuda_file)
# xpu = pd.read_excel(xpu_file)

# print("++++++++++++++++++++++++CUDA++++++++++++++++++++++++")
# print_ut_stats(cuda)
# print("++++++++++++++++++++++++XPU++++++++++++++++++++++++")
# print_ut_stats(xpu)

file_name = "cuda_also_skipped.txt"
from utils import read_txt_to_list

case_list = read_txt_to_list(file_name)

import pdb; pdb.set_trace()
from utils import save_list_to_txt

save_list_to_txt(case_list, "cuda_also_failed.txt")