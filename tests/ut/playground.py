import pandas as pd
from utils import *

# -----------CASE 1: merge transformers' UT Results--------------

# merge_excels_and_get_stats("RERUN", "trans_ut_merged.xlsx")

# -----------CASE 2: need to rerun test subset after shallow analysis--------------
# excel_file = "cuda_trans_ut3.xlsx"

# df = pd.read_excel(excel_file)
# message_list = [
#     "ModuleNotFoundError: No module named 'mpi4py'"
# ]

# rerun = df[df["message"].isin(message_list)]

# save_cases_to_bash(rerun, "cuda_rerun2.sh")

# -----------CASE 3: need to rerun test subset after deep analysis--------------

# excel_file = "xpu_ut.xlsx"
# df = pd.read_excel(excel_file)

# rerun = df[df["need rerun"] == 1.0] # manually marked as need rerun
# save_cases_to_bash(rerun, "xpu_rerun.sh")

# -----------CASE 4: need to rerun test subset after deep analysis--------------

# excel_file = "xpu_ut.xlsx"
# df = pd.read_excel(excel_file)

# rerun = df[df["need rerun"] == 1.0] # manually marked as need rerun
# save_cases_to_bash(rerun, "xpu_rerun.sh")

# -----------CASE 5: merge files after rerun--------------

# update_ut_result_after_rerun("RERUN", "cuda_trans_ut2.xlsx", "cuda_trans_ut3.xlsx")

