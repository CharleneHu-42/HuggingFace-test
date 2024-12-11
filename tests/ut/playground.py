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

# -----------CASE 6: align cuda with xpu tests--------------
# cuda_excel = "cuda_trans_ut5.xlsx"
# xpu_excel = "FINAL2/updated_test_results.xlsx"

# cuda_df = pd.read_excel(cuda_excel)
# cuda = cuda_df[cuda_df["ignore"] != 1]
# xpu_df = pd.read_excel(xpu_excel)
# xpu = xpu_df[xpu_df["ignore"] != 1]

# compare_xpu_with_cuda_ut(cuda, xpu, "xpu_align_ut.xlsx")

# -----------CASE 7: print stats--------------

# xpu_excel = "FINAL2/updated_test_results.xlsx"
# xpu_df = pd.read_excel(xpu_excel)
# xpu = xpu_df[xpu_df["ignore"] != 1]

# save_skipped_stats_to_excel(xpu, f"xpu_trans_skipped_final.xlsx")
# save_failed_stats_to_excel(xpu, f"xpu_trans_failed_final.xlsx")

# print_ut_stats(xpu)

# -----------CASE 8: save cuda also--------------
# cuda_df = "cuda_trans_ut5.xlsx"
# save_skipped_cases_to_txt(cuda_df, "cuda_also_skipped.txt")
# save_failed_cases_to_txt(cuda_df, "cuda_also_failed.txt")

# -----------CASE 9: update UT--------------
# xpu_excel = "FINAL/updated_test_results.xlsx"
# new_excel = "all_sdpa.xlsx"

# update_ut_with_new_excel(xpu_excel, new_excel, "final_xpu_trans.xlsx")