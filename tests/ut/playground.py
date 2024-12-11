import pandas as pd
from utils import *

# Below is a typical workflow for transformers' UT.

# -----------STEP 0: run transformer's UT--------------
# ./run-ut.sh xpu transformers 1
# ./run-ut.sh xpu transformers 

# -----------STEP 1: validate transformers' UT Results--------------
# validate_ut_run("transformers")
   

# -----------STEP 2: manually check and rerun subsets if needed--------------
# manually check the validation result.
# if the real test numbers don't match with the actual test numbers, you need to manually adapt the `run_ut.sh` and rerun. 
# Sometimes, you will need to iterate on this step for several times in order to get all results completed

 
# -----------STEP 3: merge transformers' UT results and get initial merged test results--------------
# merge_excels_and_get_stats("transformers/test_results", "xpu_trans_ut_merged.xlsx")


# -----------STEP 4: manually check the skip and fail statistics and rerun test subsets if needed--------------
# excel_file = "xpu_trans_ut_merged.xlsx"
#
# df = pd.read_excel(excel_file)
#
# message_list = [
#     "ModuleNotFoundError: No module named 'mpi4py'"
# ]
# rerun = df[df["message"].isin(message_list)]
#
# save_cases_to_bash(rerun, "xpu_rerun.sh")


# -----------STEP 5: merge results to the original excel file after rerun--------------
# update_ut_result_after_rerun("RERUN", "xpu_trans_ut_merged.xlsx", "xpu_trans_ut_merged2.xlsx")
 

# -----------STEP 6: manually check the statistic results, add a column `need rerun`, mark 1 if need to rerun--------------
# excel_file = "xpu_trans_ut_merged2.xlsx"
# df = pd.read_excel(excel_file)
#
# rerun = df[df["need rerun"] == 1.0] # manually marked as `need rerun`
# save_cases_to_bash(rerun, "xpu_rerun2.sh")


# -----------STEP 7: merge results to the original excel file after rerun--------------
# update_ut_result_after_rerun("RERUN", "xpu_trans_ut_merged2.xlsx", "xpu_trans_ut_merged3.xlsx")


# -----------STEP 8: repeat step 0-7 on CUDA device--------------
# once finished, you should have 2 excel files for XPU and CUDA respectively.


# -----------STEP 9: validate xpu ut results with cuda ut results--------------
# cuda_excel = "cuda_trans_ut_merged3.xlsx"
# xpu_excel = "xpu_trans_ut_merged3.xlsx"
#
# cuda_df = pd.read_excel(cuda_excel)
# xpu_df = pd.read_excel(xpu_excel)
#
# compare_xpu_with_cuda_ut(cuda, xpu, "xpu_trans_ut_merged_aligned.xlsx")


# -----------STEP 10: manually check the ut diff, add a column `ignore` and mark if needed--------------
# if the xpu total ut number differs from cuda, you will need to manually compare them using vscode's compare selected function.
# mark the duplicated cases with 1 in `ignore` column to make test cases on XPU align with that on CUDA 


# -----------STEP 11: gather final UT results and statistics--------------
# xpu_excel = "xpu_trans_ut_merged_aligned.xlsx"
# cuda_excel = "cuda_trans_ut_merged3.xlsx"
# xpu_df = pd.read_excel(xpu_excel)
# cuda_df = pd.read_excel(cuda_excel)
# xpu = xpu_df[xpu_df["ignore"] != 1]
# cuda = cuda_df[cuda_df["ignore"] != 1]
# 
# result_dir = "transformers/FINAL"
# os.makedirs(result_dir, exist_ok=False)
# 
# xpu.to_excel(os.path.join(result_dir, "updated_xpu_trans_ut.xlsx"), index=False)
# cuda.to_excel(os.path.join(result_dir, "updated_cuda_trans_ut.xlsx"), index=False)
# save_skipped_stats_to_excel(xpu, os.path.join(result_dir, "updated_xpu_trans_skipped.xlsx"))
# save_failed_stats_to_excel(xpu, os.path.join(result_dir, "updated_xpu_trans_failed.xlsx"))
# save_skipped_stats_to_excel(cuda, os.path.join(result_dir,"updated_cuda_trans_skipped.xlsx"))
# save_failed_stats_to_excel(cuda, os.path.join(result_dir, "updated_cuda_trans_failed.xlsx"))
#
# print_ut_stats(xpu)
# print_ut_stats(cuda)


# -----------STEP 12: save cuda failed and skipped cases--------------
# cuda_df = os.path.join(result_dir,"updated_cuda_trans_ut.xlsx")
# save_skipped_cases_to_txt(cuda_df, os.path.join(result_dir, "cuda_skipped.txt"))
# save_failed_cases_to_txt(cuda_df, os.path.join(result_dir, "cuda_failed.txt"))


# -----------STEP 13: mark cases with existing knowledge for later analysis--------------
# first copy the `cuda_also_skipped.txt` and `cuda_also_failed.txt` to the `cases_to_ignore` folder
# xpu_file = os.path.join(result_dir, "updated_xpu_trans_ut.xlsx")
# ignore_path = "transformers/cases_to_ignore"
# update_ut_results_with_ignore_cases(xpu_file, ignore_path, result_dir)


# -----------STEP 14: gather final statistics for both CUDA and XPU--------------
# use `summarize.py` to print the results for report 


# -----------STEP 15: manually categorize, analyze and debug--------------