import pandas as pd
from utils import *
import shutil
import os

# =======================================================================================================
#                               Workflow recipe for transformers&diffusers UT
# =======================================================================================================

# target_lib = "diffusers"
# device = "xpu"
# result_path = os.path.join(target_lib, "test_result")
# raw_result_path = os.path.join(target_lib, "raw_ut_result")
# ignore_path = os.path.join(target_lib, "cases_to_ignore")

# ==============================================================
#       PHASE 1: Run UT on XPU and gather all UT results
# ==============================================================

# ----STEP 0: run UT----
# fist, dry run to detect any anomaly
# ./run-ut.sh xpu <target_lib> 1
# then run the whole test suite
# ./run-ut.sh xpu <target_lib>


# ----STEP 1: validate UT Results----
# check whether all ut cases are finished
# validate_ut_run(target_lib)


# ----STEP 2: manually check and rerun subsets if needed----
# manually check the validation result. 
# If the real ut number doesn't match the real ut number, this is either due to core-dump or hang.  
# in this case, you need to manually find out the test file or test cases that cause the issue.
# sort the test cases in a seperate txt file, and run them one by one, e.g.  
# save_txt_cases_to_bash(f"{target_lib}/need_rerun.txt", target_lib, f"{target_lib}/need_rerun.sh")

# ==============================================================
#       PHASE 2: Rough analyze, rerun and update UT results
# ==============================================================

# ----STEP 3: merge and get initial merged test results----
# consolidate_and_get_stats(raw_result_path, f"{device}_{target_lib}_ut.xlsx", f"{target_lib}/RERUN")

# ----STEP 4: manually check the skip and fail statistics and rerun test subsets if needed----
# excel_file = os.path.join(raw_result_path, f"{device}_{target_lib}_ut.xlsx")
#
# df = pd.read_excel(excel_file)
#
# message_list = [
#     "ModuleNotFoundError: No module named 'mpi4py'"
# ]
# rerun = df[df["message"].isin(message_list)]
#
# save_df_cases_to_bash(rerun, "xpu_rerun.sh")


# ----STEP 5: merge results to the original excel file after rerun----
# update_ut_result_after_rerun("RERUN", f"{raw_result_path}/{device}_{target_lib}_ut.xlsx", f"{raw_result_path}/{device}_{target_lib}_ut2.xlsx")


# ----STEP 6: manually check the statistic results, add a column `need rerun`, mark 1 if needed----
# excel_file = os.path.join(raw_result_path, f"{device}_{target_lib}_ut2.xlsx")
# df = pd.read_excel(excel_file)
#
# rerun = df[df["need rerun"] == 1.0] # manually marked as `need rerun` for abnormal failed messages like OOM or ccl error etc.
# save_df_cases_to_bash(rerun, "xpu_rerun2.sh")


# ----STEP 7: merge results to the original excel file after rerun----
# update_ut_result_after_rerun("RERUN", f"{raw_result_path}/{device}_{target_lib}_ut.xlsx", f"{raw_result_path}/{device}_{target_lib}_ut2.xlsx")


# ==============================================================
#       PHASE 3: Run UT on CUDA and calibrate with XPU results
# ==============================================================

# ----STEP 8: repeat step 0-7 on CUDA device----
# once finished, you should have 2 excel files for XPU and CUDA respectively.


# ----STEP 9: validate xpu ut results with cuda ut results----
# cuda_excel = os.path.join(raw_result_path, f"cuda_{target_lib}_ut.xlsx")
# xpu_excel = os.path.join(raw_result_path, f"xpu_{target_lib}_ut.xlsx")

# cuda_df = pd.read_excel(cuda_excel)
# xpu_df = pd.read_excel(xpu_excel)

# compare_xpu_with_cuda_ut(cuda_df, xpu_df, target_lib)

# ----STEP 10: manually check the ut diff, add a column `ignore` and mark if needed----
# if the xpu total ut number differs from cuda, you will need to manually compare them using vscode's compare selected function.
# depending on the results, you may need to mark the duplicated cases with 1 in `ignore` column to make test cases on XPU align with that on CUDA


# ----STEP 11: gather final UT results----
# xpu_excel = f"{raw_result_path}/aligned_xpu_ut.xlsx"
# cuda_excel = f"{raw_result_path}/cuda_{target_lib}_ut.xlsx"
# xpu_df = pd.read_excel(xpu_excel)
# cuda_df = pd.read_excel(cuda_excel)
# xpu_df = xpu_df[xpu_df["align with cuda"] == 1]
# # cuda_df = cuda_df[cuda_df["ignore"] != 1]

# xpu = xpu_df[RELEVANT_COLS]
# cuda = cuda_df[RELEVANT_COLS]

# os.makedirs(result_path, exist_ok=False)

# xpu.to_excel(os.path.join(result_path, f"clean_xpu_{target_lib}_ut.xlsx"), index=False)

# cuda.to_excel(os.path.join(result_path, f"clean_cuda_{target_lib}_ut.xlsx"), index=False)

# save_cases_to_txt(xpu, os.path.join(result_path, "all_cases_xpu.txt"))
# save_cases_to_txt(cuda, os.path.join(result_path, "all_cases_cuda.txt"))


# ==============================================================
#       PHASE 4: Gather report statistics
# ==============================================================

# ----STEP 12: save cuda failed and skipped cases----
# cuda_df = os.path.join(result_path, f"clean_cuda_{target_lib}_ut.xlsx")
# save_skipped_cases_to_txt(cuda_df, os.path.join(result_path, "cuda_skipped.txt"))
# save_failed_cases_to_txt(cuda_df, os.path.join(result_path, "cuda_failed.txt"))

# # move these 2 files to `cases_to_ignore` folder
# cuda_skipped_ignore = os.path.join(ignore_path, "cuda_skipped.txt")
# cuda_failed_ignore = os.path.join(ignore_path, "cuda_failed.txt")
# if os.path.exists(cuda_skipped_ignore):
#     os.remove(cuda_skipped_ignore)

# if os.path.exists(cuda_failed_ignore):
#     os.remove(cuda_failed_ignore)

# shutil.move(os.path.join(result_path, "cuda_skipped.txt"), f"{target_lib}/cases_to_ignore")
# shutil.move(os.path.join(result_path, "cuda_failed.txt"), f"{target_lib}/cases_to_ignore")


# ----STEP 13: mark cases with existing knowledge for statistics----
# xpu_file = os.path.join(result_path, f"clean_xpu_{target_lib}_ut.xlsx")
# cuda_file = os.path.join(result_path, f"clean_cuda_{target_lib}_ut.xlsx")

# xpu = pd.read_excel(xpu_file)
# cuda = pd.read_excel(cuda_file)

# mark_ut_results_with_ignore_cases(xpu_file, ignore_path, result_path, f"final_xpu_{target_lib}_ut.xlsx")
# mark_ut_results_with_ignore_cases(cuda_file, ignore_path, result_path, f"final_cuda_{target_lib}_ut.xlsx")

# # ----STEP 14: gather final statistics for both CUDA and XPU----

# xpu_file = os.path.join(result_path, f"final_xpu_{target_lib}_ut.xlsx")
# cuda_file = os.path.join(result_path, f"final_cuda_{target_lib}_ut.xlsx")

# create_final_report(result_path, cuda_file, xpu_file)


# ==============================================================
#       PHASE 5: In-depth analyze, debug and update statistics
# ==============================================================

# ----STEP 15: manually categorize, analyze and debug----
# you might need to rerun some cases depending on the statistics, e.g.
# the skipped cases between cuda and xpu don't align with each other
# you need to update the test result and then iterate STEP 13 to STEP 14.


# =======================================================================================================
#                              Workflow recipe for other libraries
# =======================================================================================================

# target_lib = "trl"
# device = "cuda"
# result_path = os.path.join(target_lib, "test_result")
# raw_result_path = os.path.join(target_lib, "raw_ut_result")
# ignore_path = os.path.join(target_lib, "cases_to_ignore")

# ==============================================================
#       PHASE 1: Run UT on XPU and gather all UT results
# ==============================================================

# ----STEP 0: run UT----
# fist, dry run to detect any anomaly
# ./run-ut.sh xpu <target_lib> 1
# then run the whole test suite
# ./run-ut.sh xpu <target_lib>


# ----STEP 1: validate UT Results----
# validate_ut_run(target_lib)


# ==============================================================
#       PHASE 2: Rough analyze, rerun and update UT results
# ==============================================================

# ----STEP 2: consolidate and get initial test results----
# consolidate_and_get_stats(raw_result_path, f"{device}_{target_lib}_ut.xlsx")


# ----STEP 3: manually check the skip and fail statistics and rerun test subsets if needed----
# you can manually modify the rerun test result in excel sheet for small libraries like accelerate, trl


# ==============================================================
#       PHASE 3: Run UT on CUDA and calibrate with XPU results
# ==============================================================

# ----STEP 4: repeat step 0-3 on CUDA device----
# once finished, you should have 2 excel files for XPU and CUDA respectively.


# ----STEP 5: validate xpu ut results with cuda ut results----
# cuda_excel = os.path.join(target_lib, f"cuda_{target_lib}_ut.xlsx")
# xpu_excel = os.path.join(target_lib, f"xpu_{target_lib}_ut.xlsx")

# cuda_df = pd.read_excel(cuda_excel)
# xpu_df = pd.read_excel(xpu_excel)

# compare_xpu_with_cuda_ut(cuda_df, xpu_df, target_lib)


# ----STEP 6: manually check the ut diff, add a column `ignore` and mark if needed----
# if the xpu total ut number differs from cuda, you will need to manually compare them using vscode's compare selected function.
# mark the duplicated cases with 1 in `ignore` column to make test cases on XPU align with that on CUDA


# ----STEP 7: gather final UT results----

# xpu_excel = os.path.join(target_lib, f"xpu_{target_lib}_ut.xlsx")
# cuda_excel = os.path.join(target_lib, f"cuda_{target_lib}_ut.xlsx")
# xpu_df = pd.read_excel(xpu_excel)
# cuda_df = pd.read_excel(cuda_excel)
# # xpu_df = xpu_df[xpu_df["ignore"] != 1]
# # cuda_df = cuda_df[cuda_df["ignore"] != 1] # depending on STEP6, you may not need to manually mark cuda file

# xpu = xpu_df[RELEVANT_COLS]
# cuda = cuda_df[RELEVANT_COLS]
# os.makedirs(result_path, exist_ok=False)

# xpu.to_excel(os.path.join(result_path, f"clean_xpu_{target_lib}_ut.xlsx"), index=False)
# cuda.to_excel(os.path.join(result_path, f"clean_cuda_{target_lib}_ut.xlsx"), index=False)

# save_cases_to_txt(xpu, os.path.join(result_path, "all_cases_xpu.txt"))
# save_cases_to_txt(cuda, os.path.join(result_path, "all_cases_cuda.txt"))


# ==============================================================
#       PHASE 4: Gather report statistics
# ==============================================================

# ----STEP 8: save cuda failed and skipped cases----

# cuda_path = os.path.join(result_path, f"clean_cuda_{target_lib}_ut.xlsx")
# save_skipped_cases_to_txt(cuda_path, os.path.join(result_path, "cuda_skipped.txt"))
# save_failed_cases_to_txt(cuda_path, os.path.join(result_path, "cuda_failed.txt"))

# # move these 2 files to `cases_to_ignore` folder
# cuda_skipped_ignore = os.path.join(ignore_path, "cuda_skipped.txt")
# cuda_failed_ignore = os.path.join(ignore_path, "cuda_failed.txt")
# if os.path.exists(cuda_skipped_ignore):
#     os.remove(cuda_skipped_ignore)

# if os.path.exists(cuda_failed_ignore):
#     os.remove(cuda_failed_ignore)

# shutil.move(os.path.join(result_path, "cuda_skipped.txt"), f"{target_lib}/cases_to_ignore")
# shutil.move(os.path.join(result_path, "cuda_failed.txt"), f"{target_lib}/cases_to_ignore")


## ----STEP 9: mark cases with existing knowledge for statistics----

# xpu_file = os.path.join(result_path, f"clean_xpu_{target_lib}_ut.xlsx")
# cuda_file = os.path.join(result_path, f"clean_cuda_{target_lib}_ut.xlsx")

# xpu = pd.read_excel(xpu_file)
# cuda = pd.read_excel(cuda_file)

# mark_ut_results_with_ignore_cases(xpu_file, ignore_path, result_path, f"final_xpu_{target_lib}_ut.xlsx")
# mark_ut_results_with_ignore_cases(cuda_file, ignore_path, result_path, f"final_cuda_{target_lib}_ut.xlsx")


# ----STEP 10: gather final statistics for both CUDA and XPU----

# xpu_file = os.path.join(result_path, f"final_xpu_{target_lib}_ut.xlsx")
# cuda_file = os.path.join(result_path, f"final_cuda_{target_lib}_ut.xlsx")

# create_final_report(result_path, cuda_file, xpu_file)


# ==============================================================
#       PHASE 5: In-depth analyze, debug and update statistics
# ==============================================================

# ----STEP 11: manually analyze to_*.xlsx files, categorize and update statistics----
# you need to go through the tests one by one and categorize them. Then update the report statistics.
#

# ----STEP 12: reorder txt test case files for upstreaming----
# file1 = os.path.join(ignore_path, "cuda_shouldnot_only.txt")
# file2 = os.path.join(ignore_path, "xpu_missing_thirdPartyLib.txt")
# file2 = os.path.join(ignore_path, "xpu_missing_quantization.txt")
# reorder_txt_cases(file1)
# reorder_txt_cases(file2)
# reorder_txt_cases(file3)
