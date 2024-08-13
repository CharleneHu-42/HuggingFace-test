import glob
import os
import pandas as pd 


def export_rerun_cases(file_name):
    df = pd.read_excel(file_name)

    skipped = df[df["result"] == "SKIPPED"]
    rerun = skipped[skipped["message"].isnull()]

    cases  = []
    
    for index, row in rerun.iterrows():
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        cases.append(f"pytest -rA tests -k '{suite_name} and {test_name}' --excelreport RERUN/{suite_name[:10]+test_name[-15:]}.xlsx")
    
    with open("rerun.sh", "w") as file:
        for case in cases:
            file.write(case + "\n")
            

def merge_excel_results(excel_path, output_file_name):    
    
    all_test_files = glob.glob(os.path.join(excel_path, "*.xlsx"))

    rerun_df = pd.concat(
            pd.read_excel(excel_file) for excel_file in all_test_files
        ).reset_index()
    rerun_df.drop(columns=["index"], inplace=True)

    rerun_df.to_excel(output_file_name, index=False)
        
    
def save_skipped_cases_to_txt(input_file, output_file):
    
    df = pd.read_excel(input_file)
    df_skipped = df[df["result"] == "SKIPPED"]
    
    df_skipped = df_skipped.sort_values(by=["suite_name","test_name"])
    
    cases = []
    for _, row in df_skipped.iterrows():
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        cases.append(f"{suite_name}::{test_name}")
    
    with open(output_file, "w") as file:
        for case in cases:
            file.write(case + "\n")
        
        
    
def mark_same_with_cuda(cuda_df, xpu_df, xpu_df_all):
    
    for _, row in cuda_df.iterrows():
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        target = xpu_df[(xpu_df["suite_name"] == suite_name) & (xpu_df["test_name"] == test_name)] 
        xpu_df_all.loc[target.index, "same with cuda"]  = 1

    return xpu_df_all
    
    
def mark_cuda_failed_skipped_cases(xpu_file, cuda_file, output_dir):
    xpu_df = pd.read_excel(xpu_file)
    cuda_df = pd.read_excel(cuda_file)
    
    pvc_skipped = xpu_df[xpu_df["result"] == "SKIPPED"]
    cuda_skipped = cuda_df[cuda_df["result"] == "SKIPPED"]
    
    xpu_df = mark_same_with_cuda(cuda_skipped, pvc_skipped, xpu_df)

    pvc_failed = xpu_df[xpu_df["result"] == "FAILED"]
    cuda_failed = cuda_df[cuda_df["result"] == "FAILED"]
    
    xpu_df = mark_same_with_cuda(cuda_failed, pvc_failed, xpu_df)
    
    xpu_df.to_excel(os.path.join(output_dir, "new_raw_test_results.xlsx"), index=False)
    

    