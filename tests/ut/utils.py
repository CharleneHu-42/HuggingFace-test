import glob
import os
import pandas as pd
import re
from datetime import datetime

RELEVANT_COLS = [
    "file_name",
    "suite_name",
    "test_name",
    "result",
    "message",
]


def replace_unittests(df):
    for i, v in df["file_name"].items():
        if "/unittest/" in str(v):
            suite_name = df.loc[i, "suite_name"]
            new_df = (
                df.groupby(["suite_name", "file_name"], as_index=False)
                .size()
                .sort_values(["size"], ascending=False)
            )
            new_value = new_df[new_df["suite_name"] == suite_name]["file_name"].iloc[0]
            df.loc[i, "file_name"] = new_value
    return df


def read_txt_to_list(ignore_file):
    cases_list = []
    if os.path.isfile(ignore_file):
        with open(ignore_file, "r") as f:
            cases_list.append([line.strip() for line in f.readlines()])

    cases = [case for cases in cases_list for case in cases]
    return cases


def save_list_to_txt(list, output_file):
    list = sorted(list)
    with open(output_file, "w") as file:
        for case in list:
            file.write(case + "\n")


def reorder_txt_cases(file):
    cases = read_txt_to_list(file)
    save_list_to_txt(cases, file)


def extract_short_cases(in_txt_file, out_txt_file):
    case_list = []
    with open(in_txt_file, "r") as f:
        case_list.append(
            [line.strip() for line in f.readlines() if line.startswith("tests/")]
        )
    cases = [case for cases in case_list for case in cases]

    final_cases = sorted(
        [f"{case.split('::')[1]}::{case.split('::')[2]}" for case in cases]
    )
    save_list_to_txt(final_cases, out_txt_file)

    return final_cases


def save_df_cases_to_bash(df, target_lib, output_file):
    df = df.sort_values(by=["suite_name", "test_name"])

    rerun_dir_name = f"{target_lib}/RERUN" 
    os.makedirs(rerun_dir_name, exist_ok=False)

    cases = []

    for _, row in df.iterrows():
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        cases.append(
            f"pytest -rA tests -k '{suite_name} and {test_name}' --excelreport /mnt/{rerun_dir_name}/{suite_name+test_name}.xlsx"
        )

    save_list_to_txt(cases, output_file)


def save_txt_cases_to_bash(txt_file, target_lib, output_file):
    case_list = read_txt_to_list(txt_file)
    
    rerun_dir_name = f"{target_lib}/RERUN" 
    os.makedirs(rerun_dir_name, exist_ok=False)
    rerun_command = []
    
    for case in case_list:
        suite_name = case.split("::")[1]
        test_name = case.split("::")[2]
        rerun_command.append(
            f"pytest -rA {case} --excelreport /mnt/{rerun_dir_name}/{suite_name+test_name}.xlsx"
        )
    
    save_list_to_txt(rerun_command, output_file)
    

def save_cases_to_txt(df, output_file):
    df = df.sort_values(by=["suite_name", "test_name"])

    cases = []
    for _, row in df.iterrows():
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        cases.append(f"{suite_name}::{test_name}")

    save_list_to_txt(cases, output_file)


def save_cases_with_empty_messages(file_name):
    df = pd.read_excel(file_name)

    skipped = df[(df["result"] == "SKIPPED") | (df["result"] == "FAILED")]
    rerun = skipped[skipped["message"].isnull()]

    save_df_cases_to_bash(rerun, "rerun.sh")


def merge_excel_files_to_one(input_path, output_file):
    all_test_files = glob.glob(os.path.join(input_path, "*.xlsx"))

    df_merged = pd.concat(
        pd.read_excel(excel_file) for excel_file in all_test_files
    ).reset_index()
    df_merged.drop(columns=["index"], inplace=True)

    df_merged.to_excel(output_file, index=False)


def save_skipped_cases_to_txt(input_file, output_file):
    df = pd.read_excel(input_file)
    df_skipped = df[df["result"] == "SKIPPED"]

    save_cases_to_txt(df_skipped, output_file)


def save_failed_cases_to_txt(input_file, output_file):
    df = pd.read_excel(input_file)
    df_skipped = df[df["result"] == "FAILED"]

    save_cases_to_txt(df_skipped, output_file)


def save_skipped_stats_to_excel(df, output_file):
    skipped_df = df[df["result"] == "SKIPPED"]["message"].value_counts().reset_index()
    skipped_df.to_excel(output_file, index=False)


def save_failed_stats_to_excel(df, output_file):
    skipped_df = df[df["result"] == "FAILED"]["message"].value_counts().reset_index()
    skipped_df.to_excel(output_file, index=False)


def save_skipped_stats_to_excel_from_excel(input_file, output_file):
    df = pd.read_excel(input_file)
    save_skipped_stats_to_excel(df, output_file)


def save_failed_stats_to_excel_from_excel(input_file, output_file):
    df = pd.read_excel(input_file)
    save_failed_stats_to_excel(df, output_file)


def mark_cuda_failed_skipped_cases(xpu_file, cuda_file, output_file):
    xpu_df = pd.read_excel(xpu_file)
    cuda_df = pd.read_excel(cuda_file)

    xpu_df["cuda also"] = [0] * xpu_df.shape[0]

    pvc_skipped = xpu_df[xpu_df["result"] == "SKIPPED"]
    cuda_skipped = cuda_df[cuda_df["result"] == "SKIPPED"]

    def mark_same_with_cuda(cuda_df, xpu_df, xpu_df_all):
        for _, row in cuda_df.iterrows():
            suite_name = row["suite_name"]
            test_name = row["test_name"]
            target = xpu_df[
                (xpu_df["suite_name"] == suite_name)
                & (xpu_df["test_name"] == test_name)
            ]
            xpu_df_all.loc[target.index, "cuda also"] = 1

        return xpu_df_all

    xpu_df = mark_same_with_cuda(cuda_skipped, pvc_skipped, xpu_df)

    pvc_failed = xpu_df[xpu_df["result"] == "FAILED"]
    cuda_failed = cuda_df[cuda_df["result"] == "FAILED"]

    xpu_df = mark_same_with_cuda(cuda_failed, pvc_failed, xpu_df)

    xpu_df.to_excel(output_file, index=False)


def print_ut_stats(tests_df):
    result_stats = tests_df["result"].value_counts()
    pass_rate = result_stats["PASSED"] / sum(result_stats)
    print(f"=====UT PASS RATE=====\n{pass_rate}")
    print(f"=====TOTAL UT=====\n{tests_df.shape[0]}")
    print(f"=====DETAILS=====\n{result_stats}")


def remove_cases_duplicated(input_txt, output_txt):
    cases_list = read_txt_to_list(input_txt)
    cases_list = ["::".join(case.split("::")[-2:]) for case in cases_list]

    cased_unique = list(set(cases_list))
    cased_unique = sorted(cased_unique)

    save_list_to_txt(cased_unique, output_txt)


def merge_excel_files_to_df(excel_path):
    all_excel_files = glob.glob(os.path.join(excel_path, "*.xlsx"))

    df = pd.concat(
        pd.read_excel(excel_file) for excel_file in all_excel_files
    ).reset_index()
    df.drop(columns=["index"], inplace=True)

    return df


def update_ut_with_new_excel(ori_excel, new_excel, output_file):
    ori_df = pd.read_excel(ori_excel)
    new_df = pd.read_excel(new_excel)

    ori_df = ori_df.sort_values(by=["file_name", "suite_name", "test_name"])
    new_df = new_df.sort_values(by=["file_name", "suite_name", "test_name"])

    for _, row in new_df.iterrows():
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        result = row["result"]
        message = row["message"]

        sample = ori_df[
            (ori_df["suite_name"] == suite_name) & (ori_df["test_name"] == test_name)
        ]
        ori_df.loc[sample.index, "result"] = result
        ori_df.loc[sample.index, "message"] = message

    save_skipped_stats_to_excel(ori_df, f"{output_file.split('.')[0]}_skipped.xlsx")
    save_failed_stats_to_excel(ori_df, f"{output_file.split('.')[0]}_failed.xlsx")
    print_ut_stats(ori_df)

    ori_df.to_excel(output_file, index=False)


def update_ut_result_after_rerun(rerun_excel_path, ori_excel_file, output_file):
    rerun_df = merge_excel_files_to_df(rerun_excel_path)

    ori_df = pd.read_excel(ori_excel_file)

    ori_df = ori_df.sort_values(by=["file_name", "suite_name", "test_name"])
    rerun_df = rerun_df.sort_values(by=["file_name", "suite_name", "test_name"])

    for _, row in rerun_df.iterrows():
        file_name = row["file_name"]
        suite_name = row["suite_name"]
        test_name = row["test_name"]
        result = row["result"]
        message = row["message"]

        sample = ori_df[
            (ori_df["file_name"] == file_name)
            & (ori_df["suite_name"] == suite_name)
            & (ori_df["test_name"] == test_name)
        ]
        ori_df.loc[sample.index, "result"] = result
        ori_df.loc[sample.index, "message"] = message

    ori_df.to_excel(output_file, index=False)


def add_column_and_mark_with_case_list(df, new_column, case_list, value=None):
    if not value:
        df[new_column] = [0] * df.shape[0]
        value = 1
    else:
        df[new_column] = ["none"] * df.shape[0]

    for index, row in df.iterrows():
        suite_name = row["suite_name"]
        test_name = row["test_name"]

        if f"{suite_name}::{test_name}" in case_list:
            df.iloc[index, -1] = value
    return df


def consolidate_and_get_stats(excel_dir, out_file_name, rerun_folder=""):
    # read xpu ut excel files
    raw_df = merge_excel_files_to_df(excel_dir)

    if os.path.exists(rerun_folder):
        rerun_df = merge_excel_files_to_df(rerun_folder)
        tests_df = pd.concat([raw_df, rerun_df])

    tests_df = tests_df[RELEVANT_COLS]

    tests_df.to_excel(os.path.join(excel_dir, out_file_name), index=False)

    save_skipped_stats_to_excel(
        tests_df, os.path.join(excel_dir, f"{out_file_name.split('.')[0]}_skipped.xlsx")
    )
    save_failed_stats_to_excel(
        tests_df, os.path.join(excel_dir, f"{out_file_name.split('.')[0]}_failed.xlsx")
    )


def save_ut_results_to_txt(xpu_df, output_dir, name_prefix):
    passed = xpu_df[xpu_df["result"] == "PASSED"]
    failed = xpu_df[xpu_df["result"] == "FAILED"]
    skipped = xpu_df[xpu_df["result"] == "SKIPPED"]

    save_cases_to_txt(xpu_df, os.path.join(output_dir, f"{name_prefix}_all_cases.txt"))
    save_cases_to_txt(
        passed, os.path.join(output_dir, f"{name_prefix}_all_passed_cases.txt")
    )
    save_cases_to_txt(
        failed, os.path.join(output_dir, f"{name_prefix}_all_failed_cases.txt")
    )
    save_cases_to_txt(
        skipped, os.path.join(output_dir, f"{name_prefix}_all_skipped_cases.txt")
    )


def compare_xpu_with_cuda_ut(cuda_df, xpu_df, save_dir):
    xpu_df["align with cuda"] = [0] * xpu_df.shape[0]
    xpu_df = xpu_df.sort_values(by=["suite_name", "test_name"])
    cuda_df = cuda_df.sort_values(by=["suite_name", "test_name"])

    for index, row in xpu_df.iterrows():
        suite_name = row["suite_name"]
        test_name = row["test_name"]

        target = cuda_df[
            (cuda_df["suite_name"] == suite_name) & (cuda_df["test_name"] == test_name)
        ]
        if target.shape[0] > 0:
            xpu_df.loc[index, "align with cuda"] = 1

    if cuda_df.shape[0] != xpu_df.shape[0]:
        print(f"----------total ut numbers are different, pls double-check------------")
        save_cases_to_txt(cuda_df, os.path.join(save_dir, "all_cases_cuda.txt"))
        save_cases_to_txt(xpu_df, os.path.join(save_dir, "all_cases_xpu.txt"))
        xpu_df.to_excel(os.path.join(save_dir, "aligned_xpu_ut.xlsx"), index=False)
    else:
        print(f"----------PASSED------------")


def validate_ut_run(target_lib):
    pattern1 = r"(.*) tests"
    pattern2 = r"(.*) test"
    save_dir = os.path.join(os.path.dirname(__file__), target_lib, "raw_ut_result")

    if target_lib in ["transformers", "diffusers"]:
        txt_files_glob = sorted(glob.glob(os.path.join(save_dir, "*.txt")))

        rerun_cases = []
        total_num = 0
        for txt_file in txt_files_glob:
            txt_file_name = os.path.basename(txt_file).split(".")[0]
            excel_file_path = txt_file.replace("_collected", "").replace("txt", "xlsx")

            print(f"-------{txt_file_name}-------")

            if not os.path.exists(excel_file_path):
                rerun_cases.append(txt_file_name)
                continue

            with open(txt_file, "r") as file:
                lines = file.readlines()
                last_line = lines[-1] if lines else None
                match1 = re.match(pattern1, last_line)
                match2 = re.match(pattern2, last_line)
                if match1:
                    true_case_num = match1.group(1)
                elif match2:
                    true_case_num = match2.group(1)
                else:
                    true_case_num = "0"

            if "/" in true_case_num:
                true_case_num = true_case_num.split("/")[0]

            total_num = total_num + int(true_case_num)

            df = pd.read_excel(excel_file_path)

            real_case_num = df.shape[0]

            if int(true_case_num) != real_case_num:
                rerun_cases.append(txt_file_name)
                save_cases_to_txt(
                    df, os.path.join(save_dir, f"{txt_file_name}_real.txt")
                )
                extract_short_cases(
                    txt_file, os.path.join(save_dir, f"{txt_file_name}_true.txt")
                )
        print(f"\nThere are {total_num} test cases in total.")
        print(f"\n{rerun_cases} need double-check.")
    else:
        txt_file = os.path.join(
            save_dir,
            "all_cases_collected.txt",
        )
        excel_file_path = os.path.join(save_dir, "all_cases.xlsx")

        txt_file_name = os.path.basename(txt_file).split(".")[0]

        with open(txt_file, "r") as file:
            lines = file.readlines()
            last_line = lines[-1] if lines else None
            match1 = re.match(pattern1, last_line)
            match2 = re.match(pattern2, last_line)
            if match1:
                true_case_num = match1.group(1)
            elif match2:
                true_case_num = match2.group(1)
            else:
                true_case_num = "0"

        if "/" in true_case_num:
            true_case_num = true_case_num.split("/")[0]

        df = pd.read_excel(excel_file_path)
        real_case_num = df.shape[0]

        if int(true_case_num) != real_case_num:
            save_cases_to_txt(df, os.path.join(save_dir, f"{txt_file_name}_real.txt"))
            extract_short_cases(
                txt_file, os.path.join(save_dir, f"{txt_file_name}_true.txt")
            )
            print(f"\nNeed double-check.")
        else:
            print(f"\nPASSED.")


def mark_ut_results_with_ignore_cases(
    file_name, ignore_path, output_dir, output_file_name
):
    os.makedirs(output_dir, exist_ok=True)
    tests_df = pd.read_excel(file_name)

    ignore_files = glob.glob(os.path.join(ignore_path, "*.txt"))

    for file in ignore_files:
        if "cuda_failed" in file:
            cuda_failed = read_txt_to_list(file)
            tests_df = add_column_and_mark_with_case_list(
                tests_df, "cuda also failed?", cuda_failed
            )
        elif "cuda_skipped" in file:
            cuda_skipped = read_txt_to_list(file)
            tests_df = add_column_and_mark_with_case_list(
                tests_df, "cuda also skipped?", cuda_skipped
            )
        elif "cuda_should_only" in file:
            cuda_should = read_txt_to_list(file)
            tests_df = add_column_and_mark_with_case_list(
                tests_df, "cuda should only?", cuda_should
            )
        elif "cuda_shouldnot_only" in file:
            cuda_shouldnot = read_txt_to_list(file)
            tests_df = add_column_and_mark_with_case_list(
                tests_df, "cuda shouldnot only?", cuda_shouldnot
            )

    xpu_missing_files = glob.glob(os.path.join(ignore_path, "xpu_missing_*.txt"))

    if len(xpu_missing_files) >= 1:
        all_missing_files = [read_txt_to_list(file) for file in xpu_missing_files]
        xpu_missing_cases = [case for file in all_missing_files for case in file]
        tests_df = add_column_and_mark_with_case_list(
            tests_df, "xpu missing features?", xpu_missing_cases
        )

        for file in xpu_missing_files:
            value = file.split("/")[-1].split(".")[0].split("_")[-1]
            case_list = read_txt_to_list(file)

            tests_df = add_column_and_mark_with_case_list(
                tests_df, f"xpu missing {value}", case_list, value
            )

    tests_df.to_excel(os.path.join(output_dir, output_file_name), index=False)

    base_file_name = output_file_name.split(".")[0]
    save_skipped_stats_to_excel(
        tests_df, os.path.join(output_dir, f"{base_file_name}_skipped.xlsx")
    )
    save_failed_stats_to_excel(
        tests_df, os.path.join(output_dir, f"{base_file_name}_failed.xlsx")
    )


def create_final_report(output_dir, cuda_path, xpu_path):
    os.makedirs(output_dir, exist_ok=True)
    cuda_df = pd.read_excel(cuda_path)
    xpu_df = pd.read_excel(xpu_path)

    for col in cuda_df.columns:
        if col in [
            "cuda should only?",
            "cuda shouldnot only?",
            "xpu missing features?",
        ]:
            cuda_df = cuda_df[cuda_df[col] != 1]
            xpu_df = xpu_df[xpu_df[col] != 1]

    save_ut_results_to_txt(cuda_df, output_dir, "cuda")
    save_ut_results_to_txt(xpu_df, output_dir, "xpu")

    def print_ut_stats(tests_df):
        print("========Overview========")
        print(f"TOTAL: {tests_df.shape[0]}")

        passed_df = tests_df[tests_df["result"] == "PASSED"]
        print(f"PASSED: {passed_df.shape[0]}")

        failed_df = tests_df[(tests_df["result"] == "FAILED")]
        print(f"FAILED: {failed_df.shape[0]}")

        skipped_df = tests_df[(tests_df["result"] == "SKIPPED")]
        print(f"SKIPPED: {skipped_df.shape[0]}")

        return passed_df, failed_df, skipped_df

    print(f"+++++++++++++++++CUDA+++++++++++++++++")
    _, _, _ = print_ut_stats(cuda_df)
    print(f"+++++++++++++++++XPU+++++++++++++++++")
    _, failed_df, skipped_df = print_ut_stats(xpu_df)

    print("========FAILED========")
    cuda_also_fails = failed_df[failed_df["cuda also failed?"] == 1]
    print(f"Cuda also fails: {cuda_also_fails.shape[0]}")
    save_cases_to_txt(cuda_also_fails, os.path.join(output_dir, "cuda_also_fails.txt"))

    to_debug = failed_df[failed_df["cuda also failed?"] != 1]
    print(f"To debug: {to_debug.shape[0]}")
    to_debug[RELEVANT_COLS].to_excel(
        os.path.join(output_dir, "to_debug.xlsx"), index=False
    )

    print("========SKIPPED========")
    cuda_also_skipped = skipped_df[skipped_df["cuda also skipped?"] == 1]
    print(f"Cuda also skips: {cuda_also_skipped.shape[0]}")
    save_cases_to_txt(
        cuda_also_skipped, os.path.join(output_dir, "cuda_also_skips.txt")
    )

    other_skipped = skipped_df[skipped_df["cuda also skipped?"] != 1]
    xpu_missing_features = other_skipped[other_skipped["xpu missing features?"] == 1]
    print(f"XPU missing features: {xpu_missing_features.shape[0]}")

    to_investigate = other_skipped[other_skipped["xpu missing features?"] != 1]
    print(f"To investigate: {to_investigate.shape[0]}")
    to_investigate[RELEVANT_COLS].to_excel(
        os.path.join(output_dir, "to_investigate.xlsx"), index=False
    )
