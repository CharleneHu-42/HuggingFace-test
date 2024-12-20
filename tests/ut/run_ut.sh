#!/bin/bash

device="$1"
target="$2"
dry_run="$3"

result_dir="/mnt/${target}/raw_ut_result"
mkdir -p $result_dir
report="${result_dir}/ut.xlsx"


collect_test_count() {
    total=0
    for file in "$result_dir"/*.txt; do
        if [ -f "$file" ]; then
            last_line=$(tail -n 1 "$file")
            first_number=$(echo "$last_line" | grep -oE '^[0-9]+')
            if [ -n "$first_number" ]; then
                total=$((total + first_number))
            fi
        fi 
    done 
    echo "Total number of tests collected: $total"
}


run_test_folder() {
    local folder=$1
    local save_name=$2

    echo "+++++++++run test folder $folder++++++++++++++++"

	if [ "$folder" = "tests" ]; then 
		run_target=$folder 
	else
		run_target=tests/"$folder"
	fi 

    if [ "$dry_run" = "1" ]; then
        pytest $run_target --collectonly -q 2>&1 | tee "${result_dir}/${save_name}_collected.txt"
    else
        pytest $run_target -sv --excelreport="${result_dir}/${save_name}.xlsx" --timeout=600
    fi

}


run_transformers_ut() {
    local folder=$1
    local save_name=$2

	# only skip tests that are cpu-only, tpu-only, npu-only, sagemaker-only and tf-only.
	# tests that are not xpu-relevant, e.g. apex, torch.fx, will be filtered during analysis
	NOT_RUN_MARKERS="not (not_device_test)"
	NOT_RUN_KEYWORDS="not (tpu or npu or tf or ModelOnTheFlyConversionTester or SigOpt or TrainerHyperParameterRayIntegrationTest or TrainerHyperParameterWandbIntegrationTest or TestTrainerDistributedNeuronCore or TestTrainerDistributedNPU)"

	run_target=tests/"$folder"

    if [ "$dry_run" = "1" ]; then
        pytest $run_target -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --collectonly -q 2>&1 | tee "${result_dir}/${save_name}_collected.txt"
    else
        pytest $run_target -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${result_dir}/${save_name}.xlsx" --timeout=600
    fi

}


if [ "$target" = "transformers" ]; then

	export RUN_SLOW=1
	export TRANSFORMERS_TEST_DEVICE="${device}"
	export TRANSFORMERS_TEST_DEVICE_SPEC="spec_${device}.py"
	export RUN_PT_TF_CROSS_TESTS="False"
	export RUN_PT_FLAX_CROSS_TESTS="False"
	export WANDB_DISABLED="true"

	cp /mnt/spec_${device}.py .

	echo "+++++++++run single files++++++++++++++++"
	run_transformers_ut "*.py" "single_files"

	for folder in benchmark extended fsdp generation peft_integration trainer pipelines deepspeed
	do
		echo "+++++++++run test folder $folder ++++++++++++++++"
		run_transformers_ut "$folder" "$folder"
	done 

	for folder in autoawq bnb quanto_integration
	do 
		echo "+++++++++run test folder quantization/$folder ++++++++++++++++"
		run_transformers_ut "quantization/$folder" "$folder"
	done

	for folder in $(find tests/pipelines -mindepth 1 -maxdepth 1 -type d)
	do
		echo "+++++++++run test folder models/$x* ++++++++++++++++"
		run_transformers_ut "models/${folder}" "models_${folder}"
	done

	if [ "$dry_run" = "1" ]; then 
		collect_test_count
	fi

elif [ "$target" = "accelerate" ] || [ "$target" = "peft" ] || [ "$target" = "optimum-quanto" ] || [ "$target" = "trl" ]; then

	if [ "$target" = "trl" ]; then
		if [ "$device" = "cuda" ]; then 
			export CUDA_VISIBLE_DEVICES=2,3
		elif [ "$device" = "xpu" ]; then 
			export ZE_AFFINITY_MASK=2,3
		fi
	fi 

	run_test_folder "tests" "all_ut"
	
elif [ "$target" = "diffusers" ]; then

	export RUN_SLOW=1
    export DIFFUSERS_TEST_DEVICE="${device}"
    export DIFFUSERS_TEST_DEVICE_SPEC="spec_${device}.py"
    
    for folder in lora models others quantization schedulers
    do
		echo "+++++++++run test folder $folder ++++++++++++++++"
		run_test_folder "$folder" "$folder"
    done 

	for file in $(find tests/single_file -type f -name "*.py")
	do
		echo "+++++++++run test folder single_file/$folder ++++++++++++++++"
		run_test_folder "$file" "$(basename "$file" .py)"
	done

    for folder in $(find tests/pipelines -mindepth 1 -maxdepth 1 -type d)
	do 
		echo "+++++++++run test folder pipelines/$folder ++++++++++++++++"
		run_test_folder "$folder" "$(basename "$folder")"
	done 

    if [ "$dry_run" = "1" ]; then 
      collect_test_count
    fi 
fi