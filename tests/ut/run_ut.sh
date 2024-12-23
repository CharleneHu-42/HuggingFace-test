#!/bin/bash

device="$1"
target="$2"
dry_run="$3"

result_dir="/mnt/${target}/raw_ut_result"
mkdir -p $result_dir
report="${result_dir}/ut.xlsx"

export RUN_SLOW=1

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
	local is_transformers=$3

	if [ "$is_transformers" = "1" ]; then
		NOT_RUN_MARKERS="not (not_device_test)"
		NOT_RUN_KEYWORDS="not (tpu or npu or tf or ModelOnTheFlyConversionTester or SigOpt or TrainerHyperParameterRayIntegrationTest or TrainerHyperParameterWandbIntegrationTest or TestTrainerDistributedNeuronCore or TestTrainerDistributedNPU)"
		IGNORE1="tests/sagemaker"
		IGNORE2="tests/bettertransformer"
	else
		NOT_RUN_MARKERS=""
		NOT_RUN_KEYWORDS=""
		IGNORE1=""
		IGNORE2=""
	fi 

	if [ "$target" = "transformers" ]; then
		NOT_RUN_MARKERS="not (not_device_test)"
		NOT_RUN_KEYWORDS="not (tpu or npu or tf or ModelOnTheFlyConversionTester or SigOpt or TrainerHyperParameterRayIntegrationTest or TrainerHyperParameterWandbIntegrationTest or TestTrainerDistributedNeuronCore or TestTrainerDistributedNPU)"
		IGNORE1="tests/sagemaker"
		IGNORE2="tests/bettertransformer"
	elif [ "$target" = "diffusers" ]; then
		NOT_RUN_MARKERS=""
		NOT_RUN_KEYWORDS="not animatediff"
		IGNORE1=""
		IGNORE2="tests/single_file/test_stable_diffusion_img2img_single_file.py"
	else
		NOT_RUN_MARKERS=""
		NOT_RUN_KEYWORDS=""
		IGNORE1=""
		IGNORE2=""
	fi 

    if [ "$dry_run" = "1" ]; then
        pytest $folder -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore "${IGNORE1}" --ignore "${IGNORE2}" --collectonly -q 2>&1 | tee "${result_dir}/${save_name}_collected.txt"
    else
        pytest $folder -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore "${IGNORE1}" --ignore "${IGNORE2}" -sv --excelreport="${result_dir}/${save_name}.xlsx" --timeout=600 2>&1 | tee "${result_dir}/${save_name}.log"
    fi
}


if [ "$target" = "transformers" ]; then

	cp /mnt/spec_${device}.py .

	export TRANSFORMERS_TEST_DEVICE="${device}"
	export TRANSFORMERS_TEST_DEVICE_SPEC="spec_${device}.py"
	export RUN_PT_TF_CROSS_TESTS="False"
	export RUN_PT_FLAX_CROSS_TESTS="False"
	export WANDB_DISABLED="true"
	
	echo "+++++++++run single files++++++++++++++++"
	run_test_folder "tests" "single_files"

	for folder in benchmark extended fsdp generation peft_integration trainer pipelines deepspeed
	do
		echo "+++++++++run test folder $folder ++++++++++++++++"
		run_test_folder "tests/$folder" "$folder"
	done 

	for folder in autoawq bnb quanto_integration
	do 
		echo "+++++++++run test folder quantization/$folder ++++++++++++++++"
		run_test_folder "tests/quantization/$folder" "$folder"
	done

	for x in a b c d e f g h i j k l m n o p q r s t u v w x y z
	do
		echo "+++++++++run test folder models/$x* ++++++++++++++++"
		run_test_folder "tests/models/$x*" "models_$x"
	done

	if [ "$dry_run" = "1" ]; then 
		collect_test_count
	fi
elif [ "$target" = "diffusers" ]; then
	
	cp /mnt/spec_${device}.py .

    export DIFFUSERS_TEST_DEVICE="${device}"
    export DIFFUSERS_TEST_DEVICE_SPEC="spec_${device}.py"
    
    for folder in lora models others quantization schedulers
    do
		echo "+++++++++run test folder $folder ++++++++++++++++"
		run_test_folder "tests/$folder" "$folder"
    done 

	for file in $(find tests/single_file -type f -name "*.py")
	do
		if [ "$(basename "$file")" = "__init__.py" ] || [ "$(basename "$file")" = "single_file_testing_utils.py" ]; then
			echo "----skip $file----"
		else
			run_test_folder "tests/single_file/$(basename "$file")" "$(basename "$file" .py)"
			fi 
		fi
	done

    for x in a b c d f h i k l m p s t u w
	do
		echo "+++++++++run test folder pipelines/$x* ++++++++++++++++"
		run_test_folder "tests/pipelines/$x*" "${x}_pipeline"
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

	run_test_folder "tests" "all_cases" 
fi