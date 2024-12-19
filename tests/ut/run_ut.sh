device="$1"
target="$2"
dry_run="$3"

excel_dir=$(dirname "$report")


if [[ $target == "transformers" ]]; then

	result_dir="${excel_dir}/test_results" 

	mkdir -p $result_dir

	export TRANSFORMERS_TEST_DEVICE="${device}"
	export TRANSFORMERS_TEST_DEVICE_SPEC="spec_${device}.py"
	export RUN_PT_TF_CROSS_TESTS="False"
	export RUN_PT_FLAX_CROSS_TESTS="False"
	export WANDB_DISABLED="true"

	# only skip tests that are cpu-only, tpu-only, npu-only, sagemaker-only and tf-only.
	# tests that are not xpu-relevant, e.g. apex, torch.fx, will be filtered during analysis
	NOT_RUN_MARKERS="not (not_device_test)"
	NOT_RUN_KEYWORDS="not (tpu or npu or tf or ModelOnTheFlyConversionTester or SigOpt or TrainerHyperParameterRayIntegrationTest or TrainerHyperParameterWandbIntegrationTest or TestTrainerDistributedNeuronCore or TestTrainerDistributedNPU)"

	echo "+++++++++run single test files++++++++++++++++"
	if [[ $dry_run == "1" ]]; then 
		pytest tests/*.py -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --collectonly -q 2>&1 | tee "${result_dir}/single_files.txt"
	else
		pytest tests/*.py -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${result_dir}/single_files.xlsx" --make-reports="${result_dir}/single_files" --timeout=600
	fi
	echo "+++++++++++++++++++++++done for single_files+++++++++++++"

	test_folders=("benchmark" "extended" "fsdp" "generation" "peft_integration" "trainer" "pipelines" "deepspeed")
	for folder in "${test_folders[@]}"
	do 
		echo "+++++++++run test folder $folder++++++++++++++++"
		if [[ $dry_run == "1" ]]; then 
			pytest tests/$folder -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --collectonly -q 2>&1 | tee "${result_dir}/${folder}.txt"
		else
			pytest tests/$folder -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${result_dir}/${folder}.xlsx" --make-reports="${result_dir}/${folder}" --timeout=600
		fi
		echo "+++++++++++++++++++++++done for $folder+++++++++++++"
	done 

	echo "+++++++++run quantization test folder++++++++++++++++"
	folder="quantization"
	quant_targets=("autoawq" "bnb" "quanto_integration")
	for quant_target in "${quant_targets[@]}"
	do 
		if [[ $dry_run == "1" ]]; then 
			pytest tests/$folder/$quant_target -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --collectonly -q 2>&1 | tee "${result_dir}/${folder}_${quant_target}.txt"
		else
			pytest tests/$folder/$quant_target -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${result_dir}/${folder}_${quant_target}.xlsx" --make-reports="${result_dir}/${folder}_${quant_target}" --timeout=600
		fi
	done

	for x in {a..z}
	do 
		echo "++++++++run models beginning with $x++++++++++++++++"
		if [[ $dry_run == "1" ]]; then 
			pytest tests/models/${x}* -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --collectonly -q 2>&1 | tee "${excel_dir}/${x}_models.txt"
		else 
			pytest tests/models/${x}* -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${result_dir}/${x}_models.xlsx" --make-reports="${result_dir}/${x}_models" --timeout=600
		fi
		echo "+++++++++++++++++++++++done for $x+++++++++++++"
	done

	# values=("c")
	# for x in "${values[@]}"; do
	#     python -m pytest tests/models/${x}* -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${result_dir}/${x}_models.xlsx" --make-reports="${result_dir}/${x}_models" --timeout=600
	#     echo "============done for $x============="
	# done 


	if [[ $dry_run == "1" ]]; then 
		total=0
		for file in "$result_dir"/*.txt; do
			if [[ -f "$file" ]]; then
				# Extract the last line of the file
        		last_line=$(tail -n 1 "$file")

				# Extract the first number in the last line
        		first_number=$(echo "$last_line" | grep -oE '^[0-9]+')

				# Add the number to the total
				if [[ -n "$first_number" ]]; then
					total=$((total + first_number))
				fi
			fi 
		done 

		echo "Total number of tests collected: $total"
	fi 
elif [[ $target == "accelerate" ]]; then
	if [[ $dry_run == "1" ]]; then 
		pytest tests --collectonly -q 2>&1 | tee "${excel_dir}/all_cases_collected.txt"
	else
		RUN_SLOW=1 pytest tests -sv --excelreport $report --timeout 600 2>&1 | tee $log
	fi
elif [[ $target == "peft" ]]; then
	if [[ $dry_run == "1" ]]; then 
		pytest tests --collectonly -q 2>&1 | tee "${excel_dir}/all_cases_collected.txt"
	else
		RUN_SLOW=1 pytest tests -sv --excelreport $report --timeout 600 2>&1 | tee $log
	fi
elif [[ $target == "diffusers" ]]; then
	export DIFFUSERS_TEST_DEVICE="${device}"
	export DIFFUSERS_TEST_DEVICE_SPEC="spec_${device}.py"
	
	if [[ $dry_run == "1" ]]; then 
		pytest tests --collectonly -q 2>&1 | tee "${excel_dir}/all_cases_collected.txt"
	else
		RUN_SLOW=1 pytest tests -sv --excelreport $report --timeout 600 2>&1 | tee $log
	fi
elif [[ $device == "optimum-quanto" ]]; then
	if [[ $dry_run == "1" ]]; then 
		pytest -rA test --collectonly -q 2>&1 | tee "${excel_dir}/all_cases_collected.txt"
	else
		RUN_SLOW=1 pytest -rA test --excelreport $report | tee $log
	fi
fi