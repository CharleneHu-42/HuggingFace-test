device="$1"
excel_dir="$2"

export RUN_SLOW=1
export RUN_PT_TF_CROSS_TESTS="False"
export RUN_PT_FLAX_CROSS_TESTS="False"
export TRANSFORMERS_TEST_DEVICE="${device}"
export TRANSFORMERS_TEST_DEVICE_SPEC="spec_${device}.py"
export WANDB_DISABLED="true"

echo "+++++++++remove excel dir if exists and create a new++++++++++++"
rm -fr $excel_dir 
mkdir $excel_dir

# only skip tests that are cpu-only, tpu-only, npu-only, sagemaker-only and tf-only.
# tests that are not xpu-relevant will be filtered during analysis
NOT_RUN_MARKERS="not (not_device_test)"
NOT_RUN_KEYWORDS="not (tpu or npu or tf or ModelOnTheFlyConversionTester or SigOpt or TrainerHyperParameterRayIntegrationTest)"

echo "+++++++++run single test files++++++++++++++++"
pytest tests/*.py -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${excel_dir}/single_files.xlsx" --make-reports="single_files" --timeout=600
echo "+++++++++++++++++++++++done for single_files+++++++++++++"

test_folders=("benchmark" "extended" "fsdp" "generation" "peft_integration" "quantization" "trainer" "pipelines" "deepspeed")
for folder in "${test_folders[@]}"
do 
	echo "+++++++++run test folder $folder++++++++++++++++"
	pytest tests/$folder -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${excel_dir}/${folder}.xlsx" --make-reports="${folder}" --timeout=600
	echo "+++++++++++++++++++++++done for $folder+++++++++++++"
done 

for x in {a..z}
do 
	echo "++++++++run models beginning with $x++++++++++++++++"
	pytest tests/models/${x}* -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${excel_dir}/${x}_models.xlsx" --make-reports="${x}_models" --timeout=600
	echo "+++++++++++++++++++++++done for $x+++++++++++++"
done