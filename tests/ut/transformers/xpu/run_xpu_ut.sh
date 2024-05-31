export RUN_SLOW=1
export TRANSFORMERS_TEST_DEVICE="xpu"
export TRANSFORMERS_TEST_DEVICE_SPEC="spec.py"

excel_dir="$1"

echo "+++++++++remove excel dir if exists and create a new++++++++++++"
rm -fr $excel_dir 
mkdir $excel_dir

NOT_RUN_MARKERS="not (not_device_test or require_ray or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_neuroncore or require_torch_tensorrt_fx or require_pytorch_quantization or require_apex or require_tf or require_flax or require_torch_xla or torch_fx or require_detectron2 or flash_attn_test or require_flash_attn or require_quanto or require_auto_gptq or require_auto_awq or require_aqlm or require_natten or require_bitsandbytes or is_staging_test or is_pt_tf_cross_test or is_pt_flax_cross_test)"
NOT_RUN_KEYWORDS="not (tpu or npu or cuda or flax or tf)"

echo "+++++++++run single test files++++++++++++++++"
pytest tests/*.py -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --excelreport="${excel_dir}/single_files.xlsx" --make-reports="single_files" --timeout=600
echo "+++++++++++++++++++++++done for single_files+++++++++++++"

test_folders=("benchmark" "extended" "fsdp" "generation" "peft_integration" "quantization" "trainer" "pipelines")
for folder in "${test_folders[@]}"
do 
	echo "+++++++++run test folder $folder++++++++++++++++"
	pytest tests/$folder -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --excelreport="${excel_dir}/${folder}.xlsx" --make-reports="${folder}" --timeout=600
	echo "+++++++++++++++++++++++done for $folder+++++++++++++"
done 

for x in {a..z}
do 
	echo "++++++++run models beginning with $x++++++++++++++++"
	pytest tests/models/${x}* -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --excelreport="${excel_dir}/${x}_models.xlsx" --make-reports="${x}_models" --timeout=600
	echo "+++++++++++++++++++++++done for $x+++++++++++++"
done