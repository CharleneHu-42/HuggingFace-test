#!/bin/bash

########################### INPUT Variables START (modify if need)#########################
task_name="$1"                             # benchmark task name, choose one froem "mmlu" and "simple_bench"
backend="$2"                               # Backend, choose one from "trt-llm", "optimum-intel", "hf", "ipex", "tgi" and "vllm"
device="$3"                                # choose one from "cuda", "cpu" and "xpu"
models_list=(llama)                        # FIXED - DONT change this value, as currently only llama is supported
precision_list=(float16)                   # model data dtype
in_out_lengths=("32,1" "32,20")            # (max_input_token_num,max_new_tokens), to add more combinations, use ("32,1" "32,20" ...)
max_input_len="2048"                       # FIXED - DONT change this value
batch_sizes=(1)                            # to add multiple batch sizes, use (2 4 6 8 ...)
num_beams=(1)                              # number of beams, to add multiple batch sizes, use (1 2 ...)
do_sample="False"                          # FIXED - DONT change this value
############################ INPUT Variables END ##########################

data_dir=/workspace/data
log_folder=/workspace/mmlu-benchmark-log
tmp_log_folder=$log_folder/tmp
engine_build_logs_folder=${log_folder}/engine_build_logs
`mkdir -p $log_folder`
`rm -rf $engine_build_logs_folder`
`rm -rf $tmp_log_folder`
`mkdir -p $tmp_log_folder`
`mkdir -p $engine_build_logs_folder`
csv=${log_folder}/${backend}_MMLU_Benchmark_Results.csv

if [ ! -f "$csv" ]
then
	echo "Model, Precision, Do Sample, Beams, BS, Input length , Output length , Throughput(tokens/s), Avg Latency(ms), Start, End, Runtime(s)" > $csv
fi

for model in "${models_list[@]}"; do
    for precision in "${precision_list[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for num_beam in "${num_beams[@]}"; do
                for usecase in "${in_out_lengths[@]}"; do
                    IFS=',' read -ra elements <<< "$usecase"
                    input_len="${elements[0]}"
                    output_len="${elements[1]}"
                    echo "========== Running model: ${model} precision: ${precision} do sample: ${do_sample} num_beam: ${num_beam} BS: ${batch_size} input_len: ${input_len} output_len: ${output_len} =========="

                    if [ "$model" = "llama" ]; then
                        hf_model_dir=meta-llama/Llama-2-7b-chat-hf
                        trt_example_path=/workspace/TensorRT-LLM/examples/llama
                    else
                        echo " model name is invalid "
                    fi  
                    if [ "$backend" = "trt-llm" ]; then                             
                        if [ "$precision" = "float16" ]; then
                            engine_dir="${tmp_log_folder}/engines/${model}/${precision}/bs${batch_size}-beam${num_beam}-ip${input_len}-op${output_len}"
                            checkpoint_dir="${tmp_log_folder}/checkpoints/${model}/${precision}"
                            if [ ! -d "$checkpoint_dir" ]; then 
                                echo "========== Build model checkpoint for tensorrt-llm =========="
                                python3 ${trt_example_path}/convert_checkpoint.py --model_dir $hf_model_dir --dtype $precision --output_dir $checkpoint_dir
                            fi
                            if [ ! -d "$engine_dir" ]; then 
                                echo "========== Build model engine for tensorrt-llm =========="
                                trtllm-build --checkpoint_dir $checkpoint_dir --gemm_plugin $precision --max_batch_size $batch_size --max_input_len $max_input_len --max_beam_width $num_beam --output_dir $engine_dir 2>&1 | tee ${engine_build_logs_folder}/${model}-${precision}-${batch_size}-${no_of_beam}-${usecase}-build.log
                            fi
                            wait
                        else
                            echo "========== Enter valid precision =========="
                        fi
                    else
                        engine_dir="."
                    fi
                    echo "========== Ready for running benchmark =========="
                    start_time=$(date +%F-%T)
                    start_time_epoch=$(date +%s)
                    echo Start: $start_time
                    tmp_log_name=$tmp_log_folder/$model/BS_${batch_size}_beam_${num_beam}_ip_${input_len}_op_${output_len}.log
                    mkdir -p "$(dirname "$tmp_log_name")" && touch "$tmp_log_name"
                    python3 /workspace/benchmark/main.py --task_name $task_name --model_name $hf_model_dir --engine_dir $engine_dir --data_dir $data_dir --data_type $precision --device $device --input_tokens $input_len --max_new_tokens $output_len --max_input_len $max_input_len --backend $backend --batch_size $batch_size --num_beams $num_beam --do_sample $do_sample 2>&1 | tee $tmp_log_name 
                    wait
                    end_time=$(date +%F-%T)
                    end_time_epoch=$(date +%s)
                    elapsed=$(( end_time_epoch - start_time_epoch ))
                    echo End: $end_time
                    tmp_log_name=$tmp_log_folder/$model/BS_${batch_size}_beam_${num_beam}_ip_${input_len}_op_${output_len}.log
                    accuray=$(tail -n 1 $tmp_log_name | awk '{print $(NF-2)}')
                    avg_latency=$(tail -n 1 $tmp_log_name | awk '{print $(NF-4)}')
                    total_output_length=$(tail -n 1 $tmp_log_name | awk '{print $(NF-6)}')
                    total_input_length=$(tail -n 1 $tmp_log_name | awk '{print $(NF-8)}')
                    num_samples=$(tail -n 1 $tmp_log_name | awk '{print $(NF-10)}')
                    tokens_per_second=$(echo " $total_output_length / ($num_samples * $avg_latency / 1000 ) " | bc)
                    echo "accuray: $accuray"
                    echo "avg_latency(ms): $avg_latency"
                    echo "total_output_length: $total_output_length "
                    echo "total_input_length: $total_input_length "
                    echo "num_samples: $num_samples "
                    echo "throughput(tokens/s): $tokens_per_second "
                    echo ${model}, ${precision}, ${do_sample}, ${num_beam}, ${batch_size}, ${input_len}, ${output_len}, ${tokens_per_second}, ${avg_latency}, ${start_time}, ${end_time}, ${elapsed} >> $csv          
                done
            done
        done
    done
done
