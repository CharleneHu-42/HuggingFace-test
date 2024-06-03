#!/bin/bash

########################### INPUT Variables START #########################

batch_sizes=(1 2)               # if you want you can add multiple batch sizes like (2 4 6 8 ...)
models_list=(llama2)            # FIXED - DONT remove or change this value.
precision_list=(float16)        # FIXED - DONT REMOVE or chnage this value.
no_of_beams=1                   # Change the parametre depending on number of beams you want to pass.
: '
    Usecase is just some random name to that ip / op token combination.
    ==> ex : "usecase-name,ip-token-value,op-token-value" ==> "normal,32,32" 
'
# For Single usecase with ip & op tokens use this.
usecases=(
    "normal,32,32"  
)

############################ INPUT Variables END ##########################

wrk_dir=/code/tensorrt_llm
log_folder=${wrk_dir}/usecase-sweep-log
tmp_log_folder=$log_folder/tmp
engine_build_logs_folder=${log_folder}/engine_build_logs
`mkdir -p $log_folder`
`rm -rf $engine_build_logs_folder`
`rm -rf $tmp_log_folder`
`mkdir -p $tmp_log_folder`
`mkdir -p $engine_build_logs_folder`
gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
gpu_name=$(echo "$gpu_info" | tr -d '[:space:]')
csv=${log_folder}/$(hostname)_GPU-BEAM-4-INT8-KV-CACHE_results.csv
if [ ! -f "$csv" ]
then
	echo "Hostname, Script revision, GPU , No of gpus , Model, Precision, Beams, Warmup-steps,  steps, BS, Usecase, input length , output length , Throughput(tokens/s), Latency(ms), first token latency(ms), Second token latency(ms), start, End, Runtime(s)" > $csv
fi
for model in "${models_list[@]}"; do
    for precision in "${precision_list[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for usecase in "${usecases[@]}"; do
                IFS=',' read -ra elements <<< "$usecase"
                for element in "${elements[@]}"; do
                    usecase_value="${elements[0]}"
                    input_len="${elements[1]}"
                    output_len="${elements[2]}"
                    echo "========= Running model: ${model} usecase: ${usecase_value} ${precision} BS: ${batch_size} input_len: ${input_len} output_len: ${output_len} =========="

                    if [ "$model" = "gptj_6b" ]; then
			            # use this condition when u need kv-cache for gptj
                        cd /code/tensorrt_llm/examples/gptj/                              
                        if ["$precision" = "float16" ]; then
                            engine_dir="${tmp_log_folder}/engines/${model}/${precision}/${usecase_value}"
                            python build.py --model_dir ./gptj_model --dtype $precision --int8_kv_cache --parallel_build --world_size 1 --remove_input_padding --use_gpt_attention_plugin $precision --enable_context_fmha --use_gemm_plugin $precision --max_batch_size $batch_size --max_input_len $input_len --max_output_len $output_len --output_dir $engine_dir --max_beam_width $no_of_beams  > ${engine_build_logs_folder}/${model}-${precision}-${batchsize}-${usecase_value}-build.log 2>&1 &
                            wait
                        else
                            echo " Enter valid precision "
                        fi
                    else
                        echo " model name is invalid "
                    fi
                    echo "Building engine Completed now running benchmark"
                    start_time=$(date +%s)
                    start_time=$(date +%F-%T)
                    start_time_epoch=$(date +%s)
                    echo Start: $start_time
                    tmp_log_name=$tmp_log_folder/$model/${usecase_value}_BS_${batch_size}_ip_${input_len}_op${output_len}.log
                    mkdir -p "$(dirname "$tmp_log_name")" && touch "$tmp_log_name"
                    mpirun --allow-run-as-root -n $num_gpu python3 /code/tensorrt_llm/benchmarks/python/benchmark.py -m gptj_6b --mode plugin --engine_dir $engine_dir --dtype $precision --warm_up $warmup_steps --num_runs $steps --batch_size $batch_size --input_output_len "$input_len,$output_len" --num_beams $no_of_beams > $tmp_log_name 2>&1 & 
                    wait
                    end_time=$(date +%F-%T)
                    end_time_epoch=$(date +%s)
                    elapsed=$(( end_time_epoch - start_time_epoch ))
                    echo End: $end_time
                    tokens_per_second=$(grep -oP 'tokens_per_sec \K\d+\.\d+' $tmp_log_name)
                    latency=$(grep -oP 'latency\(ms\) \K\d+\.\d+' $tmp_log_name)
                    echo "tokens_per_second: $tokens_per_second"
                    echo "latency(ms): $latency"
                    iter_latency=$(tail -n 1 $tmp_log_name | awk '{print $(NF-2)}' )
                    iter_latency=$(echo "scale=3; $iter_latency / 1000" | bc)
                    input_length=$(grep "first" $tmp_log_name | tail -1 |awk 'BEGIN{FS=","} {print $1 ",", $2 }')
                    first_latency=$(grep "first" $tmp_log_name | tail -5 |awk 'BEGIN{FS=","} {sum+=$4} END {print sum/NR}')
                    next_latency=$(grep "first" $tmp_log_name | tail -1 |awk 'BEGIN{FS=","} {print $(NF)}')
                    echo "first token latency(ms): $first_latency "
                    echo "second token latency(ms): $next_latency "
                    echo $(hostname), ${script_rev%.*}, ${gpu_name},${num_gpu}, ${model}, ${precision}, ${no_of_beams}, ${warmup_steps}, ${steps}, ${batch_size}, ${usecase_value}, ${input_len}, ${output_len}, ${tokens_per_second}, ${latency}, ${first_latency}, ${next_latency} , ${start_time}, ${end_time}, ${elapsed} >> $csv
                    break
                done               
            done
        done
    done
done
