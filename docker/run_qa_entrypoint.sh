#!/bin/bash

# Activate ssh
mkdir /run/sshd
/usr/sbin/sshd

if [ $master_node = True ];then
  # Activate mpi and conda
  source /opt/intel/oneapi/setvars.sh

  # Set env param
  export MASTER_ADDR=${master_addr}
  export MASTER_PORT=${master_port}
  export CCL_WORKER_COUNT=${ccl_worker_count}
  
  # Start training BERT
  run_file="transformers/examples/pytorch/question-answering/run_qa.py"

  mpirun -f ./tmp/hostfile -np ${mpi_n} -ppn ${mpi_ppn} -genv OMP_NUM_THREADS=${omp_num_threads} python3 ${run_file} --model_name_or_path ${model_name} --dataset_name ${dataset_name}   --do_train  --do_eval  --per_device_train_batch_size ${per_device_train_batch_size} --learning_rate ${learning_rate} --num_train_epochs ${num_train_epochs} --max_seq_length ${max_seq_length} --doc_stride ${doc_stride} --output_dir ${output_dir}  --no_cuda   --xpu_backend ${xpu_backend} --dataloader_pin_memory ${dataloader_pin_memory} --use_ipex ${use_ipex} --bf16 ${bf16} --overwrite_output_dir
else
  while true
  do
     echo "sleep 1"
     sleep 1
  done
fi
