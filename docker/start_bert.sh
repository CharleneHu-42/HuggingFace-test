#!/bin/sh
# source /root/anaconda3/bin/activate pytorch

# Activate mpi and conda
source /opt/intel/oneapi/setvars.sh

# Set env param
master_addr=127.0.0.1
master_port=29500
ccl_worker_count=1
export MASTER_ADDR=${master_addr}
export MASTER_PORT=${master_port}
export CCL_WORKER_COUNT=${ccl_worker_count}

# Start training BERT
mpi_n=2
omp_num_threads=23
run_file="transformers/examples/pytorch/question-answering/run_qa.py"
model_name=bert-large-uncased
dataset_name=squad
per_device_train_batch_size=12
learning_rate=3e-5
num_train_epochs=2
max_seq_length=384
doc_stride=128
output_dir="/tmp/debug_squad/"
xpu_backend=ccl
dataloader_pin_memory=false

mpirun -n ${mpi_n} -genv OMP_NUM_THREADS=${omp_num_threads} python3 ${run_file} --model_name_or_path ${model_name} --dataset_name ${dataset_name}       --do_train      --do_eval       --per_device_train_batch_size ${per_device_train_batch_size} --learning_rate ${learning_rate} --num_train_epochs ${num_train_epochs} --max_seq_length ${max_seq_length} --doc_stride ${doc_stride} --output_dir ${output_dir}   --no_cuda       --xpu_backend ${xpu_backend} --dataloader_pin_memory ${dataloader_pin_memory}

