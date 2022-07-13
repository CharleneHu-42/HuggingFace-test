#!/usr/bin/env bash

#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



set -e

function usage_help() {
    echo -e "options:"
    echo -e "  -h Display help"
    echo -e "  -i {image_id}"
    echo -e "  -m {model_name}"
}


# Set env param
master_addr=127.0.0.1
master_port=29500
ccl_worker_count=1

# Start training BERT
mpi_n=2
omp_num_threads=23
model_name=bert-large-uncased
dataset_name=squad
per_device_train_batch_size=12
learning_rate=3e-5
num_train_epochs=2
max_seq_length=384
doc_stride=128
output_dir="tmp/debug_squad/"
xpu_backend=ccl
dataloader_pin_memory=False
bf16=False
use_ipex=True

# Override args
while getopts "h?r:i:m:" OPT; do
    case $OPT in
        h|\?)
            usage_help
            exit 0
            ;;
        i)
            echo -e "Option $OPTIND, image_id = $OPTARG"
            image_id=$OPTARG
            ;;
        m)
            echo -e "Option $OPTIND, model_name = $OPTARG"
            model_name=$OPTARG
            ;;
        ?)
            echo -e "Unknown option $OPTARG"
            usage_help
            exit 0
            ;;
    esac
done

docker run \
    --privileged --shm-size 800g \
    -v /tmp/:/usr/local/tmp/ \
    -e learning_rate=${learning_rate} \
    -e max_seq_length=${max_seq_length} \
    -e dataloader_pin_memory=${dataloader_pin_memory} \
    -e model_name=${model_name} \
    -e output_dir=${output_dir} \
    -e mpi_n=${mpi_n} \
    -e omp_num_threads=${omp_num_threads} \
    -e per_device_train_batch_size=${per_device_train_batch_size} \
    -e learning_rate=${learning_rate} \
    -e num_train_epochs=${num_train_epochs} \
    -e xpu_backend=${xpu_backend} \
    -e doc_stride=${doc_stride}\
    -e ccl_worker_count=${ccl_worker_count} \
    -e master_addr=${master_addr} \
    -e master_port=${master_port} \
    -e dataset_name=${dataset_name} \
    -e use_ipex=${use_ipex} \
    -e bf16=${bf16} \
    ${image_id}

