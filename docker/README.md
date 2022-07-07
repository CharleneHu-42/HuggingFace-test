1. Build dockerfile: docker build -t df_1 .
2. Start the image: docker run -it --privileged --shm-size 800g "image id" /bin/bash
3. Activate env: i) source /opt/intel/oneapi/setvars.sh
		 ii) export MASTER_ADDR=127.0.0.1 && export MASTER_PORT=29500 && export CCL_WORKER_COUNT=1

Then, you can run the model.
Run distribution BERT large refine tune for question and answer in two CPU sockets in one machine by this command:

mpirun -n 2 -genv OMP_NUM_THREADS=23 python3 examples/pytorch/question-answering/run_qa.py   --model_name_or_path bert-large-uncased   --dataset_name squad   --do_train   --do_eval   --per_device_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir /tmp/debug_squad/ --no_cuda --xpu_backend ccl --dataloader_pin_memory false
