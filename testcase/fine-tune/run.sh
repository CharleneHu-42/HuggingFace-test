task_name="$1"
distributed="$2"

export CCL_ZE_IPC_EXCHANGE=sockets

if [[ -z "$distributed" ]]; then
    python run_$task_name.py
else 
    accelerate launch --config_file lora_config.yaml run_$task_name.py
fi
