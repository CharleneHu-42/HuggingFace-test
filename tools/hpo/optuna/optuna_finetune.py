import optuna
import subprocess
import re
import os

f1_reg=re.compile("eval_f1[ ]+=[ ]+(?P<eval_f1>[1-9]\d*.\d*|0.\d*[1-9]\d*)")
training_speed_reg=re.compile("train_samples_per_second[ ]+=[ ]+(?P<speed>[1-9]\d*.\d*|0.\d*[1-9]\d*)")
def get_eval_f1(lines):
    f1 = None
    for line in lines:
        match = f1_reg.search(line)
        if not match:
            continue
        f1 = match.groupdict().get('eval_f1')
        if f1:
            f1 = float(f1)
            break
    return f1

def get_training_speed(lines):
    time = None
    for line in lines:
        match = training_speed_reg.search(line)
        if not match:
            continue
        speed = match.groupdict().get('speed')
        if speed:
            speed = float(speed)
            break
    return speed

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 4e-4, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64])
    cmdstr = "mpirun --bootstrap ssh -np 2 -ppn 2 -genv CCL_WORKER_COUNT=1 -genv I_MPI_PIN_DOMAIN=socket -genv OMP_NUM_THREADS=24 -genv HF_DATASETS_OFFLINE=1 -genv TRANSFORMERS_OFFLINE=1  python3 run_qa.py --model_name_or_path bert-large-uncased --dataloader_pin_memory false --dataset_name squad --do_train --do_eval --max_seq_length 384   --doc_stride 128  --output_dir /skyrex05/wangyi/tmp/ --overwrite_output_dir --no_cuda --xpu_backend ccl --logging_steps 500 --data_seed 0 --use_ipex --bf16"
    cmdstr += " --learning_rate "+str(learning_rate)
    cmdstr += " --num_train_epochs "+str(num_train_epochs)
    cmdstr += " --per_device_train_batch_size " + str(per_device_train_batch_size)
    exit, output = subprocess.getstatusoutput(cmdstr)
    F1 = 0
    speed = 0
    if exit == 0:
        F1 = get_eval_f1(output.split('\n'))
        speed = get_training_speed(output.split('\n'))
    if F1 != 0 and speed != 0:
        print(F1,speed)
        return F1, speed
    else:
        print(output)
        raise Exception("no output")

study = optuna.create_study(study_name="squad-study", storage="sqlite:///finetune.db", load_if_exists=True, directions=["maximize","maximize"])
study.optimize(objective, n_trials=10)

best_trial = study.best_trial
print("best trial",best_trial)
