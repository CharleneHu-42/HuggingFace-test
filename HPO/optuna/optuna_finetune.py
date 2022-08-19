import optuna
import subprocess
import re
import os

f1_reg=re.compile("eval_f1[ ]+=[ ]+(?P<eval_f1>[1-9]\d*.\d*|0.\d*[1-9]\d*)")
training_time_reg=re.compile("'train_runtime':[ ]+(?P<time>[1-9]\d*.\d*|0.\d*[1-9]\d*)")
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

def get_training_time(lines):
    time = None
    for line in lines:
        match = training_time_reg.search(line)
        if not match:
            continue
        time = match.groupdict().get('time')
        if time:
            time = float(time)
            break
    return time

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 20)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64])
    cmdstr = "mpirun --bootstrap ssh -n 2 -genv OMP_NUM_THREADS=24 python examples/pytorch/text-classification/run_glue.py --model_name_or_path bert-base-cased --task_name mrpc --do_train --do_eval --max_seq_length 128 --output_dir /tmp/mrpc/ --overwrite_output_dir True --xpu_backend ccl --no_cuda"
    cmdstr += " --learning_rate "+str(learning_rate)
    cmdstr += " --num_train_epochs "+str(num_train_epochs)
    cmdstr += " --per_device_train_batch_size " + str(per_device_train_batch_size)
    exit, output = subprocess.getstatusoutput(cmdstr)
    F1 = 0
    if exit == 0:
        F1 = get_eval_f1(output.split('\n'))
    else:
        print(output)
    return F1

if os.path.isfile('glue.db'):
    os.remove("glue.db")
study = optuna.create_study(study_name="glue-study", storage="sqlite:///glue.db", load_if_exists=False, direction="maximize")
study.optimize(objective, n_trials=10)

best_trial = study.best_trial
print("best trial",best_trial)
