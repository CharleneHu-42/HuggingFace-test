import optuna
import subprocess
import re
import yaml
import os

f1_reg=re.compile("Optimized model with eval_f1 of (?P<eval_f1>[1-9]\d*.\d*|0.\d*[1-9]\d*)")
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
    ce_loss_weight = trial.suggest_float("ce_loss_weight", 0.0, 1.0, step=0.1)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 20)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64])
    try:
        with open("distillation.yml", "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print("no distillation.yml, copy it from example/config to current folder")
        raise
    config_dict['distillation']['train']['criterion']['KnowledgeDistillationLoss']['loss_weights']=[ce_loss_weight, 1.0-ce_loss_weight]
    with open("distillation.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f)
    cmdstr = "mpirun --bootstrap ssh -n 2 -genv OMP_NUM_THREADS=24 python run_qa.py --overwrite_output_dir --model_name_or_path nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large --dataset_name squad --apply_distillation  --teacher_model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad --do_train --do_eval --output_dir ./output_dir/bert-large-distill_MiniLM/squad_output_test_3 --per_device_eval_batch_size 32 --no_cuda --xpu_backend ccl --distillation_config ."
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
