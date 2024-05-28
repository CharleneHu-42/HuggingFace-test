import optuna
study = optuna.load_study(study_name="squad-study", storage="sqlite:///finetune.db")

fig = optuna.visualization.plot_pareto_front(study,target_names=["F1","speed"])
fig.write_image(format='png',file="result_pf.png")


def get_eval_f1(t):
    return t.values[0]

def get_speed(t):
    return t.values[1]

fig = optuna.visualization.plot_optimization_history(study,target=get_eval_f1, target_name="eval_f1")
fig.write_image(format='png',file="result.png")

fig = optuna.visualization.plot_parallel_coordinate(study,target=get_eval_f1, target_name="eval_f1")
fig.write_image(format='png',file="param_rel.png")

fig = optuna.visualization.plot_param_importances(study,target=get_eval_f1,target_name="eval_f1")
fig.write_image(format='png',file="param_importances.png")

#fig = optuna.visualization.plot_intermediate_values(study)
#fig.write_image(format='png',file="param_intermediate.png")

