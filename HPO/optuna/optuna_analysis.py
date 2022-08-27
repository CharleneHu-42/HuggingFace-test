import optuna
loaded_study = optuna.load_study(study_name="glue-study", storage="sqlite:///glue.db")
study = optuna.create_study(study_name="squad1",storage="sqlite:///glue.db",direction="maximize")

for trial in loaded_study.trials:
    if trial.values != None and trial.values != [0.0]:
        study.add_trial(trial)

fig = optuna.visualization.plot_optimization_history(study)

fig.write_image(format='png',file="result.png")

fig = optuna.visualization.plot_parallel_coordinate(study)
fig.write_image(format='png',file="param_rel.png")

fig = optuna.visualization.plot_param_importances(study)
fig.write_image(format='png',file="param_importances.png")

fig = optuna.visualization.plot_intermediate_values(study)
fig.write_image(format='png',file="param_intermediate.png")

optuna.delete_study(study_name="squad1",storage="sqlite:///glue.db")
