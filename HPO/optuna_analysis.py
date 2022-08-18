import optuna
loaded_study = optuna.load_study(study_name="glue-study", storage="sqlite:///glue.db")
fig = optuna.visualization.plot_optimization_history(loaded_study)

fig.write_image(format='png',file="result.png")

fig = optuna.visualization.plot_parallel_coordinate(loaded_study)
fig.write_image(format='png',file="param_rel.png")
