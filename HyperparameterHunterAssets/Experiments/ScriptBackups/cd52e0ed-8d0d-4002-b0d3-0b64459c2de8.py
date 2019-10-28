# # 开始
# # 1.环境
#
# from hyperparameter_hunter import Environment, CVExperiment
# import pandas as pd
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import StratifiedKFold
#
# data = load_breast_cancer()
# df = pd.DataFrame(data=data.data, columns=data.feature_names)
# df['target'] = data.target
#
# env = Environment(
#     train_dataset=df,  # Add holdout/test dataframes, too
#     results_path='/Users/gaofan/Desktop/directory',  # Where your result files will go
#     metrics=['roc_auc_score'],  # Callables, or strings referring to `sklearn.metrics`
#     cv_type=StratifiedKFold,  # Class, or string in `sklearn.model_selection`
#     cv_params=dict(n_splits=5, shuffle=True, random_state=32)
# )
#
# # 2.实验
# from xgboost import XGBClassifier
# experiment = CVExperiment(
#     model_initializer=XGBClassifier,
#     model_init_params=dict(objective='reg:linear', max_depth=3, n_estimators=100, subsample=0.5)
# )
#
# from hyperparameter_hunter import Real, Integer, Categorical
# from hyperparameter_hunter import optimization as opt
#
# optimizer = opt.BayesianOptPro(iterations=10)
# optimizer.forge_experiment(
#     model_initializer=XGBClassifier,
#     model_init_params=dict(
#         max_depth=Integer(low=2, high=20),
#         learning_rate=Real(0.0001, 0.5),
#         n_estimators=200,
#         subsample=0.5,
#         booster=Categorical(['gbtree', 'gblinear', 'dart']),
#     )
# )
# optimizer.go()


from hyperparameter_hunter import Environment, CVExperiment, BayesianOptPro, Integer
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
from xgboost import XGBClassifier

# Start by creating an `Environment` - This is where you define how Experiments (and optimization) will be conducted
env = Environment(
    train_dataset=get_breast_cancer_data(target='target'),
    results_path='HyperparameterHunterAssets',
    metrics=['roc_auc_score'],
    cv_type='StratifiedKFold',
    cv_params=dict(n_splits=10, shuffle=True, random_state=32),
)

# Now, conduct an `Experiment`
# This tells HyperparameterHunter to use the settings in the active `Environment` to train a model with these hyperparameters
experiment = CVExperiment(
    model_initializer=XGBClassifier,
    model_init_params=dict(
        objective='reg:linear',
        max_depth=3
    )
)

# That's it. No annoying boilerplate code to fit models and record results
# Now, the `Environment`'s `results_path` directory will contain new files describing the Experiment just conducted

# Time for the fun part. We'll set up some hyperparameter optimization by first defining the `OptPro` (Optimization Protocol) we want
optimizer = BayesianOptPro(verbose=1)

# Now we're going to say which hyperparameters we want to optimize.
# Notice how this looks just like our `experiment` above
optimizer.forge_experiment(
    model_initializer=XGBClassifier,
    model_init_params=dict(
        objective='reg:linear',  # We're setting this as a constant guideline - Not one to optimize
        max_depth=Integer(2, 10)  # Instead of using an int like the `experiment` above, we provide a space to search
    )
)
# Notice that our range for `max_depth` includes the `max_depth=3` value we used in our `experiment` earlier

optimizer.go()  # Now, we go

assert experiment.experiment_id in [_[2] for _ in optimizer.similar_experiments]
# Here we're verifying that the `experiment` we conducted first was found by `optimizer` and used as learning material
# You can also see via the console that we found `experiment`'s saved files, and used it to start optimization

last_experiment_id = optimizer.current_experiment.experiment_id
# Let's save the id of the experiment that was just conducted by `optimizer`

optimizer.go()  # Now, we'll start up `optimizer` again...

# And we can see that this second optimization round learned from both our first `experiment` and our first optimization round
assert experiment.experiment_id in [_[2] for _ in optimizer.similar_experiments]
assert last_experiment_id in [_[2] for _ in optimizer.similar_experiments]
# It even did all this without us having to tell it what experiments to learn from

# Now think about how much better your hyperparameter optimization will be when it learns from:
# - All your past experiments, and
# - All your past optimization rounds
# And the best part: HyperparameterHunter figures out which experiments are compatible all on its own
# You don't have to worry about telling it that KFold=5 is different from KFold=10,
# Or that max_depth=12 is outside of max_depth=Integer(2, 10)