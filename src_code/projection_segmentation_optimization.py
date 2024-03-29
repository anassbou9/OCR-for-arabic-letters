import numpy as np
import preprocessing as pre
import segmentation as seg
from sklearn.metrics import mean_squared_error
import optuna
import os


def objective(trial):
    cut = trial.suggest_int("cut", 2, 8)
    thresh = trial.suggest_int("thresh", 0, 26)
    y_true = [16, 14, 16, 18, 14, 14]
    y_predicted = []
    for i in range(2, 8):
        curr_direct_path = os.path.dirname(os.path.abspath(__file__))
        input_test_path = (
            curr_direct_path + "/../books_for_ocr/scanned_pics/test_" + str(i) + ".PNG"
        )
        original_img, preprocessed_img = pre.preprocess(input_test_path)
        segments = seg.segment(preprocessed_img, "horizontal", thresh, cut)
        y_predicted.append(len(segments))

    return mean_squared_error(y_true, y_predicted)


study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=30)  # Invoke optimization of the objective function.
best_trial, best_params, best_value = (
    study.best_trial,
    study.best_params,
    study.best_value,
)
print("best params:", best_params)
print("best value:", best_value)
