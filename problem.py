import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Titanic survival classification'
prediction_type = rw.prediction_types.multiclass
workflow = rw.workflows.FeatureExtractorClassifier()
prediction_labels = [0, 1]

score_types = [
    rw.score_types.ROCAUC(name='auc', n_columns=len(prediction_labels)),
    rw.score_types.Accuracy(name='acc', n_columns=len(prediction_labels)),
    rw.score_types.NegativeLogLikelihood(
        name='nll', n_columns=len(prediction_labels)),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


_target_column_name = 'Survived'
_ignore_column_names = ['PassengerId']


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
