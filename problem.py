import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Fake News Detection'
_target_column_name = 'label'
_ignore_column_names = ['ID']
#_prediction_label_names = [0, 1, 2, 3, 4, 5]
_prediction_label_names = ['barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true']
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

score_types = [
#    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.Accuracy(name='acc'),
#    rw.score_types.NegativeLogLikelihood(name='nll'),
#    rw.score_types.F1Above()
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
#    11972.json	true	 Building a wall on the U.S.-Mexico border will take literally years.	
#immigration	rick-perry	Governor	Texas	republican	30	30	42	23	18	Radio interview

    data = pd.read_csv(os.path.join(path, 'data', f_name), sep='\t', names=['ID', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'true_counts',   'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire', 'context'])
    y_array = data['label'].values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
