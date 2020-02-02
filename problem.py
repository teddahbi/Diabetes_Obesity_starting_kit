import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit


problem_title = 'Diabetes rate estimation'
_target_column_name = 'PCT_OBESE_ADULTS'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow

class FAN(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor']):
        super(FAN, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names

workflow = FAN()

# define the score (specific score for our problem)
class CUSTOM_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='custom error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        scores = (y_true - y_pred)/np.maximum(5, y_true)
        loss = np.mean(scores**2)*100

        return loss

score_types = [
    CUSTOM_error(name='custom error', precision=2),
]

def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=4, test_size=0.33,random_state = 42)
    return cv.split(X,y, groups=X['FIPS'])

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False,index_col = 0)
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_train_data(path='.'):
    f_name = 'final_dataset_TRAIN.csv'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'final_dataset_TEST.csv'
    return _read_data(path, f_name)