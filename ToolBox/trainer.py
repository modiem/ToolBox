import time
from datetime import datetime
from sklearn.linear_model import LinearRegression


from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_selector

from tempfile import mkdtemp
from shutil import rmtree

class MlflowTracker(object):
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
    
    @memoized_property
    def mlflow_client(self):
        return MlflowClient()
    
    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.\
                    create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.\
                    get_experiment_by_name(self.experiment_name).experiment_id
        
    def mlflow_create_run(self):
        self.mlflow_run = self.mlflow_client.create_run(self.mlflow_experiment_id)
        
    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)
        
    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
        


class Trainer(MlflowTracker):
    '''
    Usage:
    params = {
        'nrows': 10000,
        'estimator': RamdonForest(),
        'estimator_params' = {},
        'split' = True,
        'test_size = 0.2,
        'pipeline' = pipeline,
        'pipeline_use_distance' = True
    }

    trainer = Trainer(X_train, y_train, **params)
    '''
    ESTIMATOR = LinearRegression()

    def __init__(self, X, y,  **kwargs):
        self.kwargs = kwargs

        ## Initiate MLflow
        self.experiment_name = self.kwargs.get("experiment_name", str(datetime.now()))   
        super().__init__(self.experiment_name)
        self.mlflow_create_run()
        
        ## Train test split
        self.split = self.kwargs.get("split", True)
        self.test_size = self.kwargs.get("test_size", 0.3)
        self.X_train = X
        self.y_train = y
        if self.split:
            self.X_train, self.X_val, self.y_train, self_y_val = \
                train_test_split(self.X_train, self.y_train, test_size = self.test_size)
        
        ## Print shape
        print("Train data shape:", X_train.shape)
        

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        params = self.kwargs.get("estimator_params", {})
        estimator.set_params(**params)
        print(estimator.__class__.__name__)
        return estimator

    def set_pipeline(self):

        self.pipeline = self.kwargs.get("pipeline", None)

        # Create a temp folder
        cachedir = mkdtemp()

        # Pipeline structure
        num_transformer = MinMaxScaler()
        cat_transformer = OneHotEncoder(handle_unknown = 'ignore')
        feateng_blocks = [
            ("num_transformer", num_transformer, make_column_selector(dtype_include = ['int', 'float'])),
            ("cat_transformer", cat_transformer, make_column_selector(dtype_include = ['object', 'bool']))
        ]               
        features_encoder = columnTransformer(feateng_blocks,
                                             n_jobs = None,
                                             remainder = "drop"
                                            )

        # Combine preprocessing and model:
        self.pipeline = Pipeline(steps = [
            ('features', features_encoder),
            ('model', self.get_estimator())
        ],
        memory = cachedir # Avoid recalculating transformer variables during cross validations or grid searches
        )

        # Clear the cache directory after the cross-validation
        rmtree(cachedir)


    @simple_time_tracker
    def cross_validate(self, cv = 5, scoring = "r2"):
        if self.pipeline = None:
            self.set_pipeline()
        
        for key,value in self.pipeline.get_params().items():
            self.mlflow_log_param(key, value)

        score = cross_val_score(self.pipeline, self.X_train, self.y_train, cv=cv, scoring=scoring).mean()
        self.mlflow_log_metric(scoring, score)
        return score

    def save_model(self):
        """
        Save the model into a .joblib format
        """
        joblib.dump(self.pipeline, "model.joblib")
        print("model.joblib saved locally.")