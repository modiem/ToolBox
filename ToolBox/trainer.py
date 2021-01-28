import time
from sklearn.linear_model import LinearRegression

from memoized_property import 

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

class Mlflow(object):
    
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
        
    # def train(self):      
    #     self.mlflow_create_run()
    #     self.mlflow_log_param("model", "linear")
    #     self.mlflow_log_metric("rmse", 4)


class Trainer(Mlflow):
    ESTIMATOR = LinearRegression()

    def __init__(self, X, y, **kwargs):
        self.kwargs = kwargs
        self.split = self.kwargs.get("split", True)
        self.test_size = self.kwargs.get("test_size", 0.3)
        self.X_train = X
        self.y_train = y
        if self.split:
            self.X_train, self.X_val, self.y_train, self_y_val = \
                train_test_split(self.X_train, self.y_train, test_size = self.test_size)
        self.nrows = self.X_train.shape[0]
        self.pipeline = self.kwargs.get("pipeline", None)

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        params = self.kwargs.get("estimator_params", {})
        estimator.set_params(**params)
        print(estimator.__class__.__name__)
        return estimator

    def set_pipeline(self):
        self.pipeline = Pipeline(steps = [
            # ('features', features_encoder),
            ('model', self.get_estimator())
        ])

    @simple_time_tracker
    def train(self):
        if self.pipeline = None:
            self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

        