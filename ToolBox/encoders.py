import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ToolBox.utils import get_manhattan_distance, get_euclidian_distance, get_haversine_distance
import pygeohash as gh

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extract the day of week (dow), the hour, the month and the year from a timestamp column.
        Return a copy of the DataFrame X with only four columns: 'dow', 'hour', 'mont', 'year'.
    """
    def __init__(self, time_col = "time_stamp", time_zone = "UTC"):
        self.time_col = time_col
        self.time_zone = time_zone
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_col])
        X_.index = X_.index.tz_convert(self.time_zone)
        X_['dow'] = X_.index.weekday
        X_['hour'] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[['dow', 'hour', 'month', 'year']]

class DistanceTransformer(BaseEstimator, TransformerMixin):
    '''
        Compute the haversine/euclidian/mahattan distance between two GPS points.
        Return a copy of the DataFrame X with only one column: 'distance'
    '''
    def __init__(self, 
                 lat1 = "start_latitude", 
                 lat2 = "end_latitude", 
                 lon1 = "start_longitude", 
                 lon2 = "end_longitude",
                 distance = "haversine"):
        self.lat1 = lat1
        self.lat2 = lat2
        self.lon1 = lon1
        self.lon2 = lon2
        self.distance = distance
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        if self.distance == "haversine":
            X_["distance"] = get_haversine_distance(X_[self.lat1],
                                                X_[self.lat2],
                                                X_[self.lon1],
                                                X_[self.lon2])
        elif self.distance == "euclidian":
            X_["distance"] = get_euclidian_distance(X_[self.lat1],
                                                X_[self.lat2],
                                                X_[self.lon1],
                                                X_[self.lon2])
        elif self.distance == "manhattan":
            X_["distance"] = get_manhattan_distance(X_[self.lat1],
                                                X_[self.lat2],
                                                X_[self.lon1],
                                                X_[self.lon2])
                            
        return X_[['distance']]

class AddGeohash(BaseEstimator, TransformerMixin):

    def __init__(self,  
                 lat = "latitude", 
                 lon = "longitude",
                 precision=6):
        self.lat = lat
        self.lon = lon
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['geohash'] = X.apply(
            lambda x: gh.encode(x[self.lat], x[self.lon], precision=self.precision), axis=1)
        return X[['geohash']]