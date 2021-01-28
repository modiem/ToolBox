import pandas as pd 
from ToolBox.utils import simple_time_tracker

@simple_time_tracker
def get_data(path, nrows=None):
    """method to get the training data (or a portion of it) """
    df = pd.read_csv(path, nrows=nrows)
    return df

if __name__ == '__main__':
    df = get_data()