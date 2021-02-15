import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Visualizer(object):
    def __init__(self):
        pass

    def plot_var(self, explained_variance_ratio):
        """
        This function plot the cumulative explained variance ratio as the number of components increases.
        """
        plt.plot(explained_variance_ratio.cumsum())
        plt.xlabel('number of components')
        plt.ylabel('Cumulative percent of variance')  
        plt.axhline(y=0.8, c="g", linestyle="--", label="80%")
        plt.legend()
        plt.grid()
        plt.show()