"""
Collection of helper functions
"""
import pandas as pd

def create_X_and_y(data) -> tuple:
        y = data['alter']
        X = data.drop(columns='alter')
        
        return X, y