"""
Collection of helper functions
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import *

def create_X_and_y(data) -> tuple:
        y = data['alter']
        X = data.drop(columns='alter')
        
        return X, y
    
def tukey_anscombe(y, residuals):
    sns.set_theme(style="whitegrid")
    n_classes = len(set(y))
    plt.scatter(y, residuals)
    plt.title("Tukey-Anscombe")
    plt.ylabel("Resiudals")
    plt.xlabel("Classes")
    plt.xticks(range(n_classes))
    plt.hlines(0, 0, n_classes-1, colors='r')
    
def plot_feature_importance_gini(model, df, max_display=999):
    fmap = df.columns.values.tolist()
    importances = model.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    importances = np.sort(importances)[::-1]

    importances = importances[:max_display]
    indices = indices[:max_display]
    shape = len(importances)

    g = plt.figure()
    plt.title("Feature importances")
    plt.bar(range(shape), importances, color="r", align="center")
    plt.xticks(range(shape), [fmap[i] for i in indices], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Gini Impurity")
    plt.xlim([-1,shape])
    plt.show()
    
def one_hot(df, feature):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(df[[feature]]).toarray())
    df = df.join(enc_df)
    df.drop(feature, inplace=True, axis=1)
    return df
    