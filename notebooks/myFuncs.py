# Programmer: Connor Fricke
# File: myFuncs.py
# Latest Revision: 1-Nov-2024
# Desc: Helper functions written for Physics 5680 Final Project
# NFL Data Analysis & Predictive Modeling

import pandas as pd
import numpy as np


def nanalysis(df: pd.DataFrame, show=False) -> dict:
    """
    formatted output of numeric features and the number of NaNs detected in
    the dataframe for each feature. Useful for deciding how to handle NaNs
    for different features. NaN analysis = NaNalysis
    """
    # get numeric columns as a subset
    df_numeric = df.select_dtypes(include='number')
    # let's find the NaN values in the numeric columns
    output = {}
    features_left = 0   # track features remaining with NaNs
    if show:
        print("Feature   --->   NaNs")
    for f in df_numeric.columns.tolist():
        nan_count = np.sum(np.isnan(df[f]))
        if show and nan_count > 0:
            print(f"{f}   --->   {nan_count}")
            features_left += 1
        output[f] = nan_count
    if not features_left:
        print("No NaNs remaining in numeric data! Well Done!")
    return output
