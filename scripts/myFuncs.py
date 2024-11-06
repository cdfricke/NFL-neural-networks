# Programmer: Connor Fricke
# File: myFuncs.py
# Latest Revision: 1-Nov-2024
# Desc: Helper functions written for Physics 5680 Final Project
# NFL Data Analysis & Predictive Modeling

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


def get_image(df: pd.DataFrame, game_id: int, columns: list):
    """
    Returns a 2D numpy array of "pixel values" that correspond to the image representing
    a single game (game_id) from the dataset. The columns to be selected for use in the
    image should be pre-scaled and should be listed with the columns parameter
    """
    df_game = df[df['game_id'] == int(game_id)]
    df_game_cols = df_game[columns]
    return df_game_cols.to_numpy()