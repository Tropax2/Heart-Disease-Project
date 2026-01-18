from __future__ import annotations 

import pandas as pd 
from sklearn.model_selection import train_test_split 

response = "target"
# Loads the csv file as a pandas dataframe and drops any rows with missing values
def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna()

    return df 
# split the dataframe into training and testing sets for a validation set approach
def make_splits(
    df: pd.DataFrame,
    target_column: str,
    test_size = 0.15,
    random_state = 42
): 
    X = df.drop(columns = [target_column])
    Y = df[target_column]

    return train_test_split(
        X,
        Y, 
        test_size = test_size,
        random_state = random_state,
        shuffle = True
    )
