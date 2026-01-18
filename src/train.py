from __future__ import annotations 

import argparse
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data import load_dataframe, make_splits 
from src.features import build_preprocessor 
from src.models.knn import build_model

def main():
    # define a parser for the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = str, required = True, help = "Path to the CSV file") 
    parser.add_argument("--test-size", type = float, default = 0.15)
    parser.add_argument("--seed", type = int, default = 42)
    args = parser.parse_args()
    # load the data and split it into training and testing sets
    df = load_dataframe(args.data) 
    X_train, X_test, Y_train, Y_test = make_splits(df, target_column="target", test_size = args.test_size, random_state = args.seed)
    # build the pipeline for the model
    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", build_model(5))
    ])
    # fit the pipeline to the training data
    pipe.fit(X_train, Y_train)
    # predict the testing data
    preds = pipe.predict(X_test)
    # calculate the accuracy and confusion matrix
    acc = accuracy_score(Y_test, preds) 
    cm = confusion_matrix(Y_test, preds) 
    print(f"Confusion Matrix: \n {cm}")
    print(f"Accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()
