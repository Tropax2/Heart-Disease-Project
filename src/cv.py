import numpy as np
import pandas as pd
from src.data import load_dataframe
from src.features import build_preprocessor 
from sklearn.model_selection import cross_validate 
from src.models.lda import build_model as LDA 
from src.models.qda import build_model as QDA 
from src.models.knn import build_model as KNN 
from src.models.naive_bayes import build_model as NB 
from src.models.logistic_regression import build_model as LR  
from sklearn.pipeline import Pipeline
import argparse 

def main():
    # define a parser for the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = str, required = True, help = "Path to the CSV file") 
    args = parser.parse_args() 
    # load the data 
    df = load_dataframe(args.data)
    # drop the target column and create the X and Y variables
    X = df.drop(columns = ["target"])
    Y = df["target"]

    # build the preprocessor for the model
    preprocessor = build_preprocessor()
    # define the models to be testes in the cross-validation
    models = {
        "LDA": LDA(),
        "QDA": QDA(),
        "KNN": KNN(n_neighbors=5),
        "NB": NB(),
        "LR": LR()
    }

    results = []
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        # perform the cross-validation with 5 folds
        score = cross_validate(pipe, X, Y, cv=5)
        results.append({
            "name":name, 
            "test score": round(np.mean(score["test_score"]), 2),
        }) 

        results_df = pd.DataFrame(results).sort_values(by="test score", ascending = False)

    print(results_df)

    results_df.to_csv("results/cv_results.csv", index = False)
if __name__ == "__main__":
    main()
