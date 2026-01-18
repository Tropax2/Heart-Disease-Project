from __future__ import annotations 

from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
# Define the numerical and categorical predictors
Numerical_predictors = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_predictors = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"] 
# Builds the preprocessor for the model by changin the types of the predictors 
# and applying the transformations to the data 
def build_preprocessor():
    """
    Returns a transformer that:
        - scales numerical predictors 
        - one-hot encodes categorical predictors  
    """
    numeric = StandardScaler() 
    enc = OneHotEncoder(drop = "first", handle_unknown = "ignore")

    return ColumnTransformer(
        transformers = [
            ("num", numeric, Numerical_predictors),
            ("cat", enc, categorical_predictors)
        ],
        remainder = "drop",
    )
    

