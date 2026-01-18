# Heart Disease Prediction 

This project compares several classification models to predict the presence of a heart
disease using clinical data from a Kaggle dataset (https://www.kaggle.com/datasets/arezaei81/heartcsv)

## Dataset 

The dataset contains demographic and clinical measurements such as age, sex, chest pain type, 
blood pressure, cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart
rate achieved and many others. The response variable is the target variable indicating the presence of
a heart disease in the given patient. 

## Methods 

Categorical predictors were one-hot encoded and numerical predictors were standardised. The following 
models are evaluated:
- Logistic Regression;
- Linear Discriminant Analysis (LDA);
- Quadratic Discriminant Analysis (QDA);
- Naive Bayes (NB);
- K-Nearest Neighbourhood.

The models were evaluated by a validation set approach as well as a 5-fold cross-validation.

## Results 

### Validation Set Approach 

By using a validation set approach with a split of 85% training and 15% testing, Logistic Regression 
obtained the best accuracy, followed by QDA and KNN with K = 5.

### 5-fold Cross-Validation Approach 

By using a 5-fold cross-validation approach, LDA obtained the highest mean accuracy among the evaluated
models, followed by Logistic Regression.

Since cross-validation uses more data for training (giving more stable results)and the fact that is the standard approach for model selection, LDA is chosen as the final model for predicting heart disease.

The confusion matrix and accuracy of the validation set approach, as well as the results obtained from
5-fold cross-validation are present in a CSV file in `results/results.csv`.

## Project Structure 

- `src-`- source code for model and evaluations
- `data-`- dataset files (not included in the repo)
- `results-` - cross-validation results