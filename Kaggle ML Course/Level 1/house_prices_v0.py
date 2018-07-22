import pandas as pd
from sklearn.tree import DecisionTreeRegressor

main_file_path  = '../input/train.csv' # this is the path to the Iowa data
iowa_data = pd.read_csv(main_file_path)

y = iowa_data.SalePrice # this is the prediction target

iowa_predictors = [
    'LotArea',
    'YearBuilt',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd'
]

X = iowa_data[iowa_predictors] # this is the predictor data

# Define model
iowa_model = DecisionTreeRegressor()

# Fit model
iowa_model.fit(X, y)

# In practice, we'll want to make predictions for new houses coming
# on the market rather than the houses we already have prices for.
# But we'll make predictions for the first rows of the training data
# to see how the predict function works.
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))

# If the predictions are good, this should be the results
# print(y.head())