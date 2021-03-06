import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

main_file_path  = '../input/train.csv' # this is the path to the Iowa data
iowa_data = pd.read_csv(main_file_path)

#iowa_target = iowa_data.SalePrice

# this detect which cells have missing values, and then count
# how many there are in each column
print(iowa_data.isnull().sum())

# this drop columns with missing values
#data_without_missing_values = iowa_data.dropna(axis=1)

# In many cases, you'll have both a training dataset and a test dataset.
# You will want to drop the same columns in both DataFrames
#cols_with_missing = [col for col in iowa_data.columns 
#                            if iowa_data[col].isnull().any()]

#reduced_iowa_data = iowa_data.drop(cols_with_missing, axis=1)
#reduced_test_data = test_data.drop(cols_with_missing, axis=1)

#-------------------------------------------------------------------------------

# Imputation fills in the missing value with some number.
# The imputed value won't be exactly right in most cases,
# but it usually gives more accurate models than dropping the column entirely.
# The default behavior fills in the mean value for imputation
#my_imputer = Imputer()
#data_with_imputed_values = my_imputer.fit_transform(iowa_data)

#-------------------------------------------------------------------------------

# This approach makes the imputations predictions by considering which
# values were originally missing.

# make copy to avoid changing original data (when Imputing)
new_data = iowa_data.copy()

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns 
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

# Imputation
my_imputer = Imputer()
new_data = my_imputer.fit_transform(new_data)