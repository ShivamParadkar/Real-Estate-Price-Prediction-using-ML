from joblib import dump, load
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

housing = pd.read_csv("data.csv")

# print(housing.info())
# print(housing['CHAS'])
# print(housing['CHAS'].value_counts())
# print(housing.describe())

# For plotting histogram
# housing.hist(bins=50, figsize=(20,15))

# -------------******* TRAIN TEST SPLITTING *****----//


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# train_set, test_set = split_train_test(housing, 0.2)
# print(f"Rows in train set: {len(train_set)} \n Rows in test set: {len(test_set)}\n")


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(f"Rows in train set: {len(train_set)} \nRows in test set: {len(test_set)}\n")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print(strat_test_set['CHAS'].value_counts())
# print(strat_train_set['CHAS'].value_counts())

housing = strat_train_set.copy()


#-------***** Looking for Correlations *****-------#

corr_matrix = housing.corr()
# print(corr_matrix['MEDV'].sort_values(ascending=False))

attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)

#------**** Trying out other combinations ****-----#

housing["TAXRM"] = housing["TAX"]/housing["RM"]
# print(housing.head())

corr_matrix = housing.corr()
# print(corr_matrix['MEDV'].sort_values(ascending=False))

housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)

housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

#----***** Missing Attributes ****----#

# To take care of missing attributes three options are there:
#     1.Get rid of the missing data points
#     2.Get rid of the whole attributes
#     3.set the value of some value(0, mean or median)

a = housing.dropna(subset=["RM"])  # option 1
# print(a.shape)
# note that the original housing dataframe will remain unchanged

# print(housing.drop("RM",axis=1).shape) #option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged

median = housing["RM"].median()  # compute median for option 3

# print(housing["RM"].fillna(median))  # option 3

# print(housing.shape)
# print(housing.describe()) #before we started filling missing attributes

imputer = SimpleImputer(strategy="median")
imputer.fit(housing)

# print(imputer.statistics_)

X = imputer.transform(housing)

housing_tr = pd.DataFrame(X, columns=housing.columns)

# print(housing_tr.describe())


#-------******** Scikit-learn Design ******------#
# Primarily three types of objects
# 1.Estimators - It estimates some parameters based on a database. Eg. Imputer
# It has a fit method and transform data
# Fit method - Fits the dataset and calculates internal parameters

# 2.Transformers - transform method takes input and returns output based on the learnings from fit().
# It also has a convienince function called fit_transform() which fits and then transform.

# 3.Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions.


#----*** Feature Scaling ***---#
# Primarily, two types of features scaling methods:
# 1.Min-max scaling (Normalization)
#     (value - min)/(max-min)
#     sklearn provides a class called MinMaxScaler for this

# 2.Standardization
#     (value - mean)/std
#     sklearn provides a class called Standard Scaler for this


#----**** Creatinng a pipeline ***---#
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #  ......add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing)

# print(housing_num_tr.shape)


#-----***** Selecting a desired model for real estate space *****----#

# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

prepared_data = my_pipeline.transform(some_data)

# print(model.predict(prepared_data))

# print(list(some_labels))


#--------****** Evaluating the model ****-----#

housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

# print(rmse)


#-----**** Using better evaluation technique cross validation ******-----#

scores = cross_val_score(
    model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

# print(rmse_scores)


def print_scores(scores):
    print("scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


print(print_scores(rmse_scores))

#-------*** Saving the Model *****----#

dump(model, 'RealESpace.joblib')

#----*** Testing the model on test data ****----#
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))

print(final_rmse)

print(prepared_data[3])


#-------****** MOdel usage for prediction of new values *******---------#

model = load('RealESpace.joblib')

features = np.array([[-0.42292925, 14.4898311, -0.57719868, -0.27288841, -0.5573845,
                      0.15283383, -0.52225911,  0.37882487, 8.5429938, -0.74402708,
                      0.52982668,  0.45343469, 3.81939807], ])
print(model.predict(features))
