import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

# Load data
X = pd.read_csv(".\\data\\train.csv")
y = X['Survived']
X_test = pd.read_csv(".\\data\\test.csv")
X_train, X_valid, y_train_dummy, y_valid_dummy = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                  random_state=np.random)


def drop_non_vital_columns(dataframe):
    """
    Dropping non vital columns
    :param dataframe: dataframe, raw data
    :return df: dataframe, reduced dataframe without non-informational columns
    """
    df = dataframe
    df = df.drop('Name', axis=1)
    df = df.drop('Ticket', axis=1)
    df = df.drop('Cabin', axis=1)
    return df


def treat_nan_values(dataframe):
    """
    Treating NaN values. Imputing with mean value
    :param dataframe: dataframe, raw data
    :return df: dataframe, reduced dataframe without nan values
    """
    df = dataframe
    df = df.dropna(subset=['Embarked'])  # Drop the few rows with no place of embankment
    df['Age'].fillna((dataframe['Age'].mean()), inplace=True)  # Imputing missing age with mean value
    return df


def one_hot_encoding(dataframe):
    """
    One hot encoding for categorical columns in the dataframe
    :param dataframe: raw data
    :return df: dataframe with onehotencoding
    """
    df = dataframe
    # Separating data into different columns
    numerical_cols = [col for col in X_train.columns if
                      X_train[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in X_train.columns
                        if X_train[col].nunique() < 10 and
                        X_train[col].dtype == "object"]

    # One hot encoding categorical data
    df = pd.get_dummies(df, prefix_sep="__",
                        columns=categorical_cols)
    return df


def modify_data(dataframe):
    """
    Apply all data manipulation functions on a dataframe
    :param dataframe: raw data
    :return df: manipulated dataframe
    """
    df = drop_non_vital_columns(dataframe)
    df = treat_nan_values(df)
    df = one_hot_encoding(df)
    return df


def get_y(dataframe):
    """
    :param dataframe: raw data
    :return y: y column
            dataframe_modified: dataframe without y column
    """
    y = dataframe['Survived']
    dataframe_modified = dataframe.drop('Survived', axis=1)
    return y, dataframe_modified

X_train_modified = modify_data(X_train)
X_valid_modified = modify_data(X_valid)
X_test_modified = modify_data(X_test)

# Get y
y_train, X_train_modified_no_y = get_y(X_train_modified)
y_valid, X_valid_modified_no_y = get_y(X_valid_modified)

# Define and fit model
model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.03)
model.fit(X_train_modified_no_y, y_train)

# TODO make cross validation, and graph the iterations with sns
# Checking accuracy
y_pred = model.predict(X_test_modified)
y_pred_valid = model.predict(X_valid_modified_no_y)
# TODO maybe remove MAE
# mae = mean_absolute_error(y_valid, y_pred_valid)
# print("Mean absolute error for validity is", mae)

print(confusion_matrix(y_valid, y_pred_valid))
print(classification_report(y_valid, y_pred_valid))
acc = accuracy_score(y_valid, y_pred_valid)

submission = pd.DataFrame({'PassengerId': X_test_modified['PassengerId'],
                           'Survived': y_pred})
submission.to_csv("submission.csv", index=False)
