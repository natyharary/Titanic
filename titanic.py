import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load data
X = pd.read_csv("C:\\Users\\natyh\\PycharmProjects\\Kaggle_Titanic\\data\\train.csv")
y = X['Survived']
X_test_original = pd.read_csv("C:\\Users\\natyh\\PycharmProjects\\Kaggle_Titanic\\data\\test.csv")
X_test = X_test_original
X_train, X_valid, y_train_dummy, y_valid_dummy = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Dropping non vital columns
X_train = X_train.drop('Name', axis=1)
X_train = X_train.drop('Ticket', axis=1)
X_train = X_train.drop('Cabin', axis=1)

X_valid = X_valid.drop('Name', axis=1)
X_valid = X_valid.drop('Ticket', axis=1)
X_valid = X_valid.drop('Cabin', axis=1)

X_test = X_test.drop('Name', axis=1)
X_test = X_test.drop('Ticket', axis=1)
X_test = X_test.drop('Cabin', axis=1)

# Treating NaN values
X_train = X_train.dropna(subset=['Embarked'])  # Drop the few rows with no place of embankment
X_train['Age'].fillna((X_train['Age'].mean()), inplace=True)  # Imputing missing age with mean value

X_valid = X_valid.dropna(subset=['Embarked'])  # Drop the few rows with no place of embankment
X_valid['Age'].fillna((X_valid['Age'].mean()), inplace=True)  # Imputing missing age with mean value

# Separating data into different columns
numerical_cols = [col for col in X_train.columns if
                  X_train[col].dtype in ['int64', 'float64']]
categorical_cols = [col for col in X_train.columns
                    if X_train[col].nunique() < 10 and
                    X_train[col].dtype == "object"]

# One hot encoding categorical data
X_train = pd.get_dummies(X_train, prefix_sep="__",
                         columns=categorical_cols)
X_valid = pd.get_dummies(X_valid, prefix_sep="__",
                         columns=categorical_cols)
X_test = pd.get_dummies(X_test, prefix_sep="__",
                        columns=categorical_cols)

# Get y
y_train = X_train['Survived']
X_train = X_train.drop('Survived', axis=1)

y_valid = X_valid['Survived']
X_valid = X_valid.drop('Survived', axis=1)

# Define and fit model
model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

submission = pd.DataFrame({'PassengerId': X_test['PassengerId'],
                           'Survived': y_pred})
print(submission.shape)
submission.to_csv("submission.csv", index=False)
