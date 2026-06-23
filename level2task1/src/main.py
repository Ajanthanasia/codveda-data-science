import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# data set columns
# columns = ['Crime rate',
#           'Residential land zoning',
#           'Industrial area proportion',
#           'Charles River (0 or 1)',
#           'Nitric oxide concentration',
#           'Average rooms',
#           'Old house proportion',
#           'Distance to workplaces',
#           'Highway access',
#           'Property tax',
#           'Student-teacher ratio',
#           'Black population statistic',
#           'Lower status population %',
#           'House price']

# Column names
data = pd.read_csv('house_data.csv', sep=r"\s+", header=None)

data.columns = [
    'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
    'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'
]

# Check missing values
# print(data.isnull().sum())

# Separate features and target
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

# Fill missing values using mean
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted Values:", y_pred)
print("MSE:", mse)
print("R2 Score:", r2)