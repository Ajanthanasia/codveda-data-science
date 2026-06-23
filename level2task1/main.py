import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Column names and data loading
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

data = pd.read_csv('house_data.csv', sep=r"\s+", header=None)
data.columns = column_names

# Separate features and target
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Handle missing values if present
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the models for comparison
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((model_name, mse, r2))
    print(f"{model_name} Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}\n")

# Compare model performance
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R2'])
print("Model Comparison:")
print(results_df)
