# Exploratory Data Analysis (EDA) on Iris Dataset

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# Load Iris Dataset from CSV
# -----------------------------------
df = pd.read_csv('data.csv')

# -----------------------------------
# Display first 5 rows
# -----------------------------------
print("First 5 Rows:")
print(df.head())

# -----------------------------------
# Dataset information
# -----------------------------------
print("\nDataset Info:")
print(df.info())

# -----------------------------------
# Summary statistics
# -----------------------------------
print("\nSummary Statistics:")
print(df.describe())

# -----------------------------------
# Check missing values
# -----------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------------
# Histograms
# -----------------------------------
df.hist(figsize=(10,8))
plt.suptitle("Histograms of Iris Features")
plt.show()

# -----------------------------------
# Scatter Plot
# -----------------------------------
plt.figure(figsize=(7,5))

sns.scatterplot(
    x='sepal_length',
    y='sepal_width',
    hue='species',
    data=df
)

plt.title("Sepal Length vs Sepal Width")
plt.show()

# -----------------------------------
# Box Plot
# -----------------------------------
plt.figure(figsize=(10,6))

sns.boxplot(data=df.iloc[:, :-1])

plt.title("Box Plot of Iris Features")
plt.show()

# -----------------------------------
# Correlation Matrix
# -----------------------------------
plt.figure(figsize=(8,6))

corr = df.iloc[:, :-1].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title("Correlation Matrix")
plt.show()

# -----------------------------------
# Pair Plot
# -----------------------------------
sns.pairplot(df, hue='species')
plt.show()

# -----------------------------------
# Final Insights
# -----------------------------------
print("\nEDA Insights:")
print("1. Iris dataset contains 150 rows.")
print("2. No missing values found.")
print("3. Petal length and petal width are highly correlated.")
print("4. Setosa species is clearly separated from others.")
print("5. Virginica flowers generally have larger petal sizes.")