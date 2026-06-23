# Exploratory Data Analysis (EDA) on Iris Dataset

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    return pd.read_csv('data.csv')


def show_head(df):
    print("\nFirst 5 Rows:")
    print(df.head())


def show_info(df):
    print("\nDataset Info:")
    print(df.info())


def show_summary(df):
    print("\nSummary Statistics:")
    print(df.describe())


def show_missing(df):
    print("\nMissing Values:")
    print(df.isnull().sum())


def show_histograms(df):
    df.hist(figsize=(10, 8))
    plt.suptitle("Histograms of Iris Features")
    plt.tight_layout()
    plt.show()


def show_scatter(df):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x='sepal_length',
        y='sepal_width',
        hue='species',
        data=df
    )
    plt.title("Sepal Length vs Sepal Width")
    plt.show()


def show_boxplot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df.iloc[:, :-1])
    plt.title("Box Plot of Iris Features")
    plt.show()


def show_correlation(df):
    plt.figure(figsize=(8, 6))
    corr = df.iloc[:, :-1].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()


def show_pairplot(df):
    sns.pairplot(df, hue='species')
    plt.show()


def show_insights():
    print("\nEDA Insights:")
    print("1. Iris dataset contains 150 rows.")
    print("2. No missing values found.")
    print("3. Petal length and petal width are highly correlated.")
    print("4. Setosa species is clearly separated from others.")
    print("5. Virginica flowers generally have larger petal sizes.")


def print_menu():
    print("\nSelect an option:")
    print("1. Show first 5 rows")
    print("2. Show dataset info")
    print("3. Show summary statistics")
    print("4. Show missing values")
    print("5. Show histograms")
    print("6. Show scatter plot")
    print("7. Show box plot")
    print("8. Show correlation matrix")
    print("9. Show pair plot")
    print("10. Show insights")
    print("0. Exit")


def main():
    df = load_data()
    actions = {
        '1': lambda: show_head(df),
        '2': lambda: show_info(df),
        '3': lambda: show_summary(df),
        '4': lambda: show_missing(df),
        '5': lambda: show_histograms(df),
        '6': lambda: show_scatter(df),
        '7': lambda: show_boxplot(df),
        '8': lambda: show_correlation(df),
        '9': lambda: show_pairplot(df),
        '10': show_insights
    }

    while True:
        print_menu()
        choice = input("Enter the option number: ").strip()
        if choice == '0':
            print("Exiting the EDA menu.")
            break
        action = actions.get(choice)
        if action:
            action()
        else:
            print("Invalid choice. Please select a valid option.")


if __name__ == '__main__':
    main()