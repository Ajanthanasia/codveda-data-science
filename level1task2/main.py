import pandas
import matplotlib.pyplot as plt
import seaborn

def exploratoryDataAnalysis(data):
    print(data.head())      # first 5 rows
    print(data.shape)       # rows and columns
    print(data.columns)     # column names
    print(data.info())      # data types
    
    # check missing values
    print(data.isnull().sum())
    
    #summary statistics
    print(data.describe())
    
    print("Mean:")
    print(data.mean(numeric_only=True))

    print("Median:")
    print(data.median(numeric_only=True))

    print("Variance:")
    print(data.var(numeric_only=True))
    
    # data distribution
    # data.hist(figsize=(10,8))
    # plt.show()
    
    #scatter plot
    plt.scatter(data['sepal_length'], data['sepal_width'])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title("Sepal Length vs Sepal Width")
    plt.show()

#Load the data
data = pandas.read_csv('data.csv')
eda = exploratoryDataAnalysis(data)