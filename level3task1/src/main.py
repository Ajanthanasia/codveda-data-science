import pandas
import numpy
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error
from math import sqrt

df = pandas.read_csv("data.csv")

df["Date"] = pandas.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

plt.figure(figsize=(10,5))
plt.plot(df["Sales"])
plt.title("Sales Over Time")
plt.show()

result = seasonal_decompose(df["Sales"], model="additive", period=12)

result.plot()
plt.show()

df["Moving_Avg"] = df["Sales"].rolling(window=3).mean()

plt.figure(figsize=(10,5))
plt.plot(df["Sales"], label="Original")
plt.plot(df["Moving_Avg"], label="Moving Average")
plt.legend()
plt.show()

model = ExponentialSmoothing(df["Sales"])

fit = model.fit()

forecast = fit.forecast(10)

print(forecast)

plt.plot(df["Sales"])
plt.plot(forecast)
plt.show()

train = df["Sales"][:-10]
test = df["Sales"][-10:]

model = ARIMA(train, order=(1,1,1))

model_fit = model.fit()

predictions = model_fit.forecast(steps=10)

rmse = sqrt(mean_squared_error(test, predictions))

print("RMSE:", rmse)

plt.figure(figsize=(10,5))

plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, predictions, label="Prediction")

plt.legend()
plt.show()