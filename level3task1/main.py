import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt


def load_data(filename='data.csv'):
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df.sort_values('Date').set_index('Date')
    if df.index.freq is None:
        freq = pd.infer_freq(df.index)
        if freq is None:
            df = df.asfreq('MS')
        else:
            df = df.asfreq(freq)
    return df


def plot_series(series, title='Time Series'):
    plt.figure(figsize=(10, 5))
    plt.plot(series, marker='o')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(series.name)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def decompose_series(series, period=12):
    result = seasonal_decompose(series, model='additive', period=period)
    result.plot()
    plt.tight_layout()
    plt.show()
    return result


def moving_average(series, window=12):
    return series.rolling(window=window, center=True).mean()


def exponential_smoothing(series, seasonal_periods=12):
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_periods
    )
    fit = model.fit()
    return fit


def evaluate_forecast(actual, predicted):
    rmse = sqrt(mean_squared_error(actual, predicted))
    return rmse


def fit_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit = model.fit(disp=False)
    return fit


def plot_forecast(train, test, forecast, title='Forecast vs Actual'):
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Actual')
    plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(train.name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_menu():
    print('\nSelect the chart or analysis you want:')
    print('1. Sales over time')
    print('2. Time series decomposition')
    print('3. Moving average')
    print('4. Exponential smoothing forecast')
    print('5. SARIMA forecast vs actual')
    print('6. SARIMA future forecast')
    print('7. Show data summary')
    print('0. Exit')


def show_data_summary(series):
    print('\nData preview:')
    print(series.head())
    print('\nData frequency:', series.index.freq)
    print('\nMissing values:')
    print(series.isna().sum())


def show_moving_average(series, window=12):
    ma = moving_average(series, window=window)
    plt.figure(figsize=(10, 5))
    plt.plot(series, label='Original')
    plt.plot(ma, label=f'{window}-Period Moving Average', color='orange')
    plt.title('Sales and Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_exponential_smoothing(series, seasonal_periods=12):
    exp_fit = exponential_smoothing(series, seasonal_periods=seasonal_periods)
    exp_forecast = exp_fit.forecast(steps=12)
    plt.figure(figsize=(10, 5))
    plt.plot(series, label='Original')
    plt.plot(exp_fit.fittedvalues, label='Holt-Winters Fit')
    plt.plot(exp_forecast.index, exp_forecast, label='Holt-Winters Forecast', linestyle='--')
    plt.title('Exponential Smoothing Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def prepare_sarima(series, test_size=None):
    if test_size is None:
        test_size = min(12, len(series) // 5)
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    sarima_fit = fit_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_forecast = sarima_fit.get_forecast(steps=test_size)
    sarima_pred = sarima_forecast.predicted_mean
    rmse = evaluate_forecast(test, sarima_pred)
    return train, test, sarima_fit, sarima_pred, rmse


def show_sarima_forecast(train, test, sarima_pred):
    plot_forecast(train, test, sarima_pred, title='SARIMA Forecast vs Actual')


def show_future_forecast(series, sarima_fit, periods=12):
    full_forecast = sarima_fit.get_forecast(steps=periods).predicted_mean
    future_index = pd.date_range(start=series.index[-1] + pd.offsets.DateOffset(months=1), periods=periods, freq=series.index.freq)
    full_forecast.index = future_index
    plt.figure(figsize=(10, 5))
    plt.plot(series, label='Historical Sales')
    plt.plot(full_forecast, label='SARIMA 12-Month Forecast', linestyle='--')
    plt.title('SARIMA Forecast for Next 12 Periods')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print('\nForecast summary:')
    print(full_forecast)


def main():
    df = load_data('data.csv')
    series = df['Sales']
    train = None
    test = None
    sarima_fit = None
    sarima_pred = None
    rmse = None

    show_data_summary(series)

    while True:
        show_menu()
        choice = input('Enter option number: ').strip()
        if choice == '0':
            print('Exiting.')
            break
        elif choice == '1':
            plot_series(series, title='Sales Over Time')
        elif choice == '2':
            decompose_series(series, period=12)
        elif choice == '3':
            show_moving_average(series, window=12)
        elif choice == '4':
            show_exponential_smoothing(series, seasonal_periods=12)
        elif choice == '5':
            train, test, sarima_fit, sarima_pred, rmse = prepare_sarima(series)
            print(f'\nSARIMA RMSE: {rmse:.4f}')
            show_sarima_forecast(train, test, sarima_pred)
        elif choice == '6':
            if sarima_fit is None:
                train, test, sarima_fit, sarima_pred, rmse = prepare_sarima(series)
            show_future_forecast(series, sarima_fit, periods=12)
        elif choice == '7':
            show_data_summary(series)
        else:
            print('Invalid choice. Please select a valid option.')


if __name__ == '__main__':
    main()