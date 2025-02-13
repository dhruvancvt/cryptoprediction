import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
file_path = "bitcoin_historical_data.csv"  # Update the path if needed
df = pd.read_csv(file_path)

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Plot Bitcoin price trend
plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index, y=df['price'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price (USD)')
plt.title('Bitcoin Price Over Time')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Check for stationarity using Augmented Dickey-Fuller test
adf_test = adfuller(df['price'])
print(f"ADF Statistic: {adf_test[0]}")
print(f"P-Value: {adf_test[1]}")
if adf_test[1] > 0.05:
    print("Time series is not stationary. Applying differencing.")
    df['price_diff'] = df['price'].diff().dropna()
else:
    print("Time series is stationary.")

# Re-run the ADF test on differenced data if needed
if 'price_diff' in df.columns:
    adf_test_diff = adfuller(df['price_diff'].dropna())
    print(f"ADF Statistic (Differenced): {adf_test_diff[0]}")
    print(f"P-Value (Differenced): {adf_test_diff[1]}")

# Fit ARIMA model (using p=1, d=1, q=1)
model = SARIMAX(df['price'], order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
arima_result = model.fit()

# Forecast the next 7 days
forecast_steps = 7
forecast = arima_result.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='D')[1:]
forecast_values = forecast.predicted_mean

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index, y=df['price'], label='Actual Price', marker='o', linestyle='-')
sns.lineplot(x=forecast_index, y=forecast_values, label='Forecasted Price', marker='o', linestyle='dashed', color='red')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price (USD)')
plt.title('Bitcoin Price Forecast (ARIMA Model)')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.show()

# Print model summary
print(arima_result.summary())
