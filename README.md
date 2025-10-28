# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 19/10/2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the Netflix stock dataset
file_path = "C:\\Users\\admin\\Downloads\\Netflix_stock_data.csv"  # replace with actual path if needed
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
data.set_index('Date', inplace=True)

# Use 'Close' price for analysis
close_prices = data['Close']

# Resample to weekly frequency (take mean of each week)
weekly_close = close_prices.resample('W').mean()

# Perform Augmented Dickey-Fuller test for stationarity
result = adfuller(weekly_close.dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] < 0.05:
    print("The data is stationary.")
else:
    print("The data is non-stationary.")

# Split into train and test sets (80% training, 20% testing)
train_size = int(len(weekly_close) * 0.8)
train, test = weekly_close[:train_size], weekly_close[train_size:]

# Plot ACF and PACF
fig, ax = plt.subplots(2, figsize=(8, 6))
plot_acf(train.dropna(), ax=ax[0], title='Autocorrelation Function (ACF)')
plot_pacf(train.dropna(), ax=ax[1], title='Partial Autocorrelation Function (PACF)')
plt.show()

# Fit AR model with 13 lags
ar_model = AutoReg(train.dropna(), lags=13).fit()

# Make predictions on test set
ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot predictions vs test data
plt.figure(figsize=(10, 4))
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('AR Model Prediction vs Test Data (Netflix Close Price)')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Calculate Mean Squared Error
mse = mean_squared_error(test, ar_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Plot full data: Train, Test, Predictions
plt.figure(figsize=(10, 4))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('Train, Test, and AR Model Prediction (Netflix Close Price)')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()

```
### OUTPUT:


PACF - ACF

<img width="919" height="326" alt="image" src="https://github.com/user-attachments/assets/26ef7bac-4fa1-4cab-996b-5c73357655d5" />

<img width="924" height="332" alt="image" src="https://github.com/user-attachments/assets/b7c0494c-f6e2-43d4-8268-82b37f13d7f4" />


PREDICTION

<img width="1110" height="582" alt="image" src="https://github.com/user-attachments/assets/70f8a274-9b37-4d94-a73b-a1027cb81609" />


FINIAL PREDICTION

<img width="1199" height="514" alt="image" src="https://github.com/user-attachments/assets/0b0c5349-a239-4326-baa5-6d3cd262a8c5" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
