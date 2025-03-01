import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import joblib

# Function to create sequences
def create_sequences(data, target_scaled, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(target_scaled[i + n_steps])
    return np.array(X), np.array(y)

### Trained on AAPL data
# Features --> Open, High, Low, Close, Volume
# Predicts --> Next_Month_Close
# Build the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(300, activation='relu', input_shape=(6, 5), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(200, activation='relu', return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='uniform',activation='relu'))  # Output layer for the next month's close price
    model.add(Dense(1,kernel_initializer='uniform', activation='relu'))
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# Initialize the model
model = build_model()

# Loads the trained weights
model.load_weights('model_weights.weights.h5')

# Load the scalers from the saved file

scaler = joblib.load('scaler.pkl') # For features

CWD = os.getcwd()
aapl_csv_path = os.path.join(CWD, 'AAPL.csv')
df = pd.read_csv(aapl_csv_path)

df['Date'] = pd.to_datetime(df['Date'])
df['YearMonth'] = df['Date'].dt.to_period('M')
# For each month, find the high, low, open, and close values
df['MonthHigh'] = df.groupby('YearMonth')['High'].transform('max')
df['MonthLow'] = df.groupby('YearMonth')['Low'].transform('min')
df['MonthOpen'] = df.groupby('YearMonth')['Open'].transform('first')
df['MonthClose'] = df.groupby('YearMonth')['Close'].transform('last')

# Group by the month and year, then aggregate
monthly_df = df.groupby(df['Date'].dt.to_period('M')).agg({
    'Open': 'first',    # The first Open price of the month
    'High': 'max',      # The highest price (High) of the month
    'Low': 'min',       # The lowest price (Low) of the month
    'Close': 'last',    # The last Close price of the month
    'Volume': 'sum'
})

# Shift the Close column to create a new feature for the next month's Close
monthly_df['Next_Month_Close'] = monthly_df['Close'].shift(-1)
monthly_df

# Assuming the data has been preprocessed and is available in monthly_df
# Sort by Date
monthly_df.sort_values('Date', ascending=True, inplace=True)

# Feature columns: All columns except for 'Next_Month_Close'
features = monthly_df.drop(columns=['Next_Month_Close']).values
features_scaled = scaler.transform(features.reshape(-1,5))

# Target column: 'Next_Month_Close'
target = monthly_df['Next_Month_Close'].values
target_scaler = joblib.load('target_scaler.pkl') # For predicted Next Month Close
# target_scaled = target_scaler.transform(target.reshape(-1,1))

n_steps = 6
X, y = create_sequences(features_scaled, target, n_steps)

# Reshape input to be 3D: [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Split the data into training and test sets (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

predictions = []
prediction = model.predict(X_test[-2:-1])
predictions.append(prediction[0][0])
print(predictions)
predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
print(predictions)