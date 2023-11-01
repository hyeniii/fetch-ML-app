import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

lookback = 120 # This represents how many previous months we consider

def main():
    ######## CONFIGURE LOGGING ########
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        handlers=[logging.FileHandler('logs/train.log'), logging.StreamHandler()])

    logging.info("Starting script...")

    ######## PREPROCESS ########
    logging.info("Starting data preprocessing...")

    default_filename = "data_daily.csv"
    filename_from_env = os.environ.get('DATA_FILENAME')

    if filename_from_env:
        filename = filename_from_env
    else:
        filename = default_filename

    data = pd.read_csv(f"data/{filename}")
    receipts = data['Receipt_Count'].values.reshape(-1, 1)

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(receipts)

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))


    ######## TRAIN ###########

    model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')


    model.fit(X, y, epochs=50, batch_size=16)

    logging.info("Model training completed successfully")

    model.save('model/lstm_model.h5')

    # Using the last 120 days of 2021 to predict the first day of 2022, then move the window
    test_inputs = scaled_data[len(scaled_data) - lookback:].tolist()
    predictions = []

    for i in range(365):
        test_sequence = np.array(test_inputs[-lookback:])
        test_sequence = np.reshape(test_sequence, (1, lookback, 1))
        predicted_receipts = model.predict(test_sequence)[0][0]
        predictions.append(predicted_receipts)
        test_inputs.append([predicted_receipts])

    # Rescale the predictions back to the original scale
    predicted_receipts_2022 = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    forecast_index = pd.date_range(start='2022-01-01', periods=365)
    predictions_df = pd.DataFrame(predicted_receipts_2022, index=forecast_index, columns=['Predicted_Receipts'])
    monthly_predictions = predictions_df.resample('M').sum()
    monthly_predictions.to_csv('assets/predictions_2022.csv')
    logging.info("saved 2022 predictions as csv ...")


    data['# Date'] = pd.to_datetime(data['# Date'])
    monthly_data_2021 = data.resample('M', on='# Date').sum()


    plt.figure(figsize=(15, 7))

    # Plot actual data from 2021
    plt.plot(monthly_data_2021.index, monthly_data_2021['Receipt_Count'], label='Actual Data (2021)', marker='o', linestyle='-')

    # Plot forecasted data for 2022
    plt.plot(monthly_predictions.index, monthly_predictions['Predicted_Receipts'], color='red', linestyle='dashed', label='Forecast for 2022', marker='x')

    plt.title('Receipt Count: Actual (2021) vs. Forecast (2022)')
    plt.xlabel('Date')
    plt.ylabel('Receipt Count')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("assets/receipts_plot.png")
    logging.info("saved visualization...")

if __name__ == "__main__":
    train = os.environ.get('TRAIN', 'n').strip().lower()
    if train == 'y':
        main()
    else:
        exit()
