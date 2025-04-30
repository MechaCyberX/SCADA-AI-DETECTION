import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers  import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/processed_data.csv")

X = data.drop(columns=["Label"]).values
Y = data["Label"].values

X = X.reshape((X.shape[0], 1, X.shape[1]))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_test, Y_test))

model.save("models/lstm_scada_model.keras")
print(" LSTM Model Trained and Saved!")