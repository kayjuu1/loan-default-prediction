# ffnn_model/ffnn.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def train_ffnn_model():
    # Load your custom dataset
    data = pd.read_csv('../data/lending_data.csv')

    # Assuming the last column is the target variable and the rest are features
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the FFNN model
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Assuming binary classification
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the model using joblib
    joblib.dump(model, '../_ffnn_model.pkl')

    print("Model trained and saved as _ffnn_model.pkl")


if __name__ == "__main__":
    train_ffnn_model()
