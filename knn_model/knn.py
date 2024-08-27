# knn_model/knn.py
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def train_knn_model():
    # Load your custom dataset
    data = pd.read_csv('data/lending_data.csv')

    # Assuming the last column is the target variable and the rest are features
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the KNN model
    model = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'knn_model.pkl')

    print("Model trained and saved as knn_model.pkl")


if __name__ == "__main__":
    train_knn_model()
