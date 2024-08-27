import pandas as pd
import tensorflow as tf
import joblib


class KNNModel(tf.Module):
    def __init__(self, model_path, X_train, y_train, k=5):
        self.model = joblib.load(model_path)
        self.X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        self.y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
        self.k = k

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 7], dtype=tf.float32)])  # Adjust shape based on your data
    def predict(self, X):
        # Calculate the Euclidean distance
        distances = tf.norm(self.X_train - X[:, None], axis=2)
        knn_indices = tf.argsort(distances, axis=-1)[:, :self.k]
        knn_labels = tf.gather(self.y_train, knn_indices)

        # Find the unique labels and their counts
        unique_labels, _, counts = tf.unique_with_counts(tf.reshape(knn_labels, [-1]))

        # Find the label with the highest count (the mode)
        mode_index = tf.argmax(counts)
        predictions = unique_labels[mode_index]

        return predictions


def convert_knn_to_tflite():
    # Load data (same as used in training)
    data = pd.read_csv('data/lending_data.csv')

    # Assuming the last column is the target and the rest are features
    X = data.iloc[:, :-1].values  # Features (all columns except the last)
    y = data.iloc[:, -1].values  # Target (last column)

    # Initialize the custom KNN TensorFlow model
    knn_tf_model = KNNModel(model_path='knn_model.pkl', X_train=X, y_train=y)

    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([knn_tf_model.predict.get_concrete_function()])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open('knn_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("KNN model converted to TensorFlow Lite and saved as knn_model.tflite")


if __name__ == "__main__":
    convert_knn_to_tflite()
