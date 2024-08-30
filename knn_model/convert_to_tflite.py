import pandas as pd
import tensorflow as tf
import joblib


class FFNNModel(tf.Module):
    def __init__(self, model_path, input_shape):
        # Load the trained FFNN model from the provided path
        self.model = joblib.load(model_path)
        self.input_shape = input_shape

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 7], dtype=tf.float32)])  # Adjust shape based on your data
    def predict(self, X):
        # Perform the forward pass using the loaded FFNN model
        logits = self.model(X)
        predictions = tf.argmax(logits, axis=-1)
        return predictions


def convert_ffnn_to_tflite():
    # Load data (same as used in training)
    data = pd.read_csv('../data/lending_data.csv')

    # Assuming the last column is the target and the rest are features
    X = data.iloc[:, :-1].values  # Features (all columns except the last)

    # Initialize the custom FFNN TensorFlow model
    ffnn_tf_model = FFNNModel(model_path='../_ffnn_model.pkl', input_shape=(X.shape[1],))

    # Convert the model to TensorFlow Lite
    concrete_func = ffnn_tf_model.predict.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open('../ffnn_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("FFNN model converted to TensorFlow Lite and saved as ffnn_model.tflite")


if __name__ == "__main__":
    convert_ffnn_to_tflite()
