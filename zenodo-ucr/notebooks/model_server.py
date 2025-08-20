import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import time
import traceback
import warnings
from flask import Flask, request, jsonify
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import ConvLSTM1D
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from threading import Lock, Thread

print(tf.__version__)

# Parse arguments for device and memory configuration
parser = argparse.ArgumentParser(
    description="Run model server with configurable GPU settings")
parser.add_argument("--port", type=int, required=True,
                    help="Port to run the server on")
parser.add_argument("--gpu_device", type=int, default=0,
                    help="GPU device to use (e.g., 0 or 1)")
parser.add_argument("--memory_limit", type=int, default=1090,
                    help="Memory limit for GPU in MiB (default: 1090)")
args = parser.parse_args()

# Set CUDA environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)

tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.enable_eager_execution()

warnings.filterwarnings('ignore')
ops.logging.set_verbosity(ops.logging.ERROR)
tf.get_logger().setLevel('FATAL')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize GPU memory allocation
gpus = tf.config.list_physical_devices("GPU")
allocated_memory = args.memory_limit
print("gpus:", gpus, ", allocated_memory:", allocated_memory)
if gpus:
    try:
        gpu_to_limit = gpus[0]  # Use the first GPU from the visible devices
        tf.config.set_logical_device_configuration(
            gpu_to_limit,
            [tf.config.LogicalDeviceConfiguration(
                memory_limit=allocated_memory)]
        )
    except RuntimeError as e:
        print("Failed to allocate GPU memory:", e)
        sys.exit("Memory allocation unsuccessful. Terminating process.")
else:
    sys.exit("No GPU available. Terminating process.")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Allocated GPU memory:", allocated_memory, "MiB")

# Define paths and parameters
# data_path = "shared/UCI-Benchmark/"
# batch_size = 64

app = Flask(__name__)
global_model = None
current_model_path = None
is_busy = False
busy_lock = Lock()
last_access_time = time.time()
INACTIVITY_TIMEOUT = 120


def release_if_inactive():
    """Release the server if it has been inactive for a specified timeout."""
    global is_busy, last_access_time
    while True:
        time.sleep(5)  # Sprawdza co 5 sekund
        with busy_lock:
            if is_busy and (
                    time.time() - last_access_time) > INACTIVITY_TIMEOUT:
                print(
                    f"Releasing server due to inactivity at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                is_busy = False


def load_model(model_path: str):
    """
    Loads the model from SavedModel format if it exists, otherwise from h5 and converts it.

    :param model_path: The base name of the model (without extension).
    :return: The loaded model.
    """
    global global_model, current_model_path

    model_tf_path = os.path.join(model_path, "model_tf")
    model_h5_path = os.path.join(model_path, "model.h5")

    if os.path.isdir(model_tf_path):
        print(f"Loading model from SavedModel format at {model_tf_path}")
        global_model = tf.keras.models.load_model(model_tf_path)
    elif os.path.isfile(model_h5_path):
        print(
            f"Loading model in h5 format from {model_h5_path} and converting to SavedModel.")
        temp_model = tf.keras.models.load_model(
            model_h5_path, custom_objects={'GlorotUniform': glorot_uniform,
                                           'ConvLSTM1D': ConvLSTM1D}
        )
        temp_model.save(model_tf_path, save_format='tf')
        del temp_model
        print(
            f"Model converted and saved in SavedModel format at {model_tf_path}")
        global_model = tf.keras.models.load_model(model_tf_path)
    else:
        raise FileNotFoundError(
            f"Neither {model_tf_path} directory nor {model_h5_path} file was found.")

    current_model_path = model_path
    print(f"New model loaded from {current_model_path}.")

    return global_model


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions with the current model."""
    global global_model, last_access_time

    if global_model is None:
        return jsonify({"error": "No model is currently loaded"}), 500

    last_access_time = time.time()

    data = request.json.get("data")
    if data is None:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Convert input data to a NumPy array
        data = np.array(data)

        # Perform prediction using the loaded model
        predictions = global_model.predict(data).tolist()

        return jsonify({"predictions": predictions})

    except Exception as e:
        # Capture the full stack trace in case of an error
        error_trace = traceback.format_exc()
        print("Prediction error:",
              error_trace)  # Log the full error trace on the server side for debugging
        return jsonify(
            {"error": f"Prediction failed: {str(e)}", "data": data.shape,
             "trace": error_trace}), 500

    finally:
        # ops.reset_default_graph()
        pass


@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Endpoint to load a model from a specified path."""
    model_path = request.json.get("model_path")
    if not model_path:
        return jsonify(
            {"error": f"Invalid model name provided {model_path}"}), 400

    try:
        load_model(model_path)
        return jsonify({"message": f"Model loaded from {model_path}"}), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/current_model_path', methods=['GET'])
def get_current_model_path():
    """Endpoint to get the current model path."""
    if current_model_path:
        return jsonify(
            {"current_model_path": current_model_path, "message": "OK"}), 200
    else:
        return jsonify({"message": "No model currently loaded"}), 200


@app.route('/allocated_memory', methods=['GET'])
def get_allocated_memory():
    """Endpoint to get the allocated GPU memory."""
    if allocated_memory:
        return jsonify({"allocated_memory_mb": allocated_memory}), 200
    else:
        return jsonify({"message": "Memory allocation unsuccessful"}), 500


@app.route('/set_busy', methods=['POST'])
def set_busy():
    global is_busy
    with busy_lock:
        if not is_busy:
            is_busy = True
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False}), 409


@app.route('/release_busy', methods=['POST'])
def release_busy():
    global is_busy
    with busy_lock:
        is_busy = False
    return jsonify({"success": True}), 200


@app.route('/is_busy', methods=['GET'])
def is_busy_endpoint():
    return jsonify({"is_busy": is_busy}), 200


# Run the server
def run_server(model_server_port):
    app.run(host="0.0.0.0", port=model_server_port, debug=False,
            use_reloader=False, threaded=False)


if __name__ == "__main__":
    inactivity_thread = Thread(target=release_if_inactive, daemon=True)
    inactivity_thread.start()

    run_server(args.port)
