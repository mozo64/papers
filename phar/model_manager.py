import os
import psutil
import re
import requests
import socket
import subprocess
import time
from flask import Flask, request, jsonify
from typing import Tuple

app = Flask(__name__)

# Base command template for TensorFlow Serving
BASE_COMMAND = "tensorflow_model_server --rest_api_port={} --model_name={} --model_base_path={}"
DOCKER_IMAGE = "tensorflow/serving:latest-gpu"
PORT=5005

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a new model in TensorFlow Serving with specified GPU settings and method."""
    data = request.json
    model_name: str = data.get("model_name")
    model_path: str = data.get("model_path")
    port: int = data.get("port")
    gpu: str = data.get("gpu", "0")  # Default to GPU 0 if not specified
    method: str = data.get("method", "docker")  # Default method is Docker
    grow_gpu_mem: str = data.get("grow_gpu_mem",
                                 "true")  # Dynamically grow GPU memory

    # Validate input parameters
    if not model_name or not model_path or not port:
        return jsonify(
            {"error": "model_name, model_path, and port are required"}), 400

    # Terminate any existing process or Docker container using the specified port
    terminated, old_method = terminate_process_on_port(port)
    if terminated:
        print(
            f"Terminated existing process on port {port} and method {old_method}")

    try:
        # Choose the method to load the model (Docker or Conda environment)
        if method == "docker":
            load_model_docker(model_name, model_path, port, gpu, grow_gpu_mem)
        else:
            load_model_conda(model_name, model_path, port, gpu)

        # Check if the model is successfully loaded
        # if check_model_loaded(model_name, port):
        return jsonify({
            "message": f"Model {model_name} loaded from {model_path} on port {port} using GPU {gpu} via {method}"
        }), 200
        # else:
        #     return jsonify({
        #         "error": f"Failed to load model {model_name} on port {port}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def is_port_available(port: int) -> bool:
    """Check if the given port and the next consecutive port are available on the host."""

    def check_port(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost',
                                 p)) != 0  # Port is free if connect_ex returns non-zero

    return check_port(port) and check_port(port + 1)


@app.route('/check_ports', methods=['GET'])
def check_ports():
    """Endpoint to check if a specified port and the next consecutive port are available."""
    port = request.args.get("port", type=int)
    if port is None:
        return jsonify(
            {"error": "Port parameter is required and must be an integer"}), 400

    if is_port_available(port):
        return jsonify(
            {"message": f"Ports {port} and {port + 1} are available."}), 200
    else:
        return jsonify({
            "message": f"Ports {port} and {port + 1} are already in use."}), 409


@app.route('/gpu_memory', methods=['GET'])
def gpu_memory():
    """Endpoint to check available memory on each GPU using nvidia-smi."""
    try:
        gpu_memory_info = {}

        # Run nvidia-smi command to get memory usage details
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used',
             '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            check=True)

        # Parse the output and calculate free memory for each GPU
        for i, line in enumerate(result.stdout.strip().split('\n')):
            print(i, line)
            total_memory, used_memory = map(int, line.split(', '))
            free_memory = total_memory - used_memory
            gpu_memory_info[str(i)] = f"{free_memory} MiB"

        return jsonify(gpu_memory_info), 200

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Failed to execute nvidia-smi: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def load_model_conda(model_name: str, model_path: str, port: int, gpu: str):
    """Load model in TensorFlow Serving using tensorflow-serving-env conda environment."""
    env = os.environ.copy()
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    command = f"source ~/miniconda3/bin/activate tensorflow-serving-env && {BASE_COMMAND.format(port, model_name, model_path)}"
    print("load_model_conda:", " ".join(command))
    subprocess.Popen(command, shell=True, executable='/bin/bash', env=env)


def load_model_docker(model_name: str, model_path: str, port: int, gpu: str,
                      grow_gpu_memory: str):
    """Load model in TensorFlow Serving using Docker."""

    # Configure Docker GPU option
    gpu_option = f"device={gpu}" if gpu != "all" else "all"

    # Set Docker run command with specified resource constraints
    command = [
        "docker", "run", "-d", "--rm",
        "--name", f"{model_name}_{gpu}_{port}_tf_serving",
        "--gpus", gpu_option,  # Enable GPU usage in the container
        "-p", f"{port}:8501",
        "--mount", f"type=bind,source={model_path},target=/models/{model_name}",
        "-e", f"MODEL_NAME={model_name}",
        # Dynamically grow GPU memory
        "-e", f"TF_FORCE_GPU_ALLOW_GROWTH={grow_gpu_memory}",
        "-t", DOCKER_IMAGE
    ]
    print("load_model_docker:", " ".join(command))
    subprocess.run(command, check=True)


@app.route('/unload_model', methods=['POST'])
def unload_model():
    """Stop the TensorFlow Serving process or Docker container running the model."""
    data = request.json
    port: int = data.get("port")

    terminated, method = terminate_process_on_port(port)
    if terminated:
        return jsonify(
            {"message": f"Model on port {port} stopped via {method}"}), 200
    else:
        return jsonify({
            "error": f"No model found running on port {port} via {method}"}), 404


def terminate_process_on_port(port: int) -> Tuple[bool, str]:
    """Terminate any TensorFlow Serving process or Docker container using the specified port."""
    terminated = False
    method = "docker"

    # Find and terminate Docker container whose name ends with _{port}_tf_serving
    try:
        # Use docker ps to get list of containers
        docker_ps_cmd = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}} {{.Names}}"],
            capture_output=True, text=True, check=True
        )

        # Parse output and find matching container
        for line in docker_ps_cmd.stdout.strip().splitlines():
            container_id, container_name = line.split()
            if container_name.endswith(f"_{port}_tf_serving"):
                # Stop and remove the matching container
                docker_stop_cmd = subprocess.run(
                    ["docker", "stop", container_id], capture_output=True,
                    text=True)
                docker_rm_cmd = subprocess.run(["docker", "rm", container_id],
                                               capture_output=True, text=True)

                if docker_stop_cmd.returncode == 0:
                    print(f"Stopped Docker container {container_name}")
                    terminated = True
                if docker_rm_cmd.returncode == 0:
                    print(f"Removed Docker container {container_name}")
                    terminated = True
                break
    except subprocess.CalledProcessError as e:
        print(f"Error while listing or stopping Docker containers: {e}")

    # Terminate any TensorFlow Serving process if running in conda environment
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        # print("pid", proc.pid, ", name", proc.name, ", cmdline", proc.cmdline)
        try:
            # Get 'cmdline' and ensure it is not None by using .get() with a default value of an empty list
            cmdline = proc.info.get('cmdline')
            if isinstance(cmdline,
                          list) and f"--rest_api_port={port}" in cmdline and "tensorflow_model_server" in cmdline:
                proc.terminate()
                proc.wait()
                print(
                    f"Terminated TensorFlow Serving process with PID {proc.pid} on port {port}")
                method = "conda"
                terminated = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return terminated, method


def check_model_loaded(model_name: str, port: int) -> bool:
    """Check if the model is successfully loaded by sending a request to TensorFlow Serving."""
    url = f"http://localhost:{port}/v1/models/{model_name}"
    # Retry multiple times in case TensorFlow Serving takes time to load
    for i in range(5):
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                model_info = response.json()
                # Check if model status is AVAILABLE
                if model_info.get("model_version_status", [{}])[0].get(
                        "state") == "AVAILABLE":
                    return True
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(i)
    return False


@app.route('/check_model_loaded', methods=['GET'])
def check_model_loaded_endpoint():
    """Endpoint to check if a specified model is successfully loaded."""
    model_name = request.args.get("model_name")
    port = request.args.get("port", type=int)
    if not model_name or port is None:
        return jsonify(
            {"error": "model_name and port parameters are required"}), 400

    if check_model_loaded(model_name, port):
        return jsonify({
            "message": f"Model {model_name} is successfully loaded on port {port}"}), 200
    else:
        return jsonify({
            "message": f"Model {model_name} is not loaded on port {port}"}), 404


def get_serving_containers():
    """Fetch running TensorFlow Serving containers with details."""
    containers = []

    # Get Docker containers ending with "_tf_serving"
    docker_ps_output = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}} {{.ID}}"],
        capture_output=True, text=True
    ).stdout.splitlines()

    for line in docker_ps_output:
        name, container_id = line.split()
        if name.endswith("_tf_serving"):
            # Extract model_name, gpu, and port from the container name
            match = re.match(r"(.*)_(\S+)_(\d+)_tf_serving", name)
            if match:
                model_name = match.group(1)
                gpu = match.group(2)  # Can be "0", "1", "all", "0,1,3", etc.
                port = int(match.group(3))

                # Get PID of the container
                inspect_output = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Pid}}",
                     container_id],
                    capture_output=True, text=True
                ).stdout.strip()
                pid = int(inspect_output)

                containers.append({
                    "model_name": model_name,
                    "gpu": gpu,
                    "port": port,
                    "container_id": container_id,
                    "pid": pid,
                })

    return containers


@app.route('/active_serving_containers', methods=['GET'])
def active_serving_containers():
    """Endpoint to get details of active TensorFlow Serving containers."""
    containers = get_serving_containers()
    return jsonify(containers), 200


@app.route('/terminate_all_containers', methods=['POST'])
def terminate_all_containers():
    """Endpoint to forcefully kill all running TensorFlow Serving containers using kill -9."""

    # Extract optional model_name parameter from the request JSON body
    request_data = request.get_json()
    model_name_filter = request_data.get(
        "model_name") if request_data and "model_name" in request_data else None

    # Fetch running TensorFlow Serving containers
    containers = get_serving_containers()
    # Filter containers by model_name if the filter is provided
    if model_name_filter:
        containers = [container for container in containers if
                      container["model_name"] == model_name_filter]
        print(f"Terminating containers only with model {model_name_filter}")

    terminated_containers = []

    for container in containers:
        pid = container["pid"]
        container_id = container["container_id"]

        # Force kill the process using kill -9
        try:
            os.kill(pid, 9)  # Send SIGKILL to the process
            terminated_containers.append(container)
            print(f"Forcefully killed container with PID {pid}")
        except ProcessLookupError:
            print(f"Process with PID {pid} not found.")
        except Exception as e:
            print(f"Failed to kill process with PID {pid}: {e}")

        # Force kill the container using docker kill
        kill_cmd = subprocess.run(["docker", "kill", container_id],
                                  capture_output=True, text=True)
        if kill_cmd.returncode == 0:
            terminated_containers.append(container)
            print(f"Forcefully killed container {container_id}")
        else:
            print(f"Failed to kill container {container_id}: {kill_cmd.stderr}")

    terminated_containers = list(
        {container['container_id']: container for container in
         terminated_containers}.values())

    return jsonify({
        "terminated_containers": terminated_containers,
        "message": f"Forcefully terminated {len(terminated_containers)} containers"
    }), 200


@app.route('/terminate_container', methods=['POST'])
def terminate_container():
    """Endpoint to forcefully kill a single TensorFlow Serving container identified by port."""
    data = request.json
    port = data.get("port")

    if port is None:
        return jsonify({"error": "Port is required"}), 400

    containers = get_serving_containers()
    terminated_container = None

    for container in containers:
        if container["port"] == port:
            pid = container["pid"]
            container_id = container["container_id"]

            # Try to forcefully kill the process using kill -9
            try:
                os.kill(pid, 9)  # Send SIGKILL to the process
                terminated_container = container
                print(f"Forcefully killed container with PID {pid}")
            except ProcessLookupError:
                print(f"Process with PID {pid} not found.")
            except Exception as e:
                print(f"Failed to kill process with PID {pid}: {e}")

            # Force kill the container using docker kill as a fallback
            kill_cmd = subprocess.run(["docker", "kill", container_id],
                                      capture_output=True, text=True)
            if kill_cmd.returncode == 0:
                terminated_container = container
                print(f"Forcefully killed container {container_id}")
            else:
                print(
                    f"Failed to kill container {container_id}: {kill_cmd.stderr}")

            break

    if terminated_container:
        return jsonify({
            "terminated_container": terminated_container,
            "message": f"Forcefully terminated container on port {port}"
        }), 200
    else:
        return jsonify({"error": f"No container found on port {port}"}), 404


if __name__ == '__main__':
    print(f"App will run on {PORT}")
    app.run(host='0.0.0.0', port=PORT)
