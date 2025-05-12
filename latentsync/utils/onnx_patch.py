"""
Patch for ONNXRuntime to disable thread affinity.
This file should be imported before any other imports that might use ONNXRuntime.
"""

import os

# Set environment variables
os.environ['ORT_DISABLE_THREAD_AFFINITY'] = '1'
os.environ['ORT_THREAD_POOL_ALLOW_SPINNING'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ORT_NUM_THREADS'] = '1'
os.environ['ONNXRUNTIME_DISABLE_TELEMETRY'] = '1'

print("onnx_patch.py: Setting ONNXRuntime environment variables")
print(f"ORT_DISABLE_THREAD_AFFINITY={os.environ.get('ORT_DISABLE_THREAD_AFFINITY')}")
print(f"ORT_THREAD_POOL_ALLOW_SPINNING={os.environ.get('ORT_THREAD_POOL_ALLOW_SPINNING')}")
print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
print(f"ORT_NUM_THREADS={os.environ.get('ORT_NUM_THREADS')}")

# We're not directly patching ONNXRuntime anymore as it was causing compatibility issues
# Instead, we're relying on the environment variables to control thread affinity