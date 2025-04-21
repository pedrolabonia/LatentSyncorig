"""
Patch for ONNXRuntime to disable thread affinity.
This file should be imported before any other imports that might use ONNXRuntime.
"""

import os
import sys
import importlib
from functools import wraps

# Set environment variables
os.environ['ORT_DISABLE_THREAD_AFFINITY'] = '1'
os.environ['ORT_THREAD_POOL_ALLOW_SPINNING'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ORT_NUM_THREADS'] = '1'

print("onnx_patch.py: Setting ONNXRuntime environment variables")
print(f"ORT_DISABLE_THREAD_AFFINITY={os.environ.get('ORT_DISABLE_THREAD_AFFINITY')}")
print(f"ORT_THREAD_POOL_ALLOW_SPINNING={os.environ.get('ORT_THREAD_POOL_ALLOW_SPINNING')}")
print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
print(f"ORT_NUM_THREADS={os.environ.get('ORT_NUM_THREADS')}")

# Try to patch ONNXRuntime directly
try:
    # Check if ONNXRuntime is already imported
    if 'onnxruntime' in sys.modules:
        print("ONNXRuntime already imported, patching existing module")
        ort = sys.modules['onnxruntime']
    else:
        print("Importing ONNXRuntime for patching")
        import onnxruntime as ort
    
    print(f"ONNXRuntime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    
    # Patch SessionOptions to disable thread affinity
    original_session_options = ort.SessionOptions
    
    @wraps(original_session_options)
    def patched_session_options(*args, **kwargs):
        session_options = original_session_options(*args, **kwargs)
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Try to access and modify thread affinity setting if available
        try:
            # This is a bit of a hack, but we're trying to access internal attributes
            if hasattr(session_options, '_enable_cpu_mem_arena'):
                session_options._enable_cpu_mem_arena = False
            
            # Try to set thread affinity directly if the attribute exists
            if hasattr(session_options, '_disable_thread_affinity'):
                session_options._disable_thread_affinity = True
        except Exception as e:
            print(f"Error modifying SessionOptions attributes: {e}")
        
        print("Created patched SessionOptions with thread affinity disabled")
        return session_options
    
    # Replace the original SessionOptions with our patched version
    ort.SessionOptions = patched_session_options
    print("Successfully patched ONNXRuntime SessionOptions")
    
    # Patch InferenceSession to use our patched SessionOptions
    original_inference_session = ort.InferenceSession
    
    @wraps(original_inference_session)
    def patched_inference_session(path, sess_options=None, providers=None, provider_options=None, **kwargs):
        if sess_options is None:
            print("Creating new SessionOptions in patched InferenceSession")
            sess_options = patched_session_options()
        
        if providers is None:
            # Still use GPU but with CPU as fallback
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"Using default providers in patched InferenceSession: {providers}")
        
        if provider_options is None:
            # Configure provider options to avoid thread affinity issues
            provider_options = [
                {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'DEFAULT',
                    'do_copy_in_default_stream': True,
                },
                {
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                }
            ]
            print(f"Using custom provider_options in patched InferenceSession")
        
        print(f"Creating InferenceSession with providers: {providers}")
        return original_inference_session(path, sess_options, providers, provider_options, **kwargs)
    
    # Replace the original InferenceSession with our patched version
    ort.InferenceSession = patched_inference_session
    print("Successfully patched ONNXRuntime InferenceSession")
    
except ImportError:
    print("ONNXRuntime not available for patching")
except Exception as e:
    print(f"Error patching ONNXRuntime: {e}")