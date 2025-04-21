from insightface.app import FaceAnalysis
import numpy as np
import torch
import os

INSIGHTFACE_DETECT_SIZE = 640

# Set ONNXRuntime environment variables at module level
# This ensures they're set before any ONNX models are loaded
os.environ['ORT_DISABLE_THREAD_AFFINITY'] = '1'
os.environ['ORT_THREAD_POOL_ALLOW_SPINNING'] = '0'

# Print debug information
print("face_detector.py: Setting ONNXRuntime environment variables")
print(f"ORT_DISABLE_THREAD_AFFINITY={os.environ.get('ORT_DISABLE_THREAD_AFFINITY')}")
print(f"ORT_THREAD_POOL_ALLOW_SPINNING={os.environ.get('ORT_THREAD_POOL_ALLOW_SPINNING')}")
print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
print(f"ORT_NUM_THREADS={os.environ.get('ORT_NUM_THREADS')}")

# Try to import onnxruntime directly to check if it's available
try:
    import importlib
    ort_spec = importlib.util.find_spec("onnxruntime")
    print(f"ONNXRuntime found: {ort_spec is not None}")
    if ort_spec is not None:
        import onnxruntime as ort
        print(f"ONNXRuntime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
        # Try to create a session options object to see if it works
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 1
            print("Successfully created ONNXRuntime SessionOptions")
        except Exception as e:
            print(f"Error creating ONNXRuntime SessionOptions: {e}")
except ImportError:
    print("ONNXRuntime not directly importable")
except Exception as e:
    print(f"Error checking ONNXRuntime: {e}")


class FaceDetector:
    def __init__(self, device="cuda"):
        # Configure provider options to prevent thread affinity issues
        device_id = cuda_to_int(device)
        provider_options = [
            {
                'device_id': device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
            },
            {
                'arena_extend_strategy': 'kNextPowerOfTwo',
            }
        ]
        
        # Use both CUDA and CPU providers with explicit options
        self.app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            root="checkpoints/auxiliary",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=provider_options,
        )
        self.app.prepare(ctx_id=device_id, det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))

    def __call__(self, frame, threshold=0.5):
        f_h, f_w, _ = frame.shape

        faces = self.app.get(frame)

        get_face_store = None
        max_size = 0

        if len(faces) == 0:
            return None, None
        else:
            for face in faces:
                bbox = face.bbox.astype(np.int_).tolist()
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < 50 or h < 80:
                    continue
                if w / h > 1.5 or w / h < 0.2:
                    continue
                if face.det_score < threshold:
                    continue
                size_now = w * h

                if size_now > max_size:
                    max_size = size_now
                    get_face_store = face

        if get_face_store is None:
            return None, None
        else:
            face = get_face_store
            lmk = np.round(face.landmark_2d_106).astype(np.int_)

            halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)  # lmk[73]

            sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
            halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
            upper_bond = halk_face_coord[1] - halk_face_dist  # *0.94

            x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond), np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()

            y2 += int((x2 - x1) * 0.1)
            x1 -= int((x2 - x1) * 0.05)
            x2 += int((x2 - x1) * 0.05)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(f_w, x2)
            y2 = min(f_h, y2)

            return (x1, y1, x2, y2), lmk


def cuda_to_int(cuda_str: str) -> int:
    """
    Convert the string with format "cuda:X" to integer X.
    """
    if cuda_str == "cuda":
        return 0
    device = torch.device(cuda_str)
    if device.type != "cuda":
        raise ValueError(f"Device type must be 'cuda', got: {device.type}")
    return device.index


LMK_ADAPT_ORIGIN_ORDER = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    73,
    74,
    86,
]
