# Prediction interface for Cog ⚙️
# https://cog.run/python

# Set ONNXRuntime environment variables before any imports
import os
os.environ['ORT_DISABLE_THREAD_AFFINITY'] = '1'
os.environ['ORT_THREAD_POOL_ALLOW_SPINNING'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ORT_NUM_THREADS'] = '1'
os.environ['ONNXRUNTIME_DISABLE_TELEMETRY'] = '1'

# Initialize ONNXRuntime with our desired settings before any other imports
import onnxruntime as ort
print(f"Initializing ONNXRuntime {ort.__version__} with thread affinity disabled")
print(f"Available providers: {ort.get_available_providers()}")

# Create a session options object with our desired settings
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
print("Created ONNXRuntime SessionOptions with thread affinity disabled")

# Now import everything else
from cog import BasePredictor, Input, Path
import time
import subprocess

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/chunyu-li/LatentSync/model.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the model weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Soft links for the auxiliary models
        os.system("mkdir -p ~/.cache/torch/hub/checkpoints")
        os.system(
            "ln -s $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip"
        )
        os.system(
            "ln -s $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth"
        )
        os.system(
            "ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
        )

    def predict(
        self,
        video: Path = Input(description="Input video", default=None),
        audio: Path = Input(description="Input audio to ", default=None),
        guidance_scale: float = Input(description="Guidance scale", ge=1, le=2.5, default=1.5),
        inference_steps: int = Input(description="Number of inference steps", ge=1, le=50, default=20),
        seed: int = Input(description="Set to 0 for Random seed", default=0),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        video_path = str(video)
        audio_path = str(audio)
        config_path = "configs/unet/stage2.yaml"
        ckpt_path = "checkpoints/latentsync_unet.pt"
        output_path = "/tmp/video_out.mp4"
        env = os.environ.copy()
        
        # Environment variables are already set at the module level
        # Just pass them through to the subprocess
        
        # Log the environment variables
        print(f"Using environment variables set at the beginning of predict.py")
        # --- End added lines ---

        # Command as a list for subprocess.run
        command = [
            "python",
            "-m",
            "scripts.inference",
            "--unet_config_path", config_path,
            "--inference_ckpt_path", ckpt_path,
            "--inference_steps", str(inference_steps),
            "--guidance_scale", str(guidance_scale),
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--video_out_path", output_path,
            "--seed", str(seed)
        ]

        # Run the command using subprocess.run
        # Pass the modified environment explicitly
        subprocess.run(command, check=True, env=env)

        return Path(output_path)