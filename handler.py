"""LatentSync inference handler for RunPod."""

import os
import runpod
import subprocess
import tempfile
import shutil
import time
import json
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
import urllib.parse
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper
from accelerate.utils import set_seed

# Load environment variables from .env file
load_dotenv()

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT.replace("https://", "").replace("http://", ""),
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
)

# Ensure bucket exists
try:
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
        # Set bucket policy to allow public read access
except Exception as e:
    print(f"Warning: Could not ensure bucket exists or set policy: {str(e)}")

# Global variables for preloaded models
global_config = None
global_scheduler = None
global_vae = None
global_audio_encoder = None
inference_ckpt_path = "checkpoints/latentsync_unet.pt"
# Load models into memory
def load_models():
    """Preload models into memory to speed up inference."""
    global global_config, global_scheduler, global_vae, global_audio_encoder
    
    print("Loading models into memory...")
    
    # Load config
    config_path = "configs/unet/stage2_512.yaml"
    global_config = OmegaConf.load(config_path)
    
    # Check if GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32
    
    # Load scheduler
    global_scheduler = DDIMScheduler.from_pretrained("configs")
    
    # Load VAE
    global_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    global_vae.config.scaling_factor = 0.18215
    global_vae.config.shift_factor = 0
    
    # Determine which whisper model to use based on config
    if global_config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif global_config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")
    
    # Load audio encoder
    global_audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=global_config.data.num_frames,
        audio_feat_length=global_config.data.audio_feat_length,
    )
    
    # Load denoising UNet
    
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(global_config.model),
        inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)
    
    print("Models loaded successfully")

# Initialize models
try:
    load_models()
except Exception as e:
    print(f"Warning: Failed to preload models: {str(e)}")
    print("Models will be loaded during inference")

def download_from_minio(minio_path, local_path):
    """Download a file from MinIO to a local path."""
    try:
        # Parse MinIO path
        if minio_path.startswith('minio://'):
            minio_path = minio_path[8:]
        
        # Extract bucket and object name
        bucket_name = MINIO_BUCKET
        object_name = minio_path
        
        # Download file
        minio_client.fget_object(bucket_name, object_name, local_path)
        return True
    except S3Error as e:
        print(f"Error downloading from MinIO: {str(e)}")
        return False

def upload_to_minio(local_path, minio_path):
    """Upload a file from a local path to MinIO and return a public URL."""
    try:
        # Parse MinIO path
        if minio_path.startswith('minio://'):
            minio_path = minio_path[8:]
        
        # Extract bucket and object name
        bucket_name = MINIO_BUCKET
        object_name = minio_path
        
        # Upload file
        minio_client.fput_object(bucket_name, object_name, local_path)
        
        # Generate a presigned URL that expires in 7 days (604800 seconds)
        url = minio_client.presigned_get_object(
            bucket_name, 
            object_name,
        )
        
        return url
    except S3Error as e:
        print(f"Error uploading to MinIO: {str(e)}")
        return None

def run_inference(video_path, audio_path, output_path, guidance_scale=1.0, seed=1247, inference_steps=20):
    """Run inference using the preloaded models."""
    global global_config, global_scheduler, global_vae, global_audio_encoder
    
    # If models are not preloaded, load them now
    if global_config is None:
        try:
            load_models()
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {video_path}")
    print(f"Input audio path: {audio_path}")
    print(f"Loaded checkpoint path: {inference_ckpt_path}")

    scheduler = DDIMScheduler.from_pretrained("configs")

    if global_config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif global_config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=global_config.data.num_frames,
        audio_feat_length=global_config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(global_config.model),
        inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    # use DeepCache

    helper = DeepCacheSDHelper(pipe=pipeline)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()

    if seed != -1:
        set_seed(seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    pipeline(
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=output_path,
        num_frames=global_config.data.num_frames,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        weight_dtype=dtype,
        width=global_config.data.resolution,
        height=global_config.data.resolution,
        mask_image_path=global_config.data.mask_image_path,
    )

def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]
    
    # Create a temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    try:
        # Get input parameters
        video_path = job_input.get("video_path")
        audio_path = job_input.get("audio_path")
        output_path = job_input.get("output_path")
        guidance_scale = float(job_input.get("guidance_scale", 1.0))
        seed = int(job_input.get("seed", 1247))
        
        if not video_path or not audio_path or not output_path:
            return {"error": "Missing required parameters: video_path, audio_path, or output_path"}
        
        # Download input files
        local_video_path = os.path.join(temp_dir, "input_video.mp4")
        local_audio_path = os.path.join(temp_dir, "input_audio.wav")
        local_output_path = os.path.join(temp_dir, "output_video.mp4")
        
        print(f"Downloading video from {video_path}")
        if not download_from_minio(video_path, local_video_path):
            return {"error": f"Failed to download video from {video_path}"}
        
        print(f"Downloading audio from {audio_path}")
        if not download_from_minio(audio_path, local_audio_path):
            return {"error": f"Failed to download audio from {audio_path}"}
        
        # Run inference
        print("Running inference")
        start_time = time.time()
        
        success = run_inference(
            video_path=local_video_path,
            audio_path=local_audio_path,
            output_path=local_output_path,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        if not success:
            return {"error": "Inference failed"}
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")
        
        # Upload output file
        print(f"Uploading output to {output_path}")
        output_url = upload_to_minio(local_output_path, output_path)
        if not output_url:
            return {"error": f"Failed to upload output to {output_path}"}
        
        # Return result
        return {
            "output_url": output_url,
            "inference_time": inference_time
        }
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

# Start the serverless function
runpod.serverless.start({"handler": handler})