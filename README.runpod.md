# LatentSync RunPod Serverless Deployment

This guide explains how to deploy the LatentSync inference script on RunPod serverless.

## Prerequisites

- Docker installed on your local machine
- A RunPod account
- Access to a MinIO server (in this case, https://minio.xenoumena.com)

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/LatentSync.git
   cd LatentSync
   ```

2. Update the `.env` file with your MinIO credentials:
   ```
   MINIO_ENDPOINT=https://minio.xenoumena.com
   MINIO_ACCESS_KEY=YOUR_ACCESS_KEY
   MINIO_SECRET_KEY=YOUR_SECRET_KEY
   MINIO_BUCKET=latentsync
   ```

3. Ensure you have all the necessary model files in the `checkpoints` directory:
   - `checkpoints/latentsync_unet.pt`
   - `checkpoints/whisper/tiny.pt` or `checkpoints/whisper/small.pt`
   - Any auxiliary models needed for face detection

## Building the Docker Image

1. Build the Docker image:
   ```bash
   docker build -t yourusername/latentsync-runpod:latest .
   ```

2. Push the Docker image to a registry (Docker Hub, GitHub Container Registry, etc.):
   ```bash
   docker push yourusername/latentsync-runpod:latest
   ```

## Deploying on RunPod

1. Log in to your RunPod account at https://runpod.io
2. Navigate to the Serverless section
3. Click on "Create Endpoint"
4. Fill in the following details:
   - Name: LatentSync
   - Docker Image: yourusername/latentsync-runpod:latest
   - Min Memory (GB): 16 (adjust based on your model's requirements)
   - GPU Type: NVIDIA RTX A5000 or better
   - Workers: 1 (adjust based on your needs)
   - Idle Timeout: 5 minutes (adjust based on your needs)
5. Click "Deploy"

## Using the Endpoint

Once deployed, you can use the RunPod API to submit jobs to your endpoint. Here's an example using Python:

```python
import requests
import json

# Replace with your RunPod API key and endpoint ID
API_KEY = "your_runpod_api_key"
ENDPOINT_ID = "your_endpoint_id"

# API URL
url = f"https://api.runpod.io/v2/{ENDPOINT_ID}/run"

# Request headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Request payload
payload = {
    "input": {
        "video_path": "input/video.mp4",
        "audio_path": "input/audio.wav",
        "output_path": "output/result.mp4",
        "guidance_scale": 1.5,
        "seed": 1247
    }
}

# Send request
response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
```

The response will include a `id` field that you can use to check the status of your job:

```python
job_id = response.json()["id"]
status_url = f"https://api.runpod.io/v2/{ENDPOINT_ID}/status/{job_id}"

# Check status
status_response = requests.get(status_url, headers=headers)
print(status_response.json())
```

When the job is complete, the output will include a `output_url` field with a presigned URL to download the generated video.

## Troubleshooting

- If you encounter issues with the MinIO connection, check your credentials in the `.env` file.
- If the model fails to load, ensure all required model files are present in the `checkpoints` directory.
- Check the RunPod logs for detailed error messages.

## Advanced Configuration

You can adjust the following parameters in the Dockerfile or handler.py:

- Change the base image to use a different CUDA version
- Modify the preloaded models
- Adjust the presigned URL expiration time (currently set to 7 days)
- Add additional error handling or logging