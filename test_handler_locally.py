#!/usr/bin/env python3
"""
Test script for the RunPod handler locally.
This script simulates a RunPod job request to test the handler function.
"""

import json
import os
from handler import handler

# Create a test job
test_job = {
    "id": "local_test",
    "input": {
        "video_path": "test_input/video.mp4",  # Replace with actual path
        "audio_path": "test_input/audio.wav",  # Replace with actual path
        "output_path": "test_output/result.mp4",
        "guidance_scale": 1.5,
        "seed": 1247
    }
}

# Create test directories if they don't exist
os.makedirs("test_input", exist_ok=True)
os.makedirs("test_output", exist_ok=True)

print("Make sure you have test video and audio files in the test_input directory:")
print(f"- {test_job['input']['video_path']}")
print(f"- {test_job['input']['audio_path']}")
print("\nRunning handler with test job...")

# Run the handler
result = handler(test_job)

# Print the result
print("\nResult:")
print(json.dumps(result, indent=2))

if "error" in result:
    print(f"\n❌ Test failed: {result['error']}")
else:
    print(f"\n✅ Test succeeded!")
    print(f"Output URL: {result['output_url']}")
    print(f"Inference time: {result['inference_time']:.2f} seconds")