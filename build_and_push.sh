#!/bin/bash
# Script to build and push the Docker image for RunPod deployment

# Configuration
IMAGE_NAME="latentsync-runpod"
DOCKER_USERNAME="$1"  # Pass as first argument
TAG="latest"

# Check if Docker username is provided
if [ -z "$DOCKER_USERNAME" ]; then
    echo "Error: Docker username not provided"
    echo "Usage: $0 <docker_username>"
    exit 1
fi

# Full image name
FULL_IMAGE_NAME="$DOCKER_USERNAME/$IMAGE_NAME:$TAG"

echo "Building Docker image: $FULL_IMAGE_NAME"
docker build -t "$FULL_IMAGE_NAME" .

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

echo "Docker image built successfully"

# Ask for confirmation before pushing
read -p "Do you want to push the image to Docker Hub? (y/n): " PUSH_CONFIRM

if [ "$PUSH_CONFIRM" = "y" ] || [ "$PUSH_CONFIRM" = "Y" ]; then
    echo "Logging in to Docker Hub..."
    docker login
    
    echo "Pushing image to Docker Hub..."
    docker push "$FULL_IMAGE_NAME"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to push Docker image"
        exit 1
    fi
    
    echo "Image pushed successfully: $FULL_IMAGE_NAME"
    echo "You can now deploy this image on RunPod"
else
    echo "Skipping push to Docker Hub"
    echo "You can push the image later with: docker push $FULL_IMAGE_NAME"
fi