# PowerShell script to build and push the Docker image for RunPod deployment

# Configuration
$ImageName = "latentsync-runpod"
$Tag = "latest"

# Get Docker username from command line argument
$DockerUsername = $args[0]

# Check if Docker username is provided
if (-not $DockerUsername) {
    Write-Error "Docker username not provided"
    Write-Host "Usage: .\build_and_push.ps1 <docker_username>"
    exit 1
}

# Full image name
$FullImageName = "$DockerUsername/$ImageName`:$Tag"

Write-Host "Building Docker image: $FullImageName"
docker build --platform linux/amd64 -t $FullImageName .

# Check if build was successful
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed"
    exit 1
}

Write-Host "Docker image built successfully"

# Ask for confirmation before pushing
$PushConfirm = Read-Host "Do you want to push the image to Docker Hub? (y/n)"

if ($PushConfirm -eq "y" -or $PushConfirm -eq "Y") {
    Write-Host "Logging in to Docker Hub..."
    docker login
    
    Write-Host "Pushing image to Docker Hub..."
    docker push $FullImageName
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to push Docker image"
        exit 1
    }
    
    Write-Host "Image pushed successfully: $FullImageName"
    Write-Host "You can now deploy this image on RunPod"
} else {
    Write-Host "Skipping push to Docker Hub"
    Write-Host "You can push the image later with: docker push $FullImageName"
}