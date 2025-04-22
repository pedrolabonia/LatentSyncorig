FROM runpod/base:0.6.2-cuda12.1.0

# Set working directory
WORKDIR /app

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Install system dependencies (removed curl, tar, gzip, which as not needed for pip)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# UV-specific environment variables are not needed for pip system installs
# ENV UV_PYTHON_PREFERENCE="only-system"
# ENV UV_PROJECT_ENVIRONMENT="/usr/local/" # Not needed

# Copy pyproject.toml into the working directory where pip install . will find it
COPY pyproject.toml /app/pyproject.toml

# Install dependencies from pyproject.toml using pip
# pip install . looks for pyproject.toml in the current directory (/app)
# and installs the 'project.dependencies' into the active python environment (system)
# --no-cache-dir is used to reduce the size of the Docker image layer.
RUN pip install . --no-cache-dir

# Copy necessary code and configs AFTER dependencies are installed
COPY scripts/ /app/scripts/
COPY latentsync/ /app/latentsync/
COPY configs/ /app/configs/
COPY checkpoints/ /app/checkpoints/
COPY .env /app/.env
COPY handler.py /app/

# Set environment variables
ENV PYTHONPATH=/app

# Run the handler
CMD ["python", "-u", "/app/handler.py"]
