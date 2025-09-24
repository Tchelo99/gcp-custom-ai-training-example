# Use a CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Avoid re-installing dependencies when code changes
WORKDIR /app

# Only copy files needed to install dependencies first
COPY requirements.txt .

# Avoid cache busting for dependencies (this line will cache)
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app (this is what changes often)
COPY . .

# Prevent Python from buffering outputs
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "train.py"]