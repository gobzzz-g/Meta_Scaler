# Stable base image (avoids Docker Hub auth issues)
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run inference
CMD ["python", "inference.py"]