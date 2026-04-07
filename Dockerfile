# Use stable Python image (avoid slim issues)
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (for safety)
EXPOSE 7860

# Run inference
CMD ["python", "server.py"]