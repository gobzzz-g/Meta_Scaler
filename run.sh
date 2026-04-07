#!/bin/bash
# Start the FastAPI server in the background
python server.py &
# Wait for the server to initialize
sleep 3
# Run the inference tasks
python inference.py
# Keep the container running by waiting for background processes
wait