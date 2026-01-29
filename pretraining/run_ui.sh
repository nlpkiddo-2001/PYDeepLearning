#!/bin/bash

# Install dependencies if not present
echo "Installing dependencies..."
pip install fastapi uvicorn sse-starlette jinja2

# Run the server
echo "Starting Chat UI Server..."
# Use python3 or python depending on the environment
python3 server.py
