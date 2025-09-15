#!/usr/bin/env python3
import os
import subprocess
import sys
import time

# Get PORT from environment or use default
PORT = os.environ.get('PORT', '8000')
print(f"Starting services on port {PORT}...")

# Start new.mjs in background
print("Starting Pyth Lazer price fetcher...")
node_process = subprocess.Popen(['node', 'new.mjs'])

# Give it time to connect
time.sleep(2)

# Start the WebSocket server
print(f"Starting WebSocket server on port {PORT}")
try:
    subprocess.run([
        'uvicorn', 
        'socket_server:app', 
        '--host', '0.0.0.0', 
        '--port', PORT
    ])
except KeyboardInterrupt:
    print("\nShutting down...")
    node_process.terminate()
    sys.exit(0)