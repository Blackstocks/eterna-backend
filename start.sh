#!/bin/bash

# Railway provides PORT env variable - use default if not set
PORT="${PORT:-8000}"

echo "Starting services on port $PORT..."

# Start new.mjs in background to populate Redis with Pyth prices
echo "Starting Pyth Lazer price fetcher..."
node new.mjs &

# Give it a moment to connect
sleep 2

# Start the WebSocket server with explicit port number
echo "Starting WebSocket server on port $PORT"
exec uvicorn socket_server:app --host 0.0.0.0 --port "$PORT"