#\!/bin/bash

# Railway provides PORT env variable
PORT=${PORT:-8000}

echo "Starting server on port $PORT"
uvicorn socket_server:app --host 0.0.0.0 --port $PORT
EOF < /dev/null