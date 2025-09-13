#!/bin/bash

# Update system
apt-get update
apt-get upgrade -y

# Install Python and required packages
apt-get install -y python3.11 python3.11-venv python3-pip git redis-server nginx

# Install Docker (optional, for containerized deployment)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Clone repository (you'll need to update this with your repo URL)
# git clone https://github.com/yourusername/euphoria-trading-backend.git /app

# For now, create app directory
mkdir -p /app
cd /app

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Copy application files (this would be from git in production)
# For manual deployment, you'll SCP these files

# Install Python dependencies
# pip install -r requirements.txt

# Configure Redis
sed -i 's/bind 127.0.0.1/bind 0.0.0.0/g' /etc/redis/redis.conf
systemctl restart redis-server

# Create systemd service for the WebSocket server
cat > /etc/systemd/system/euphoria-ws.service << EOF
[Unit]
Description=Euphoria Trading WebSocket Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/app
Environment="PATH=/app/venv/bin"
ExecStart=/app/venv/bin/uvicorn socket_server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Configure Nginx as reverse proxy (optional, for SSL)
cat > /etc/nginx/sites-available/euphoria-ws << EOF
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

ln -s /etc/nginx/sites-available/euphoria-ws /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
systemctl restart nginx

# Enable and start the service
# systemctl enable euphoria-ws
# systemctl start euphoria-ws

echo "Setup complete! WebSocket server will be available on port 8000"