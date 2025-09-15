FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package.json for Node dependencies
COPY package*.json ./
RUN npm install

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x start_server.py

# Expose the port
EXPOSE 8000

# Run using Python script to handle PORT properly
CMD ["python", "start_server.py"]