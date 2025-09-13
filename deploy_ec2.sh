#!/bin/bash

# AWS EC2 Deployment Script for WebSocket Server
# This script deploys the WebSocket server to an EC2 instance

# Configuration
INSTANCE_TYPE="t3.micro"
KEY_NAME="euphoria-trading-key"
SECURITY_GROUP="euphoria-ws-sg"
REGION="us-east-1"

echo "Setting up AWS EC2 deployment..."

# Create security group for WebSocket
aws ec2 create-security-group \
    --group-name $SECURITY_GROUP \
    --description "Security group for Euphoria Trading WebSocket server" \
    --region $REGION

# Allow WebSocket traffic on port 8000
aws ec2 authorize-security-group-ingress \
    --group-name $SECURITY_GROUP \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Allow SSH access
aws ec2 authorize-security-group-ingress \
    --group-name $SECURITY_GROUP \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Create key pair
aws ec2 create-key-pair \
    --key-name $KEY_NAME \
    --query 'KeyMaterial' \
    --output text > $KEY_NAME.pem

chmod 400 $KEY_NAME.pem

# Launch EC2 instance with Ubuntu 22.04
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --region $REGION \
    --user-data file://user_data.sh \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched with ID: $INSTANCE_ID"

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance is running at: $PUBLIC_IP"
echo "WebSocket endpoint will be available at: ws://$PUBLIC_IP:8000"
echo ""
echo "To connect via SSH: ssh -i $KEY_NAME.pem ubuntu@$PUBLIC_IP"
echo ""
echo "Update your Flutter app with the new endpoint: ws://$PUBLIC_IP:8000"