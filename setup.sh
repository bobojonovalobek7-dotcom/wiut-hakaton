#!/bin/bash
set -e

echo "Setting up 4GB Swap Space..."
sudo fallocate -l 4G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

echo "Installing Docker, Nginx, and Certbot..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose git nginx python3-certbot-nginx

echo "Cloning Repository..."
cd /home/ubuntu
if [ ! -d "wiut-hakaton" ]; then
    git clone https://github.com/bobojonovalobek7-dotcom/wiut-hakaton.git
fi
cd wiut-hakaton

echo "Starting Application in Docker..."
sudo docker-compose up -d --build

echo "Configuring Nginx Reverse Proxy for wiut.orifdev.uz..."
sudo bash -c 'cat > /etc/nginx/sites-available/wiut <<EOF
server {
    listen 80;
    server_name wiut.orifdev.uz;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF'

sudo ln -sf /etc/nginx/sites-available/wiut /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl restart nginx

echo "Setup completed successfully!"
