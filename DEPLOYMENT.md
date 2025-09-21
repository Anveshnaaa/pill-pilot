# 🚀 PillPilot Deployment Guide

This guide covers different deployment options for PillPilot, from local development to production environments.

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- Web server (for production)

## 🏠 Local Development

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/pillpilot.git
cd pillpilot

# Run setup script
python setup.py

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Start the application
python app.py
```

### Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env  # Edit as needed

# Run the application
python app.py
```

## 🌐 Production Deployment

### Option 1: Traditional VPS/Server

#### 1. Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv nginx -y

# Install Node.js (for frontend builds if needed)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs -y
```

#### 2. Application Setup
```bash
# Create application directory
sudo mkdir -p /var/www/pillpilot
sudo chown $USER:$USER /var/www/pillpilot

# Clone repository
cd /var/www/pillpilot
git clone https://github.com/yourusername/pillpilot.git .

# Setup application
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. Configure Nginx
```nginx
# /etc/nginx/sites-available/pillpilot
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /var/www/pillpilot/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

#### 4. Configure Systemd Service
```ini
# /etc/systemd/system/pillpilot.service
[Unit]
Description=PillPilot Medicine Inventory Management
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/pillpilot
Environment=PATH=/var/www/pillpilot/venv/bin
ExecStart=/var/www/pillpilot/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

#### 5. Start Services
```bash
# Enable and start services
sudo systemctl enable pillpilot
sudo systemctl start pillpilot
sudo systemctl enable nginx
sudo systemctl restart nginx

# Check status
sudo systemctl status pillpilot
```

### Option 2: Docker Deployment

#### 1. Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 pillpilot && chown -R pillpilot:pillpilot /app
USER pillpilot

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run application
CMD ["python", "app.py"]
```

#### 2. Create docker-compose.yml
```yaml
version: '3.8'

services:
  pillpilot:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - FLASK_DEBUG=False
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./static:/var/www/static
    depends_on:
      - pillpilot
    restart: unless-stopped
```

#### 3. Deploy with Docker
```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 3: Cloud Platform Deployment

#### Heroku
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create app
heroku create your-pillpilot-app

# Set environment variables
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=your-secret-key

# Deploy
git push heroku main
```

#### Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
railway up
```

#### DigitalOcean App Platform
1. Connect your GitHub repository
2. Configure build settings:
   - Build command: `pip install -r requirements.txt`
   - Run command: `python app.py`
3. Set environment variables
4. Deploy

## 🔧 Environment Configuration

### Production Environment Variables
```env
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-super-secret-key-here

# Database (if using persistent storage)
DATABASE_URL=postgresql://user:password@localhost/pillpilot

# ML Configuration
ML_ENABLED=True
MODEL_UPDATE_INTERVAL=3600

# Security
SESSION_COOKIE_SECURE=True
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/pillpilot/app.log
```

## 📊 Performance Optimization

### 1. Database Optimization
```python
# For production, consider using PostgreSQL
DATABASE_URL = "postgresql://user:password@localhost/pillpilot"

# Add connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

### 2. Caching
```python
# Add Redis for caching
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0'
})
```

### 3. Static File Serving
```nginx
# Serve static files directly with Nginx
location /static {
    alias /var/www/pillpilot/static;
    expires 1y;
    add_header Cache-Control "public, immutable";
    gzip_static on;
}
```

## 🔒 Security Considerations

### 1. Environment Security
- Use strong, unique secret keys
- Never commit sensitive data to version control
- Use environment variables for configuration
- Enable HTTPS in production

### 2. Application Security
```python
# Add security headers
from flask_talisman import Talisman

Talisman(app, force_https=True)

# Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

### 3. Data Security
- Encrypt sensitive data at rest
- Use secure file upload validation
- Implement proper access controls
- Regular security updates

## 📈 Monitoring and Logging

### 1. Application Monitoring
```python
# Add monitoring
from flask_monitoringdashboard import dashboard

dashboard.config.init_from(file='config.cfg')
dashboard.bind(app)
```

### 2. Logging Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/pillpilot.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
```

### 3. Health Checks
```python
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }
```

## 🔄 Backup and Recovery

### 1. Database Backups
```bash
# PostgreSQL backup
pg_dump -h localhost -U username pillpilot > backup_$(date +%Y%m%d).sql

# Restore
psql -h localhost -U username pillpilot < backup_20240101.sql
```

### 2. File Backups
```bash
# Backup application files
tar -czf pillpilot_backup_$(date +%Y%m%d).tar.gz /var/www/pillpilot

# Backup with rsync
rsync -av /var/www/pillpilot/ backup-server:/backups/pillpilot/
```

## 🚨 Troubleshooting

### Common Issues

**1. Application won't start**
```bash
# Check logs
sudo journalctl -u pillpilot -f

# Check port availability
sudo netstat -tlnp | grep :5000
```

**2. Database connection issues**
```bash
# Test database connection
python -c "from app import db; print(db.engine.execute('SELECT 1').scalar())"
```

**3. Static files not loading**
```bash
# Check Nginx configuration
sudo nginx -t

# Check file permissions
ls -la /var/www/pillpilot/static/
```

### Performance Issues

**1. Slow response times**
- Check database query performance
- Enable query logging
- Consider adding indexes
- Implement caching

**2. High memory usage**
- Monitor with `htop` or `top`
- Check for memory leaks
- Optimize data processing
- Consider horizontal scaling

## 📞 Support

For deployment issues:
- Check the logs first
- Review this documentation
- Create an issue on GitHub
- Include system information and error logs

---

**Happy Deploying! 🚀**

