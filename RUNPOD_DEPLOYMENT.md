# RunPod Deployment Guide (CPU)

Simple deployment guide for running the Clinical Trial Assistant on RunPod CPU instances.

## 🚀 Quick Start

### Prerequisites
- RunPod account (https://runpod.io)
- Your OpenAI API key

### Estimated Cost
- **CPU Pod (16 vCPU, 32GB RAM):** ~$0.20-0.40/hour (~$150-300/month if running 24/7)
- **Storage:** ~$0.10/GB/month (need ~50GB for data)

---

## Step-by-Step Deployment

### 1. Create RunPod Instance

1. Go to https://runpod.io
2. Click **"Deploy"** → **"CPU"**
3. Select a template:
   - **Recommended:** Ubuntu 22.04 with Docker pre-installed
   - Minimum specs: **8 vCPU, 16GB RAM**
   - Recommended: **16 vCPU, 32GB RAM** (for Elasticsearch)
4. Set storage: **50GB** (for database + Elasticsearch data)
5. Enable **SSH** and **HTTP Ports**
6. **Expose ports:**
   - `8000` (API)
   - `5432` (PostgreSQL - optional, for debugging)
   - `9200` (Elasticsearch - optional, for debugging)
7. Click **Deploy**

### 2. Connect to Your Instance

```bash
# Get your pod's SSH details from RunPod dashboard
ssh root@YOUR_POD_IP -p YOUR_SSH_PORT -i ~/.ssh/id_ed25519
```

### 3. Install Dependencies (if not pre-installed)

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt-get install -y docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

### 4. Upload Your Application

**Option A: Git Clone (Recommended)**
```bash
# Install git if needed
apt-get install -y git

# Clone your repository
git clone https://github.com/YOUR_USERNAME/clinical-trial-assistant.git
cd clinical-trial-assistant
```

**Option B: Upload via SCP**
```bash
# From your local machine
scp -P YOUR_SSH_PORT -r /path/to/trial-assitant root@YOUR_POD_IP:/root/
```

### 5. Configure Environment

```bash
# Copy production environment template
cp .env.production .env.prod

# Edit with your credentials
nano .env.prod
```

**Required changes in `.env.prod`:**
```bash
# Set your OpenAI API key
OPENAI_API_KEY=sk-your-actual-key-here

# Strong password for PostgreSQL
POSTGRES_PASSWORD=your_strong_password_here_change_this

# Optional: LangSmith for monitoring
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
```

### 6. Start the Services

```bash
# Start PostgreSQL and Elasticsearch first
docker compose -f docker-compose.prod.yml up -d postgres elasticsearch

# Wait for databases to be healthy (about 30-60 seconds)
docker compose -f docker-compose.prod.yml ps

# Check logs if needed
docker compose -f docker-compose.prod.yml logs postgres
docker compose -f docker-compose.prod.yml logs elasticsearch
```

### 7. Run Data Pipeline (One-Time Setup)

This downloads AACT data, imports to PostgreSQL, and indexes to Elasticsearch.

**⚠️ WARNING: This will take 30-60 minutes and download ~3GB of data!**

```bash
# Create a screen session so it doesn't stop if SSH disconnects
screen -S data-pipeline

# Run the full pipeline
docker compose -f docker-compose.prod.yml run --rm \
  -e POSTGRES_HOST=postgres \
  -e POSTGRES_PORT=5432 \
  -e ELASTICSEARCH_URL=http://elasticsearch:9200 \
  api sh -c "
    echo '📥 Step 1: Downloading AACT data...'
    mkdir -p /app/data/downloads
    uv run python data_pipeline/download_aact.py --output-dir /app/data/downloads
    
    echo '📦 Step 2: Importing to PostgreSQL...'
    LATEST_DMP=\$(ls -t /app/data/downloads/*.dmp | head -1)
    bash data_pipeline/import_aact_data_docker.sh \"\$LATEST_DMP\"
    
    echo '🔄 Step 3: Creating materialized view...'
    uv run python data_pipeline/create_materialized_view.py
    
    echo '📊 Step 4: Indexing to Elasticsearch...'
    uv run python data_pipeline/sql2es_optimized.py
    
    echo '✅ Pipeline complete!'
"

# Detach from screen: Ctrl+A then D
# Reattach later: screen -r data-pipeline
```

**Alternative: Run steps individually**
```bash
# Step 1: Download data
docker compose -f docker-compose.prod.yml run --rm api \
  uv run python data_pipeline/download_aact.py --output-dir /app/data/downloads

# Step 2: Import to PostgreSQL
docker compose -f docker-compose.prod.yml run --rm api \
  bash -c "bash data_pipeline/import_aact_data_docker.sh /app/data/downloads/*.dmp"

# Step 3: Create materialized view
docker compose -f docker-compose.prod.yml run --rm api \
  uv run python data_pipeline/create_materialized_view.py

# Step 4: Index to Elasticsearch
docker compose -f docker-compose.prod.yml run --rm api \
  uv run python data_pipeline/sql2es_optimized.py
```

### 8. Start the API

```bash
# Start the FastAPI service
docker compose -f docker-compose.prod.yml up -d api

# Check status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f api
```

### 9. Test the API

```bash
# Health check
curl http://localhost:8000/api/health

# API documentation
curl http://localhost:8000/docs

# From outside RunPod (use your pod's public IP)
curl http://YOUR_POD_IP:8000/api/health
```

### 10. Access from Outside

Your API will be available at:
- **API Endpoint:** `http://YOUR_POD_IP:8000`
- **API Docs:** `http://YOUR_POD_IP:8000/docs`

**Get your public IP:**
```bash
curl ifconfig.me
```

---

## 🔧 Management Commands

### View Logs
```bash
# All services
docker compose -f docker-compose.prod.yml logs -f

# Specific service
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f postgres
docker compose -f docker-compose.prod.yml logs -f elasticsearch
```

### Restart Services
```bash
# Restart API only
docker compose -f docker-compose.prod.yml restart api

# Restart all
docker compose -f docker-compose.prod.yml restart
```

### Stop Services
```bash
# Stop all
docker compose -f docker-compose.prod.yml down

# Stop and remove volumes (⚠️ deletes all data!)
docker compose -f docker-compose.prod.yml down -v
```

### Update Application
```bash
# Pull latest code
git pull

# Rebuild and restart API
docker compose -f docker-compose.prod.yml up -d --build api
```

### Database Access
```bash
# Connect to PostgreSQL
docker compose -f docker-compose.prod.yml exec postgres psql -U postgres -d aact

# Run SQL query
docker compose -f docker-compose.prod.yml exec postgres \
  psql -U postgres -d aact -c "SELECT COUNT(*) FROM ctgov.studies;"
```

### Check Resource Usage
```bash
# Docker stats
docker stats

# Disk usage
df -h
docker system df
```

---

## 🐛 Troubleshooting

### Elasticsearch Won't Start

**Error:** `max virtual memory areas vm.max_map_count [65530] is too low`

**Fix:**
```bash
# Increase vm.max_map_count
sysctl -w vm.max_map_count=262144

# Make it permanent
echo "vm.max_map_count=262144" >> /etc/sysctl.conf
```

### Out of Memory

**Symptoms:** Services crashing, slow performance

**Fix:**
```bash
# Reduce Elasticsearch memory in docker-compose.prod.yml
# Change ES_JAVA_OPTS from -Xms1g -Xmx1g to -Xms512m -Xmx512m

# Edit file
nano docker-compose.prod.yml

# Find this line and change:
- "ES_JAVA_OPTS=-Xms512m -Xmx512m"  # Reduced from 1g

# Restart
docker compose -f docker-compose.prod.yml restart elasticsearch
```

### Port Already in Use

**Error:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Fix:**
```bash
# Find what's using the port
lsof -i :8000

# Kill the process
kill -9 PID

# Or change port in .env.prod
API_PORT=8080
```

### Data Pipeline Failed

**Check logs:**
```bash
docker compose -f docker-compose.prod.yml logs api
```

**Common issues:**
- Not enough disk space: `df -h`
- Network timeout: Try again
- PostgreSQL not ready: Wait longer before running pipeline

---

## 💰 Cost Optimization

### For Testing/Development
```bash
# Stop when not in use
docker compose -f docker-compose.prod.yml down

# Stop RunPod instance from dashboard
```

### For Production
- Use **On-Demand** pricing (~$0.20-0.40/hour)
- Or **Spot** pricing (~50% cheaper, but can be interrupted)

### Resource Monitoring
```bash
# Install htop
apt-get install -y htop

# Monitor resources
htop
```

---

## 🔐 Security

### Basic Security Setup

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install firewall
apt-get install -y ufw

# Allow only necessary ports
ufw allow 22/tcp    # SSH
ufw allow 8000/tcp  # API
ufw enable

# Change default SSH port (optional)
nano /etc/ssh/sshd_config
# Change Port 22 to Port 2222
systemctl restart sshd
```

### Use Strong Passwords

Edit `.env.prod`:
```bash
POSTGRES_PASSWORD=$(openssl rand -base64 32)
```

---

## 📊 Monitoring

### Basic Health Check Script

Create `check_health.sh`:
```bash
#!/bin/bash
echo "=== Services Status ==="
docker compose -f docker-compose.prod.yml ps

echo -e "\n=== API Health ==="
curl -s http://localhost:8000/api/health | jq .

echo -e "\n=== PostgreSQL ==="
docker compose -f docker-compose.prod.yml exec -T postgres \
  psql -U postgres -d aact -c "SELECT COUNT(*) as study_count FROM ctgov.studies;"

echo -e "\n=== Elasticsearch ==="
curl -s http://localhost:9200/_cluster/health | jq .

echo -e "\n=== Disk Usage ==="
df -h | grep -E "Filesystem|/dev/root"

echo -e "\n=== Memory Usage ==="
free -h
```

Run it:
```bash
chmod +x check_health.sh
./check_health.sh
```

---

## 🚀 Quick Command Reference

```bash
# Start everything
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose -f docker-compose.prod.yml logs -f api

# Restart API
docker compose -f docker-compose.prod.yml restart api

# Stop everything
docker compose -f docker-compose.prod.yml down

# Update code
git pull && docker compose -f docker-compose.prod.yml up -d --build api

# Health check
curl http://localhost:8000/api/health
```

---

## 📝 Checklist

### Initial Setup
- [ ] Create RunPod instance (16 vCPU, 32GB RAM, 50GB storage)
- [ ] SSH into instance
- [ ] Install Docker & Docker Compose (if needed)
- [ ] Clone/upload application
- [ ] Configure `.env.prod` with API keys and passwords
- [ ] Set `vm.max_map_count` for Elasticsearch
- [ ] Start PostgreSQL and Elasticsearch
- [ ] Run data pipeline (30-60 minutes)
- [ ] Start API service
- [ ] Test API endpoints
- [ ] Note public IP for external access

### Regular Maintenance
- [ ] Monitor disk space weekly
- [ ] Check logs for errors
- [ ] Update dependencies monthly
- [ ] Backup PostgreSQL data (if needed)
- [ ] Check RunPod billing

---

## 🆘 Need Help?

- **RunPod Docs:** https://docs.runpod.io/
- **Docker Docs:** https://docs.docker.com/
- **Check logs:** `docker compose -f docker-compose.prod.yml logs -f`
- **Health check:** `curl http://localhost:8000/api/health`

---

**🎉 That's it! Your Clinical Trial Assistant is now running on RunPod!**
