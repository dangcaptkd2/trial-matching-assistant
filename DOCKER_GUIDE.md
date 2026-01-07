# Docker Deployment Guide

Complete guide for running the Clinical Trial Assistant with Docker.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available for Docker
- ~10GB free disk space (for data and images)

### 1. Start Services

```bash
# Using the management script (recommended)
./docker-manage.sh start

# Or manually with docker-compose
docker-compose up -d
```

This starts:
- ✅ PostgreSQL (port 5432)
- ✅ Elasticsearch (port 9200)
- ✅ FastAPI application (port 8000)

### 2. Run Data Pipeline

```bash
# Run the complete pipeline (download → import → index)
./docker-manage.sh pipeline

# Or run individual steps:
./docker-manage.sh pipeline-download   # Download AACT data
./docker-manage.sh pipeline-import     # Import to PostgreSQL
./docker-manage.sh pipeline-index      # Index to Elasticsearch
```

### 3. Access the API

```bash
# Health check
curl http://localhost:8000/api/health

# API documentation
open http://localhost:8000/docs
```

---

## 📋 Available Services

### Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| **API** | http://localhost:8000 | - |
| **API Docs** | http://localhost:8000/docs | - |
| **PostgreSQL** | localhost:5432 | postgres/postgres |
| **Elasticsearch** | http://localhost:9200 | - |

---

## 🛠️ Management Commands

Use the `docker-manage.sh` script for easy management:

### Service Control

```bash
./docker-manage.sh start      # Start all services
./docker-manage.sh stop       # Stop all services
./docker-manage.sh restart    # Restart all services
./docker-manage.sh status     # Show service status
```

### Logs

```bash
./docker-manage.sh logs          # All service logs
./docker-manage.sh logs-api      # API logs only
./docker-manage.sh logs-postgres # PostgreSQL logs
./docker-manage.sh logs-es       # Elasticsearch logs
```

### Data Pipeline

```bash
./docker-manage.sh pipeline           # Run full pipeline
./docker-manage.sh pipeline-download  # Download only
./docker-manage.sh pipeline-import    # Import only
./docker-manage.sh pipeline-index     # Index only
```

### Health Checks

```bash
./docker-manage.sh health    # Check all services
```

### Shell Access

```bash
./docker-manage.sh shell-api       # Bash in API container
./docker-manage.sh shell-postgres  # PostgreSQL CLI
./docker-manage.sh shell-es        # Test Elasticsearch
```

### Cleanup

```bash
./docker-manage.sh clean      # Remove containers (keep data)
./docker-manage.sh clean-all  # Remove containers AND data ⚠️
```

---

## 📦 Docker Compose Services

### `postgres`
- **Image**: postgres:15-alpine
- **Port**: 5432
- **Volume**: postgres_data (persists data)
- **Database**: aact
- **User**: postgres
- **Password**: postgres

### `elasticsearch`
- **Image**: elasticsearch:7.17.16
- **Port**: 9200
- **Volume**: elasticsearch_data (persists indices)
- **Memory**: 512MB heap
- **Mode**: Single-node
- **Security**: Disabled (for development)

### `api`
- **Build**: From local Dockerfile
- **Port**: 8000
- **Depends on**: postgres, elasticsearch
- **Health check**: /api/health endpoint
- **Auto-restart**: yes

### `data-pipeline`
- **Build**: From local Dockerfile
- **Profile**: init (manual trigger only)
- **Purpose**: Run data pipeline tasks
- **Depends on**: postgres, elasticsearch
- **Runs**: On-demand only

---

## 🔄 Data Pipeline Steps

The data pipeline performs these steps in order:

### Step 1: Download AACT Data
```bash
# Downloads latest clinical trial data from AACT
docker-compose run --rm data-pipeline sh -c \
  "uv run python data_pipeline/download_aact.py --output-dir /app/data/downloads"
```

### Step 2: Import to PostgreSQL
```bash
# Imports .dmp file into PostgreSQL
docker-compose run --rm data-pipeline sh -c \
  "bash data_pipeline/import_aact_data.sh /app/data/downloads/[file].dmp"
```

### Step 3: Create Materialized View
```bash
# Creates optimized view for Elasticsearch indexing
docker-compose run --rm data-pipeline sh -c \
  "uv run python data_pipeline/create_materialized_view.py"
```

### Step 4: Index to Elasticsearch
```bash
# Indexes data from PostgreSQL to Elasticsearch
docker-compose run --rm data-pipeline sh -c \
  "uv run python data_pipeline/sql2es_optimized.py"
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# OpenAI
OPENAI_API_KEY=your_key_here

# LangSmith (optional)
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true

# These are set in docker-compose.yml:
# POSTGRES_HOST=postgres
# ELASTICSEARCH_URL=http://elasticsearch:9200
```

### Resource Limits

Edit `docker-compose.yml` to adjust:

```yaml
# Elasticsearch memory
elasticsearch:
  environment:
    - "ES_JAVA_OPTS=-Xms1g -Xmx1g"  # Increase to 1GB

# API resources (add deploy section)
api:
  deploy:
    resources:
      limits:
        cpus: '1.0'
        memory: 1G
```

---

## 🐛 Troubleshooting

### Services Won't Start

```bash
# Check Docker is running
docker info

# Check service status
./docker-manage.sh status

# View logs for errors
./docker-manage.sh logs
```

### Elasticsearch Fails to Start

**Issue**: "max virtual memory areas vm.max_map_count [65530] is too low"

**Solution**:
```bash
# On Linux
sudo sysctl -w vm.max_map_count=262144

# On macOS (increase Docker memory in Docker Desktop settings)
# Settings → Resources → Memory: 4GB minimum
```

### PostgreSQL Connection Failed

```bash
# Check if PostgreSQL is healthy
docker-compose exec postgres pg_isready -U postgres

# Test connection
docker-compose exec postgres psql -U postgres -d aact -c "SELECT version();"
```

### API Can't Connect to Services

**Issue**: Connection refused errors

**Solution**:
- Services must use container names (`postgres`, `elasticsearch`) not `localhost`
- Check `docker-compose.yml` environment variables
- Ensure services are healthy: `./docker-manage.sh health`

### Pipeline Download Fails

```bash
# Check network connectivity
docker-compose run --rm data-pipeline curl -I https://aact.ctti-clinicaltrials.org

# Check available disk space
df -h ./data/downloads
```

### Out of Disk Space

```bash
# Remove unused Docker resources
docker system prune -a

# Remove old volumes (WARNING: deletes data!)
docker volume prune
```

---

## 📊 Monitoring

### Check Service Health

```bash
# All services
./docker-manage.sh health

# Individual health checks
curl http://localhost:8000/api/health  # API
curl http://localhost:9200/_cluster/health  # Elasticsearch
docker-compose exec postgres pg_isready -U postgres  # PostgreSQL
```

### View Resource Usage

```bash
# All containers
docker stats

# Specific container
docker stats clinical-trial-assistant-api
```

### Database Statistics

```bash
# PostgreSQL
docker-compose exec postgres psql -U postgres -d aact -c \
  "SELECT COUNT(*) FROM ctgov.studies;"

# Elasticsearch
curl http://localhost:9200/aact/_count?pretty
```

---

## 🚀 Deployment to Production

### 1. Update docker-compose.yml

```yaml
# Remove/comment out the data-pipeline service (run separately)
# Add resource limits
# Enable security features

services:
  postgres:
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}  # Use secret
  
  elasticsearch:
    environment:
      - xpack.security.enabled=true
      - ELASTIC_PASSWORD=${ELASTIC_PASSWORD}
```

### 2. Use Docker Secrets

```bash
# Create secrets
echo "your_password" | docker secret create postgres_password -
echo "your_es_password" | docker secret create elastic_password -
```

### 3. Run Behind Reverse Proxy

Use nginx or Traefik for HTTPS:

```yaml
# Add labels for Traefik
api:
  labels:
    - "traefik.enable=true"
    - "traefik.http.routers.api.rule=Host(`your-domain.com`)"
    - "traefik.http.routers.api.tls.certresolver=letsencrypt"
```

---

## 🔄 Update and Maintenance

### Update Application Code

```bash
# Pull latest code
git pull

# Rebuild and restart
./docker-manage.sh rebuild
./docker-manage.sh restart
```

### Update Data

```bash
# Run pipeline to get latest trials
./docker-manage.sh pipeline
```

### Backup Data

```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U postgres aact > backup.sql

# Backup Elasticsearch
curl -X PUT "localhost:9200/_snapshot/backup" \
  -H 'Content-Type: application/json' \
  -d '{"type": "fs", "settings": {"location": "/app/backup"}}'
```

### Restore Data

```bash
# Restore PostgreSQL
docker-compose exec -T postgres psql -U postgres aact < backup.sql

# Restore Elasticsearch
curl -X POST "localhost:9200/_snapshot/backup/snapshot_1/_restore"
```

---

## 📝 Common Workflows

### First Time Setup

```bash
# 1. Start services
./docker-manage.sh start

# 2. Wait for services to be healthy
./docker-manage.sh health

# 3. Run data pipeline
./docker-manage.sh pipeline

# 4. Test API
curl http://localhost:8000/api/health
```

### Development Workflow

```bash
# 1. Make code changes
# 2. Rebuild API container
docker-compose build api

# 3. Restart API
docker-compose restart api

# 4. View logs
./docker-manage.sh logs-api
```

### Data Update Workflow

```bash
# 1. Download new data
./docker-manage.sh pipeline-download

# 2. Import to PostgreSQL
./docker-manage.sh pipeline-import

# 3. Re-index to Elasticsearch
./docker-manage.sh pipeline-index
```

---

## 💡 Tips and Best Practices

1. **Always use the management script** - It handles dependencies and health checks
2. **Monitor logs during pipeline** - `./docker-manage.sh logs -f`
3. **Run pipeline during off-hours** - Can take 30-60 minutes
4. **Keep volumes backed up** - Data is stored in Docker volumes
5. **Set resource limits** - Prevent services from consuming all resources
6. **Use `.env` for secrets** - Never commit secrets to git

---

## 🆘 Getting Help

1. Check logs: `./docker-manage.sh logs`
2. Check service health: `./docker-manage.sh health`
3. View service status: `./docker-manage.sh status`
4. See script help: `./docker-manage.sh help`

For more details, see the main [README.md](README.md)
