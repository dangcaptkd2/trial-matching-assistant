#!/bin/bash
# Docker Setup and Management Script for Clinical Trial Assistant

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

# Function to show help
show_help() {
    cat << EOF
Clinical Trial Assistant - Docker Management Script

Usage: ./docker-manage.sh <command>

Commands:
    start               Start all services (PostgreSQL, Elasticsearch, API)
    stop                Stop all services
    restart             Restart all services
    logs                Show logs from all services
    logs-api            Show logs from API only
    logs-postgres       Show logs from PostgreSQL only
    logs-es             Show logs from Elasticsearch only
    
    pipeline            Run data pipeline (download, import, index)
    pipeline-download   Run only download step
    pipeline-import     Run only import step
    pipeline-index      Run only indexing step
    
    status              Show status of all services
    clean               Remove all containers (keeps volumes)
    clean-all           Remove containers AND volumes (WARNING: deletes data!)
    
    build               Build Docker images
    rebuild             Rebuild images from scratch (no cache)
    
    shell-api           Open shell in API container
    shell-postgres      Open PostgreSQL shell
    shell-es            Test Elasticsearch connection
    
    health              Check health of all services
    help                Show this help message

Examples:
    ./docker-manage.sh start
    ./docker-manage.sh pipeline
    ./docker-manage.sh logs-api
    ./docker-manage.sh health

EOF
}

# Check if docker and docker-compose are installed
check_dependencies() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Get docker-compose command (handles both docker-compose and docker compose)
get_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    else
        echo "docker compose"
    fi
}

# Main command handling
DOCKER_COMPOSE=$(get_docker_compose)

case "${1:-help}" in
    start)
        info "Starting all services..."
        $DOCKER_COMPOSE up -d postgres elasticsearch
        info "Waiting for services to be healthy..."
        sleep 10
        $DOCKER_COMPOSE up -d api
        success "All services started!"
        info "API available at: http://localhost:8000"
        info "Elasticsearch at: http://localhost:9200"
        info "PostgreSQL at: localhost:5432"
        ;;
    
    stop)
        info "Stopping all services..."
        $DOCKER_COMPOSE down
        success "All services stopped!"
        ;;
    
    restart)
        info "Restarting all services..."
        $DOCKER_COMPOSE restart
        success "All services restarted!"
        ;;
    
    logs)
        $DOCKER_COMPOSE logs -f
        ;;
    
    logs-api)
        $DOCKER_COMPOSE logs -f api
        ;;
    
    logs-postgres)
        $DOCKER_COMPOSE logs -f postgres
        ;;
    
    logs-es)
        $DOCKER_COMPOSE logs -f elasticsearch
        ;;
    
    pipeline)
        info "Running full data pipeline..."
        $DOCKER_COMPOSE run --rm data-pipeline
        success "Data pipeline completed!"
        ;;
    
    pipeline-download)
        info "Running download step..."
        $DOCKER_COMPOSE run --rm data-pipeline sh -c "uv run python data_pipeline/download_aact.py --output-dir /app/data/downloads"
        success "Download completed!"
        ;;
    
    pipeline-import)
        info "Running import step..."
        $DOCKER_COMPOSE run --rm data-pipeline sh -c "
            LATEST_DMP=\$(ls -t /app/data/downloads/*.dmp | head -1)
            if [ -f \"\$LATEST_DMP\" ]; then
                bash data_pipeline/import_aact_data_docker.sh \"\$LATEST_DMP\"
            else
                echo 'No .dmp file found in /app/data/downloads'
                exit 1
            fi
        "
        success "Import completed!"
        ;;
    
    pipeline-index)
        info "Running indexing step..."
        $DOCKER_COMPOSE run --rm data-pipeline sh -c "
            uv run python data_pipeline/create_materialized_view.py --refresh || \
            uv run python data_pipeline/create_materialized_view.py && \
            uv run python data_pipeline/sql2es_optimized.py
        "
        success "Indexing completed!"
        ;;
    
    status)
        info "Service Status:"
        $DOCKER_COMPOSE ps
        ;;
    
    clean)
        warning "This will remove all containers but keep data volumes."
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $DOCKER_COMPOSE down
            success "Containers removed!"
        fi
        ;;
    
    clean-all)
        error "WARNING: This will remove all containers AND delete all data!"
        read -p "Are you absolutely sure? (type 'yes' to confirm): " -r
        if [[ $REPLY == "yes" ]]; then
            $DOCKER_COMPOSE down -v
            success "All containers and volumes removed!"
        else
            info "Cancelled."
        fi
        ;;
    
    build)
        info "Building Docker images..."
        $DOCKER_COMPOSE build
        success "Build completed!"
        ;;
    
    rebuild)
        info "Rebuilding Docker images from scratch (no cache)..."
        $DOCKER_COMPOSE build --no-cache
        success "Rebuild completed!"
        ;;
    
    shell-api)
        info "Opening shell in API container..."
        $DOCKER_COMPOSE exec api /bin/bash
        ;;
    
    shell-postgres)
        info "Opening PostgreSQL shell..."
        $DOCKER_COMPOSE exec postgres psql -U postgres -d aact
        ;;
    
    shell-es)
        info "Testing Elasticsearch connection..."
        $DOCKER_COMPOSE exec elasticsearch curl -s http://localhost:9200/_cluster/health?pretty
        ;;
    
    health)
        info "Checking service health..."
        echo ""
        
        info "PostgreSQL:"
        $DOCKER_COMPOSE exec postgres pg_isready -U postgres -d aact && success "Healthy" || error "Unhealthy"
        
        info "Elasticsearch:"
        if $DOCKER_COMPOSE exec elasticsearch curl -sf http://localhost:9200/_cluster/health > /dev/null; then
            success "Healthy"
        else
            error "Unhealthy"
        fi
        
        info "API:"
        if curl -sf http://localhost:8000/api/health > /dev/null 2>&1; then
            success "Healthy"
        else
            error "Unhealthy or not running"
        fi
        ;;
    
    help|--help|-h)
        show_help
        ;;
    
    *)
        error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
