#!/bin/bash
# Quick start script for Clinical Trial Assistant

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║   Clinical Trial Assistant - Quick Start                 ║
║   Docker Setup                                            ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating from .env.example...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env file${NC}"
        echo -e "${YELLOW}⚠️  Please edit .env and add your OPENAI_API_KEY${NC}"
        read -p "Press Enter to continue after editing .env..."
    else
        echo -e "${YELLOW}⚠️  .env.example not found. Please create .env manually${NC}"
        exit 1
    fi
fi

# Check if OpenAI API key is set
if ! grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY not set in .env${NC}"
    echo -e "${YELLOW}   Please add your OpenAI API key to .env${NC}"
    read -p "Press Enter to continue anyway..."
fi

echo ""
echo -e "${BLUE}📦 Step 1/4: Building Docker images...${NC}"
docker-compose build

echo ""
echo -e "${BLUE}🚀 Step 2/4: Starting services...${NC}"
docker-compose up -d postgres elasticsearch

echo ""
echo -e "${BLUE}⏳ Step 3/4: Waiting for services to be healthy...${NC}"
echo "   This may take 30-60 seconds..."

# Wait for PostgreSQL
echo -n "   Waiting for PostgreSQL..."
for i in {1..30}; do
    if docker-compose exec -T postgres pg_isready -U postgres -d aact > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    sleep 2
    echo -n "."
done

# Wait for Elasticsearch
echo -n "   Waiting for Elasticsearch..."
for i in {1..30}; do
    if curl -sf http://localhost:9200/_cluster/health > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    sleep 2
    echo -n "."
done

echo ""
echo -e "${BLUE}🌐 Step 4/4: Starting API...${NC}"
docker-compose up -d api

echo ""
echo -e "${GREEN}✅ All services started successfully!${NC}"
echo ""
echo -e "${BLUE}Service URLs:${NC}"
echo "   API:            http://localhost:8000"
echo "   API Docs:       http://localhost:8000/docs"
echo "   PostgreSQL:     localhost:5432"
echo "   Elasticsearch:  http://localhost:9200"
echo ""
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo ""
echo "1. Run the data pipeline to populate the database:"
echo -e "   ${BLUE}./docker-manage.sh pipeline${NC}"
echo ""
echo "2. Or run individual pipeline steps:"
echo -e "   ${BLUE}./docker-manage.sh pipeline-download${NC}   # Download AACT data"
echo -e "   ${BLUE}./docker-manage.sh pipeline-import${NC}     # Import to PostgreSQL"
echo -e "   ${BLUE}./docker-manage.sh pipeline-index${NC}      # Index to Elasticsearch"
echo ""
echo "3. Test the API:"
echo -e "   ${BLUE}curl http://localhost:8000/api/health${NC}"
echo ""
echo "4. View logs:"
echo -e "   ${BLUE}./docker-manage.sh logs${NC}"
echo ""
echo "5. Stop services:"
echo -e "   ${BLUE}./docker-manage.sh stop${NC}"
echo ""
echo -e "${GREEN}For more commands, run: ${BLUE}./docker-manage.sh help${NC}"
echo ""
