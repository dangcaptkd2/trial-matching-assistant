#!/bin/bash

# Build script for all-in-one Docker container
# This creates a single container with PostgreSQL + Elasticsearch + FastAPI

set -e

echo "========================================="
echo "  Building All-in-One Docker Container"
echo "========================================="
echo ""

# Configuration
IMAGE_NAME="dangcaptkd2/trial-matching-assistant"
TAG="v1.0.1"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "📦 Image: ${FULL_IMAGE}"
echo "🏗️  Platform: linux/amd64"
echo ""

# Check if dump file exists
DUMP_FILE="data/downloads/20260112/postgres.dmp"
if [ -f "$DUMP_FILE" ]; then
    echo "✅ Found dump file: $DUMP_FILE ($(du -sh $DUMP_FILE | cut -f1))"
    echo "   This will be included in the image for automatic import"
    
    # Copy dump file to a standard location for inclusion in image
    mkdir -p data/downloads
    cp "$DUMP_FILE" data/downloads/postgres.dmp
    echo "   Copied to data/downloads/postgres.dmp"
else
    echo "⚠️  Warning: No dump file found at $DUMP_FILE"
    echo "   The container will start without data"
    echo "   You can import data later by mounting a dump file"
fi

echo ""
read -p "Continue with build? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Build cancelled"
    exit 0
fi

echo ""
echo "🔨 Building Docker image..."
echo "   This will take 10-20 minutes (downloading PostgreSQL + Elasticsearch)"
echo ""

# Build the image
docker buildx build \
    --platform linux/amd64 \
    -f Dockerfile.allinone \
    -t "${FULL_IMAGE}" \
    --progress=plain \
    .

echo ""
echo "✅ Build completed successfully!"
echo ""
echo "📊 Image details:"
docker images "${IMAGE_NAME}" --filter "reference=${FULL_IMAGE}"

echo ""
echo "📤 To push to Docker Hub:"
echo "   docker push ${FULL_IMAGE}"
echo ""
echo "🧪 To test locally:"
echo "   docker run -p 8000:8000 -p 5432:5432 -p 9200:9200 ${FULL_IMAGE}"
echo ""
echo "💾 To run with persistent data:"
echo "   docker run -p 8000:8000 -p 5432:5432 -p 9200:9200 \\"
echo "     -v postgres-data:/var/lib/postgresql/data \\"
echo "     -v elasticsearch-data:/var/lib/elasticsearch/data \\"
echo "     ${FULL_IMAGE}"
