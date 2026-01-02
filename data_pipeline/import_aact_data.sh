#!/bin/bash

# Script to import clinical trial data to PostgreSQL container
# Usage: ./import_aact_data.sh [path/to/postgres.dmp]

set -e  # Exit on any error

# Configuration
CONTAINER_NAME="postgres_db"
DB_NAME="aact"
DB_USER="postgres"
DB_PASSWORD="postgres"
DB_HOST="localhost"
DB_PORT="5432"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if dump file is provided
if [ $# -eq 0 ]; then
    print_error "Please provide the path to the PostgreSQL dump file"
    echo "Usage: $0 <path/to/postgres.dmp>"
    exit 1
fi

DUMP_FILE="$1"

# Check if dump file exists
if [ ! -f "$DUMP_FILE" ]; then
    print_error "Dump file not found: $DUMP_FILE"
    exit 1
fi

print_status "Starting AACT data import process..."

# Check if PostgreSQL container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    print_error "PostgreSQL container '$CONTAINER_NAME' is not running"
    print_status "Starting PostgreSQL container..."
    docker-compose up -d postgres
    
    # Wait for container to be ready
    print_status "Waiting for PostgreSQL container to be ready..."
    sleep 10
    
    # Wait for PostgreSQL to be ready
    until docker exec "$CONTAINER_NAME" pg_isready -U "$DB_USER"; do
        print_status "Waiting for PostgreSQL to be ready..."
        sleep 2
    done
fi

# Create aact database
print_status "Creating database '$DB_NAME'..."
docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -c "DROP DATABASE IF EXISTS $DB_NAME;"
docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -c "CREATE DATABASE $DB_NAME;"

# Copy dump file to container
print_status "Copying dump file to container..."
docker cp "$DUMP_FILE" "$CONTAINER_NAME:/tmp/postgres.dmp"

# Restore database from dump
print_status "Restoring database from dump file..."
docker exec "$CONTAINER_NAME" pg_restore \
    -U "$DB_USER" \
    -e \
    -v \
    -O \
    -x \
    -d "$DB_NAME" \
    --no-owner \
    /tmp/postgres.dmp

# Clean up dump file from container
print_status "Cleaning up temporary files..."
docker exec "$CONTAINER_NAME" rm -f /tmp/postgres.dmp

# Verify import
print_status "Verifying import..."
TABLE_COUNT=$(docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')
print_status "Import completed successfully! Database '$DB_NAME' now contains $TABLE_COUNT tables."

print_status "You can connect to the database using:"
echo "  Host: $DB_HOST"
echo "  Port: $DB_PORT"
echo "  Database: $DB_NAME"
echo "  Username: $DB_USER"
echo "  Password: $DB_PASSWORD" 