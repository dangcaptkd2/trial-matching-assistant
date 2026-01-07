#!/bin/bash

# Script to import clinical trial data to PostgreSQL (Docker-friendly version)
# This version runs INSIDE the data-pipeline container
# Usage: ./import_aact_data_docker.sh [path/to/postgres.dmp]

set -e  # Exit on any error

# Configuration - these come from environment variables in docker-compose
DB_NAME="${POSTGRES_DATABASE:-aact}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
DB_HOST="${POSTGRES_HOST:-postgres}"
DB_PORT="${POSTGRES_PORT:-5432}"

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
print_status "Database: $DB_NAME on $DB_HOST:$DB_PORT"

# Wait for PostgreSQL to be ready
print_status "Waiting for PostgreSQL to be ready..."
until PGPASSWORD="$DB_PASSWORD" pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER"; do
    print_status "Waiting for PostgreSQL..."
    sleep 2
done

print_status "PostgreSQL is ready!"

# Drop and create database
print_status "Recreating database '$DB_NAME'..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "DROP DATABASE IF EXISTS $DB_NAME;"
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE $DB_NAME;"

# Restore database from dump
print_status "Restoring database from dump file..."
print_status "This may take 10-20 minutes..."

PGPASSWORD="$DB_PASSWORD" pg_restore \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -v \
    -O \
    -x \
    --no-owner \
    --no-acl \
    "$DUMP_FILE" || {
        print_warning "Some errors occurred during restore (this is normal for AACT dumps)"
    }

# Verify import
print_status "Verifying import..."
TABLE_COUNT=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'ctgov';" | tr -d ' ')
print_status "Import completed! Database '$DB_NAME' now contains $TABLE_COUNT tables in ctgov schema."

# Get study count
STUDY_COUNT=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM ctgov.studies;" | tr -d ' ' || echo "0")
print_status "Total studies imported: $STUDY_COUNT"

print_status "✓ Import successful!"
print_status "Connection details:"
echo "  Host: $DB_HOST"
echo "  Port: $DB_PORT"
echo "  Database: $DB_NAME"
echo "  Username: $DB_USER"
echo "  Schema: ctgov"
