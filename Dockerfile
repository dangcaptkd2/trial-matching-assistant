FROM python:3.12-slim

# Install system dependencies for both API and data pipeline
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files AND README (needed for package build)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies using uv
RUN uv sync

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/downloads /app/data/logs

# Expose port
EXPOSE 8000

# Default command runs the API
# Can be overridden in docker-compose for pipeline tasks
CMD ["uv", "run", "python", "-m", "src.api"]
