# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_SYSTEM_PYTHON=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy the entire project first
COPY . .

# Install dependencies and the project using UV
RUN uv sync --frozen

# Install langgraph-cli with inmemory support
RUN uv pip install "langgraph-cli[inmem]"

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app || true && \
    chown -R app:app /app

# Expose port 2024
EXPOSE 2024

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:2024/health || exit 1

# Run the application
CMD ["uv", "run", "langgraph", "dev", "--host", "0.0.0.0", "--port", "2024"]