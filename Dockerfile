# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies and curl (for uv installation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv sync
RUN uv sync --frozen

# Copy application files
COPY predict.py .
COPY *.pkl .

# Expose the port the app runs on
EXPOSE 9696

# Health check (optional - can be removed if not needed)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:9696/docs || exit 1

# Run the application using uv
CMD ["uv", "run", "python", "predict.py"]

