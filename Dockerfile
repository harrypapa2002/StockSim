FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    procps \
    htop \
    vim \
    nano \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p logs charts reports configs && \
    chmod 777 logs charts reports configs

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_DIR=/app/logs
ENV RABBITMQ_HOST=rabbitmq

# Create entrypoint script for flexibility
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Default to interactive bash, but allow override
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]