FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p logs uploads static/uploads

# Set permissions
RUN chmod 755 uploads/ logs/

# Expose port
EXPOSE 8000

# Start command
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
