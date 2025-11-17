FROM python:3.11.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and source code
COPY requirements.txt ./
COPY src/ ./src/
COPY api/ ./
COPY dataset/ ./dataset/
COPY data_processing_hourly/ ./data_processing_hourly/
COPY processed_data/ ./processed_data/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose Hugging Face Spaces port
EXPOSE 7860

# Health check for API readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:7860/api/v1/health || exit 1

# Run FastAPI with uvicorn on port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]