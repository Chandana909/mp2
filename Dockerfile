# ML Reliability Platform - Production image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY platform/ platform/
COPY storage/ storage/
COPY monitoring/ monitoring/
COPY lifecycle/ lifecycle/
COPY serving/ serving/
COPY config/ config/

# Create runtime dirs
RUN mkdir -p models storage/metadata storage/audit_logs data

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run API
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
