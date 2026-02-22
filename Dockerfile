# SCL-Alpha runtime image (Python 3.11 required)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl is needed for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port Railway will assign
EXPOSE ${PORT:-8501}

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Run the Streamlit app
CMD streamlit run app/app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true
