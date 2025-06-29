FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PIP_NO_CACHE_DIR=1 \
    PORT=9999 \
    COMPLEXITY_HISTORY_LEN=2

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install \
        torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install \
        starlette uvicorn[standard] transformers huggingface_hub \
        git+https://github.com/nginxinc/f5-ai-gateway-sdk-py

# Set working directory and copy source code
WORKDIR /app
COPY complexity-classifier.py ./

# Expose the port (default 9999, can be overridden)
EXPOSE ${PORT}

# Start the server
CMD ["sh", "-c", "uvicorn complexity-classifier:app --host 0.0.0.0 --port ${PORT}"]