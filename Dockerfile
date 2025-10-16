# Stable base with manylinux wheels available
FROM python:3.10-slim

# Saner defaults + Streamlit settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Minimal build deps and libgomp (needed by torch/faiss wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Keep pip toolchain predictable and pre-pin numpy<2 to avoid source builds
RUN python -m pip install --upgrade pip "setuptools<70" wheel && \
    python -m pip install "numpy<2"

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -vvv --no-cache-dir -r /app/requirements.txt

# Copy the app
COPY . /app

# Create necessary directories
RUN mkdir -p data/docs data/images data/audio data/video vectorstore

# Expose Streamlit
EXPOSE 8501

# Run app from root
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]