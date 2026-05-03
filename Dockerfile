FROM python:3.11-slim

WORKDIR /app

# System libraries required by sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

# Install CPU-only torch + torchvision first (avoids pulling the 2+ GB CUDA build)
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Install the rest of the app dependencies
# NOTE: The app uses a NIM/OpenAI-compatible endpoint for LLM inference
# (configured via NIM_BASE_URL in .env) — no GPU required at runtime.
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
