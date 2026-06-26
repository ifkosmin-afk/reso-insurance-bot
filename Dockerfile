FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
RUN pip install --no-cache-dir \
    "python-telegram-bot==22.8" \
    "openai>=1.0.0" \
    "rank_bm25>=0.2.2" \
    "numpy>=1.24.0"

# Copy project files
COPY . .

# Build BM25 index if not present
RUN python build_index_bm25.py

# Run the bot
CMD ["python", "bot.py"]
