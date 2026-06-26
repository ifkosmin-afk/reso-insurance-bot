FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Build BM25 index at container build time
RUN python build_index_bm25.py

# Run the bot
CMD ["python", "bot.py"]
