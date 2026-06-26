FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    "python-telegram-bot==22.8" \
    "openai>=1.0.0" \
    "rank_bm25>=0.2.2" \
    "numpy>=1.24.0"

# Copy all project files (includes pre-built reso_bm25.pkl index)
COPY . .

# Run the bot
CMD ["python", "bot.py"]
