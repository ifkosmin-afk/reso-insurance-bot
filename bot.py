"""
РЕСО-Гарантия Telegram-бот для страховых агентов.
RAG: BM25 поиск по базе знаний + LLM для формирования ответа.
"""

import os
import re
import pickle
import logging
from openai import OpenAI
from rank_bm25 import BM25Okapi
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ── Config ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

INDEX_PATH  = os.path.join(os.path.dirname(__file__), "reso_bm25.pkl")
TOP_K       = 8
LLM_MODEL   = os.environ.get("LLM_MODEL", "gpt-4o-mini")
MAX_CONTEXT = 6000
# ────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Инициализация OpenAI клиента
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

# ── Загрузка BM25 индекса ────────────────────────────────────────────────────
with open(INDEX_PATH, "rb") as fh:
    _store = pickle.load(fh)
bm25: BM25Okapi = _store["bm25"]
chunks: list    = _store["chunks"]
logger.info(f"BM25 index loaded: {len(chunks)} chunks")


def tokenize(text: str) -> list:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def retrieve(query: str, top_k: int = TOP_K) -> list:
    """BM25 поиск: возвращает top_k наиболее релевантных фрагментов."""
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    seen = set()
    for idx in top_indices:
        chunk = chunks[idx]
        key = chunk["text"][:100]
        if key not in seen and scores[idx] > 0:
            seen.add(key)
            results.append({**chunk, "score": float(scores[idx])})
    return results


def build_context(retrieved: list) -> str:
    """Форматирует найденные фрагменты в контекст для LLM."""
    parts = []
    total = 0
    for item in retrieved:
        header = f"[Документ: {item['doc_name']} | Файл: {item['file_name']}]"
        block = f"{header}\n{item['text']}"
        if total + len(block) > MAX_CONTEXT:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = """Ты — умный помощник для страховых агентов компании РЕСО-Гарантия.
Твоя задача — давать точные и понятные ответы на вопросы агентов, опираясь ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты базы знаний.

Правила:
1. Отвечай только на основе предоставленного контекста. Не выдумывай факты.
2. Если ответ есть в контексте — сформулируй его чётко, структурировано, с конкретными цифрами и условиями.
3. Если ответа нет в контексте — честно скажи: «В базе знаний недостаточно информации по этому вопросу. Уточните у куратора или в методическом отделе.»
4. Стиль: профессиональный, лаконичный, ориентированный на помощь агенту в продаже продукта.
5. Используй структуру ответа: краткий вывод → детали → важные условия/исключения (если есть).
6. Не упоминай, что ты ИИ или что используешь базу знаний — просто отвечай по существу."""


def ask_llm(question: str, context: str) -> str:
    """Вызов LLM с найденным контекстом и вопросом агента."""
    if not context.strip():
        return "В базе знаний недостаточно информации по этому вопросу. Уточните у куратора или в методическом отделе."

    user_message = f"""Контекст из базы знаний:
{context}

Вопрос агента: {question}"""

    # Параметры запроса — без thinking для совместимости с разными моделями
    kwargs = dict(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    # Если используется Claude через прокси — отключаем thinking
    if "claude" in LLM_MODEL.lower():
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    response = openai_client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return content.strip() if content else "Извините, не удалось получить ответ. Попробуйте переформулировать вопрос."


# ── Telegram handlers ────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Привет! Я — ассистент страхового агента РЕСО-Гарантия.\n\n"
        "Задайте мне вопрос о страховых продуктах, тарифах, условиях страхования или "
        "переговорных техниках — и я отвечу на основе базы знаний компании.\n\n"
        "Примеры вопросов:\n"
        "• Какие риски покрывает полис «Премиум-защита»?\n"
        "• Сколько стоит «Подорожник» для клиентов РЕСО?\n"
        "• Что такое «Домовой» и какие у него опции?\n"
        "• Как предложить НС при продаже ОСАГО?\n\n"
        "Просто напишите свой вопрос ✍️"
    )
    await update.message.reply_text(text)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📚 *Как пользоваться ботом:*\n\n"
        "Просто задайте вопрос текстом — бот найдёт ответ в базе знаний РЕСО-Гарантия.\n\n"
        "*Доступные команды:*\n"
        "/start — приветствие и инструкция\n"
        "/help — эта справка\n"
        "/products — список продуктов в базе знаний\n\n"
        "*База знаний включает:*\n"
        "• Продукты НС: Премиум-защита, Подорожник, Базовая защита, Активная защита, Пассажир НС\n"
        "• Страхование имущества: Домовой (коробочный и классический)\n"
        "• Кросс-продажи: НС к ОСАГО и КАСКО\n"
        "• Ипотечные продукты: доп. полис к Ипотеке и Заёмщику Сбербанка\n"
        "• Переговорные листы и шпаргалки для агентов"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_products(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📋 *Продукты в базе знаний РЕСО-Гарантия:*\n\n"
        "🔹 *НС — Несчастный случай:*\n"
        "  • Премиум-защита (до 10 млн руб., 18–75 лет)\n"
        "  • Подорожник (4 варианта, скидка для клиентов РЕСО)\n"
        "  • Базовая защита (4 варианта, от 500 до 2000 руб.)\n"
        "  • Активная защита (для ЗОЖ и спорта, 1–75 лет)\n"
        "  • НС Пассажир (к ОСАГО/КАСКО)\n\n"
        "🔹 *Имущество:*\n"
        "  • Домовой коробочный и классический\n"
        "  • Всё включено\n\n"
        "🔹 *Ипотека:*\n"
        "  • Доп. полис к Ипотеке / Заёмщику Сбербанка\n\n"
        "🔹 *Переговорные техники:*\n"
        "  • Скрипты продаж НС от ОСАГО и КАСКО\n"
        "  • Шпаргалки по продуктам"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    if not question:
        return

    logger.info(f"User {update.effective_user.id}: {question[:80]}")

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    try:
        retrieved = retrieve(question)
        ctx = build_context(retrieved)
        answer = ask_llm(question, ctx)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        answer = "⚠️ Произошла ошибка при обработке запроса. Попробуйте ещё раз."

    await update.message.reply_text(answer)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is not set!")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("products", cmd_products))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started. Polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
