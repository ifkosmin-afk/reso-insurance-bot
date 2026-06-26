"""
Build BM25 search index from RESO knowledge base text files.

Этот индексатор:
  • корректно разбивает файлы на отдельные ДОКУМЕНТЫ (по заголовкам "## ДОКУМЕНТ:"),
    устойчиво к строке "_Источник:_" между заголовком и разделителем;
  • чистит служебный мусор (линии "====", строки заголовков/источников) из текста;
  • режет каждый документ на КРУПНЫЕ смысловые чанки по границам абзацев
    (а не вслепую по символам), не разрывая слова и факты;
  • короткие документы (меньше целевого размера) кладёт ЦЕЛИКОМ одним чанком;
  • в текст для поиска подмешивает название документа, чтобы запрос по продукту
    («КАСКО от БПЛА») лучше находил нужный документ.

Формат pickle совместим с bot.py: {"bm25", "chunks":[{text,doc_name,file_name,source}], "tokenized"}.
"""

import os
import re
import pickle
from rank_bm25 import BM25Okapi

# -- Config ------------------------------------------------------------------
KNOWLEDGE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH    = os.path.join(KNOWLEDGE_DIR, "reso_bm25.pkl")

CHUNK_TARGET  = 1300   # целевой размер чанка, символов (крупнее => факты не рвутся)
CHUNK_MAX     = 1800   # жёсткий потолок для одного чанка
CHUNK_OVERLAP = 200    # перекрытие между чанками, символов
# ----------------------------------------------------------------------------


def load_files():
    # Исключаем База_*.txt — это полный дубль категорийных файлов 01-14
    # (143 документа, все уже есть в 01-14; индексация обоих давала задвоение).
    files = sorted(
        f for f in os.listdir(KNOWLEDGE_DIR)
        if f.endswith(".txt")
        and f != "requirements.txt"
        and not f.startswith("База_")
    )
    docs = []
    for fname in files:
        with open(os.path.join(KNOWLEDGE_DIR, fname), "r", encoding="utf-8") as fh:
            docs.append((fname, fh.read()))
    print(f"Loaded {len(docs)} files")
    return docs


DOC_HEADER_RE = re.compile(r"^##\s*ДОКУМЕНТ:\s*(.+?)\s*$", re.MULTILINE)
SEP_LINE_RE   = re.compile(r"^={6,}\s*$", re.MULTILINE)
SRC_LINE_RE   = re.compile(r"^_Источник:.*$", re.MULTILINE)


def clean_text(text):
    text = SEP_LINE_RE.sub("", text)
    text = SRC_LINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_by_document(text, file_name):
    base = file_name.replace(".txt", "")
    matches = list(DOC_HEADER_RE.finditer(text))
    if not matches:
        return [(base, clean_text(text))]
    sections = []
    for i, m in enumerate(matches):
        doc_name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((doc_name, clean_text(text[start:end])))
    return sections


def split_paragraphs(text):
    raw = re.split(r"\n\s*\n", text)
    paras = []
    for p in raw:
        p = p.strip()
        if not p:
            continue
        if len(p) <= CHUNK_MAX:
            paras.append(p)
        else:
            buf = ""
            for line in p.split("\n"):
                if len(buf) + len(line) + 1 > CHUNK_MAX and buf:
                    paras.append(buf.strip())
                    buf = ""
                buf += line + "\n"
            if buf.strip():
                paras.append(buf.strip())
    return paras


def chunk_document(doc_name, doc_text, file_name):
    doc_text = doc_text.strip()
    if not doc_text:
        return []
    paras = split_paragraphs(doc_text)
    chunks_text = []
    buf = ""
    for p in paras:
        if buf and len(buf) + len(p) + 2 > CHUNK_TARGET:
            chunks_text.append(buf.strip())
            tail = buf[-CHUNK_OVERLAP:]
            buf = tail + "\n\n" + p
        else:
            buf = (buf + "\n\n" + p) if buf else p
    if buf.strip():
        chunks_text.append(buf.strip())

    records = []
    for ct in chunks_text:
        records.append({
            "text": f"[{doc_name}]\n{ct}",
            "doc_name": doc_name,
            "file_name": file_name,
            "source": doc_name,
        })
    return records


def tokenize(text):
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def main():
    print("=== Building RESO BM25 index (v2: document-aware, large chunks) ===")
    docs = load_files()
    all_chunks = []
    doc_count = 0
    for file_name, text in docs:
        for doc_name, doc_text in split_by_document(text, file_name):
            doc_count += 1
            all_chunks.extend(chunk_document(doc_name, doc_text, file_name))
    print(f"Documents parsed: {doc_count}")
    print(f"Total chunks: {len(all_chunks)}")
    lens = [len(c["text"]) for c in all_chunks]
    if lens:
        print(f"Chunk length: min={min(lens)} avg={sum(lens)//len(lens)} max={max(lens)}")
    tokenized = [tokenize(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    with open(INDEX_PATH, "wb") as fh:
        pickle.dump({"bm25": bm25, "chunks": all_chunks, "tokenized": tokenized}, fh)
    print(f"BM25 index saved to {INDEX_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
