import json
import os
from datetime import datetime

HISTORY_FILE = "chat_history.json"

def load_chat_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    if os.path.getsize(HISTORY_FILE) == 0:
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []


def save_chat(question, answer, sources, mode="qa"):
    history = load_chat_history()

    chat = {
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sources": sources,
        "mode": mode
    }

    history.append(chat)

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)

def clear_chat_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump([], file, indent=4)

