import json
import os
import uuid
from datetime import datetime

HISTORY_FILE = "chat_history.json"


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_history_file():
    if not os.path.exists(HISTORY_FILE):
        return []

    if os.path.getsize(HISTORY_FILE) == 0:
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []


def write_history_file(sessions):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(sessions, file, indent=4, ensure_ascii=False)


def load_chat_history():
    history = read_history_file()

    if not history:
        return []

    first_item = history[0]
    if "messages" in first_item:
        return history

    sessions = []
    for item in history:
        session = {
            "id": str(uuid.uuid4()),
            "title": item.get("question", "Untitled Chat")[:40],
            "created_at": item.get("timestamp", now_text()),
            "updated_at": item.get("timestamp", now_text()),
            "messages": [
                {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "answer_key": "",
                    "sources": item.get("sources", []),
                    "mode": item.get("mode", "qa"),
                    "timestamp": item.get("timestamp", now_text())
                }
            ]
        }
        sessions.append(session)

    write_history_file(sessions)
    return sessions


def create_session(title="New Chat"):
    sessions = load_chat_history()
    session = {
        "id": str(uuid.uuid4()),
        "title": title,
        "mode": "",
        "created_at": now_text(),
        "updated_at": now_text(),
        "messages": []
    }
    sessions.append(session)
    write_history_file(sessions)
    return session


def get_session(session_id):
    sessions = load_chat_history()
    for session in sessions:
        if session["id"] == session_id:
            return session
    return None


def add_message(session_id, question, answer, sources, mode="qa", answer_key="", quiz_items=None, quiz_meta=None):
    sessions = load_chat_history()

    session = None
    for item in sessions:
        if item["id"] == session_id:
            session = item
            break

    if session is None:
        session = create_session(question[:40] or "New Chat")
        sessions = load_chat_history()
        session_id = session["id"]
        for item in sessions:
            if item["id"] == session_id:
                session = item
                break

    message = {
        "question": question,
        "answer": answer,
        "answer_key": answer_key,
        "quiz_items": quiz_items or [],
        "quiz_meta": quiz_meta or {},
        "quiz_submission": None,
        "sources": sources,
        "mode": mode,
        "timestamp": now_text()
    }

    if not session["messages"]:
        session["title"] = question[:40] or "New Chat"
        session["mode"] = mode

    session["messages"].append(message)
    session["updated_at"] = now_text()
    write_history_file(sessions)
    return session


def update_quiz_submission(session_id, message_index, selected_answers):
    from get_output import repair_quiz_items_from_text

    sessions = load_chat_history()
    session = None
    for item in sessions:
        if item["id"] == session_id:
            session = item
            break

    if session is None or message_index < 0 or message_index >= len(session.get("messages", [])):
        return None

    message = session["messages"][message_index]
    quiz_items = repair_quiz_items_from_text(message)
    message["quiz_items"] = quiz_items
    details = []
    score = 0

    for index, item in enumerate(quiz_items):
        number = str(item.get("number", ""))
        selected = selected_answers.get(str(index)) or selected_answers.get(number, "")
        correct = item.get("correct", "")
        is_correct = bool(selected and correct and selected == correct)
        if is_correct:
            score += 1
        details.append({
            "number": item.get("number"),
            "question": item.get("question", ""),
            "selected": selected,
            "correct": correct,
            "is_correct": is_correct,
            "explanation": item.get("explanation", ""),
            "source": item.get("source", "")
        })

    message["quiz_submission"] = {
        "selected_answers": selected_answers,
        "score": score,
        "total": len(quiz_items),
        "details": details,
        "submitted_at": now_text()
    }
    session["updated_at"] = now_text()
    write_history_file(sessions)
    return session


def save_chat(question, answer, sources, mode="qa"):
    session = create_session(question[:40] or "New Chat")
    add_message(session["id"], question, answer, sources, mode)


def clear_chat_history():
    write_history_file([])
