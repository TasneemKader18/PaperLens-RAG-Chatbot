import json
import os
import threading
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHAT_LOG = os.path.join(DATA_DIR, "chat_history.json")
UPLOAD_LOG = os.path.join(DATA_DIR, "upload_history.json")
EVENT_LOG = os.path.join(DATA_DIR, "events.json")

_lock = threading.Lock()


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _read_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _append_json(path, entry):
    with _lock:
        _ensure_dir()
        data = _read_json(path)
        data.append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def _now():
    return datetime.now(timezone.utc).isoformat()


def log_chat(question, answer, sources, duration_s=None):
    _append_json(CHAT_LOG, {
        "timestamp": _now(),
        "question": question,
        "answer": answer,
        "sources": sources,
        "duration_s": duration_s,
    })


def log_upload(added, skipped, total_chunks):
    _append_json(UPLOAD_LOG, {
        "timestamp": _now(),
        "added": added,
        "skipped": skipped,
        "total_chunks": total_chunks,
    })


def log_event(event_type, detail=None):
    _append_json(EVENT_LOG, {
        "timestamp": _now(),
        "event": event_type,
        "detail": detail,
    })


def get_chat_history():
    return _read_json(CHAT_LOG)


def get_upload_history():
    return _read_json(UPLOAD_LOG)


def get_events():
    return _read_json(EVENT_LOG)
