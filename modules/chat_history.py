# chat_history.py
import os
import json
import asyncio
from typing import List, Dict
from asyncio import Lock
from functools import partial
from datetime import datetime, timedelta

HISTORY_FILE = "chat_data.json"
history_lock = Lock()

async def load_chat_data() -> Dict[str, Dict]:
    """Load chat data from JSON file"""
    async with history_lock:
        if not os.path.exists(HISTORY_FILE):
            return {}
        loop = asyncio.get_event_loop()
        with open(HISTORY_FILE, "r") as f:
            data = await loop.run_in_executor(None, f.read)
            if not data.strip():
                return {}
            try:
                return json.loads(data)
            except Exception:
                return {}

async def save_chat_data(data: Dict[str, Dict]):
    """Save chat data to JSON file"""
    async with history_lock:
        loop = asyncio.get_event_loop()
        with open(HISTORY_FILE, "w") as f:
            await loop.run_in_executor(None, partial(json.dump, data, f, indent=2))

async def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """Get chat history for a session"""
    data = await load_chat_data()
    session = data.get(session_id, {})
    return session.get("conversations", [])

async def append_chat_history(session_id: str, user: str, bot: str, user_id: str):
    """Append a message to chat history"""
    data = await load_chat_data()
    if session_id not in data:
        data[session_id] = {
            "user_id": user_id,
            "conversations": []
        }
    # Bangladesh Time (UTC+6), 12-hour format with AM/PM
    bd_time = datetime.utcnow() + timedelta(hours=6)
    timestamp = bd_time.strftime('%Y-%m-%d %I:%M:%S %p')
    data[session_id]["conversations"].append({
        "user": user,
        "bot": bot,
        "timestamp": timestamp
    })
    await save_chat_data(data)