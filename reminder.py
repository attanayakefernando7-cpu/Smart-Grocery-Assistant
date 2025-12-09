# reminder.py
from typing import List
import json
import os
import numpy as np

PURCHASE_HISTORY_FILE = "purchase_history.json"
os.makedirs(os.path.dirname(PURCHASE_HISTORY_FILE) or ".", exist_ok=True)
open(PURCHASE_HISTORY_FILE, "a").close()
try:
    with open(PURCHASE_HISTORY_FILE, "r") as f:
        json.load(f)
except Exception:
    with open(PURCHASE_HISTORY_FILE, "w") as f:
        json.dump([], f)

class ReminderSystem:
    def __init__(self, history_path: str = PURCHASE_HISTORY_FILE):
        self.path = history_path

    def _load(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
                return data if data else []
        except (json.JSONDecodeError, IOError):
            return []

    def _save(self, history):
        with open(self.path, "w") as f:
            json.dump(history, f, indent=2)

    def log_purchase(self, item: str):
        history = self._load()
        history.append({"item": item})
        self._save(history)

    def _top_frequent(self, top_n: int = 5):
        history = self._load()
        items = [entry.get("item") for entry in history]
        arr = np.array(items, dtype=object)
        unique, counts = np.unique(arr, return_counts=True)
        order = np.argsort(counts)[::-1]
        ordered = unique[order]
        return ordered[:top_n].tolist()

    def check_reminders(self, current_list: List[str], top_n: int = 5) -> List[str]:
        current_arr = np.array(current_list, dtype=object) if len(current_list) else np.array([], dtype=object)
        top_items = np.array(self._top_frequent(top_n * 2), dtype=object)
        mask = np.isin(top_items, current_arr)
        result = top_items[~mask][:top_n].tolist()
        return result
