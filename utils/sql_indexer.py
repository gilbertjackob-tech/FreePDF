"""Lightweight SQLIndexer shim using a JSON-backed dict.

This is a compatibility layer so the app can store and retrieve file_id ->
filename mappings without requiring a full SQL setup during local dev.
"""
import json
import os
from threading import Lock

class SQLIndexer:
    def __init__(self, path: str):
        self._path = path
        self._lock = Lock()
        self._data = {}
        try:
            if os.path.exists(self._path):
                with open(self._path, 'r', encoding='utf-8') as fh:
                    self._data = json.load(fh)
        except Exception:
            self._data = {}

    def _persist(self):
        try:
            with self._lock:
                with open(self._path, 'w', encoding='utf-8') as fh:
                    json.dump(self._data, fh)
        except Exception:
            pass

    def set(self, key: str, value: str):
        self._data[key] = value
        self._persist()

    def get(self, key: str):
        return self._data.get(key)

    def delete(self, key: str):
        if key in self._data:
            del self._data[key]
            self._persist()
