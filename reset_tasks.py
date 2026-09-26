#!/usr/bin/env python3
from __future__ import annotations

import os
import sqlite3

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, "data", "task_history.db")


def main() -> int:
    if not os.path.exists(DB_PATH):
        print(f"No task DB found at: {DB_PATH}")
        return 0

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM tasks")
        cur.execute("DELETE FROM logs")
        conn.commit()
        print(f"Cleared all tasks and logs from: {DB_PATH}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
