#!/usr/bin/env python3
from __future__ import annotations

import os
import sqlite3

import config.settings as cfg
from utils.control_plane import ControlPlane
from utils.html_report import write_html_report

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, "data", "task_history.db")


def main() -> int:
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM tasks")
            cur.execute("DELETE FROM logs")
            conn.commit()
            print(f"Cleared all tasks and logs from: {DB_PATH}")
        finally:
            conn.close()
    else:
        print(f"No task DB found at: {DB_PATH}")

    control = ControlPlane(cfg.CONTROL_DB_PATH)
    control.reset()
    print(f"Reset dashboard state in: {cfg.CONTROL_DB_PATH}")
    write_html_report(cfg.HTML_REPORT_PATH, None, [])
    print(f"Reset HTML report in: {cfg.HTML_REPORT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
