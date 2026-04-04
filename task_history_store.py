# -*- coding: utf-8 -*-
"""历史任务本地持久化（SQLite，无需单独安装服务）。"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Sequence, Tuple

Row = Tuple[int, str, str, str, str, int, float, str]


class TaskHistoryStore:
    """单表 history：一行一条任务摘要。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    task_type TEXT,
                    source TEXT,
                    model TEXT,
                    objects INTEGER,
                    inference_s REAL,
                    note TEXT
                )
                """
            )
            conn.commit()

    def add(
        self,
        time_str: str,
        task_type: str,
        source: str,
        model: str,
        objects: int,
        inference_s: float,
        note: str,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO history
                (created_at, task_type, source, model, objects, inference_s, note)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    time_str,
                    task_type or "",
                    source or "",
                    model or "",
                    int(objects),
                    float(inference_s),
                    note or "",
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_recent(self, limit: int) -> List[Row]:
        lim = max(1, min(int(limit), 50_000))
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT id, created_at, task_type, source, model, objects, inference_s, note
                FROM history
                ORDER BY datetime(created_at) DESC, id DESC
                LIMIT ?
                """,
                (lim,),
            )
            return [tuple(r) for r in cur.fetchall()]

    def delete_ids(self, ids: Sequence[int]) -> int:
        ids = [int(i) for i in ids if i is not None]
        if not ids:
            return 0
        ph = ",".join("?" * len(ids))
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(f"DELETE FROM history WHERE id IN ({ph})", ids)
            conn.commit()
            return cur.rowcount

    def delete_all(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM history")
            conn.commit()

    def prune_to_limit(self, limit: int) -> None:
        """仅保留最近的 limit 条（按时间与 id）。"""
        limit = max(0, int(limit))
        if limit <= 0:
            self.delete_all()
            return
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id FROM history
                ORDER BY datetime(created_at) DESC, id DESC
                """,
            ).fetchall()
            if len(rows) <= limit:
                return
            drop = [r[0] for r in rows[limit:]]
            ph = ",".join("?" * len(drop))
            conn.execute(f"DELETE FROM history WHERE id IN ({ph})", drop)
            conn.commit()

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM history").fetchone()[0])
