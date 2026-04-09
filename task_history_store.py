# -*- coding: utf-8 -*-
"""历史任务本地持久化（SQLite，无需单独安装服务）。"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Sequence, Tuple

# id, created_at(结束), started_at, task_type, source, model, objects, wall_s, note,
# det_status, avg_proc_fps, frames_done, frames_total, sum_infer_s
Row = Tuple[int, str, str, str, str, str, int, float, str, str, float, int, int, float]

_EXTRA_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("started_at", "TEXT DEFAULT ''"),
    ("det_status", "TEXT DEFAULT ''"),
    ("avg_proc_fps", "REAL DEFAULT 0"),
    ("frames_done", "INTEGER DEFAULT 0"),
    ("frames_total", "INTEGER DEFAULT 0"),
    ("sum_infer_s", "REAL DEFAULT 0"),
)


class TaskHistoryStore:
    """单表 history：一行一条任务摘要。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _migrate_columns(self, conn: sqlite3.Connection) -> None:
        cur = conn.execute("PRAGMA table_info(history)")
        existing = {row[1] for row in cur.fetchall()}
        for col_name, col_def in _EXTRA_COLUMNS:
            if col_name not in existing:
                conn.execute(
                    f"ALTER TABLE history ADD COLUMN {col_name} {col_def}")

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
            self._migrate_columns(conn)
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
        *,
        started_at: str = "",
        det_status: str = "",
        avg_proc_fps: float = 0.0,
        frames_done: int = 0,
        frames_total: int = 0,
        sum_infer_s: float = 0.0,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO history
                (created_at, task_type, source, model, objects, inference_s, note,
                 started_at, det_status, avg_proc_fps, frames_done, frames_total, sum_infer_s)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    time_str,
                    task_type or "",
                    source or "",
                    model or "",
                    int(objects),
                    float(inference_s),
                    note or "",
                    started_at or "",
                    det_status or "",
                    float(avg_proc_fps),
                    int(frames_done),
                    int(frames_total),
                    float(sum_infer_s),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_recent(self, limit: int) -> List[Row]:
        lim = max(1, min(int(limit), 50_000))
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT id, created_at,
                       COALESCE(started_at, '') AS started_at,
                       COALESCE(task_type, '') AS task_type,
                       COALESCE(source, '') AS source,
                       COALESCE(model, '') AS model,
                       COALESCE(objects, 0) AS objects,
                       COALESCE(inference_s, 0) AS inference_s,
                       COALESCE(note, '') AS note,
                       COALESCE(det_status, '') AS det_status,
                       COALESCE(avg_proc_fps, 0) AS avg_proc_fps,
                       COALESCE(frames_done, 0) AS frames_done,
                       COALESCE(frames_total, 0) AS frames_total,
                       COALESCE(sum_infer_s, 0) AS sum_infer_s
                FROM history
                ORDER BY datetime(created_at) DESC, id DESC
                LIMIT ?
                """,
                (lim,),
            )
            return [tuple(r) for r in cur.fetchall()]

    def list_page(self, page: int, page_size: int) -> List[Row]:
        """分页查询：按结束时间倒序返回指定页。"""
        ps = max(1, min(int(page_size), 50_000))
        pg = max(1, int(page))
        off = (pg - 1) * ps
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT id, created_at,
                       COALESCE(started_at, '') AS started_at,
                       COALESCE(task_type, '') AS task_type,
                       COALESCE(source, '') AS source,
                       COALESCE(model, '') AS model,
                       COALESCE(objects, 0) AS objects,
                       COALESCE(inference_s, 0) AS inference_s,
                       COALESCE(note, '') AS note,
                       COALESCE(det_status, '') AS det_status,
                       COALESCE(avg_proc_fps, 0) AS avg_proc_fps,
                       COALESCE(frames_done, 0) AS frames_done,
                       COALESCE(frames_total, 0) AS frames_total,
                       COALESCE(sum_infer_s, 0) AS sum_infer_s
                FROM history
                ORDER BY datetime(created_at) DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (ps, off),
            )
            return [tuple(r) for r in cur.fetchall()]

    @staticmethod
    def _task_type_filter_sql(mode_key: str) -> Tuple[str, Tuple[str, ...]]:
        mk = (mode_key or "all").strip().lower()
        if mk == "image":
            return "WHERE COALESCE(task_type, '') LIKE ?", ("%图片%",)
        if mk == "video":
            return "WHERE COALESCE(task_type, '') LIKE ?", ("%视频%",)
        if mk == "batch":
            return "WHERE COALESCE(task_type, '') LIKE ?", ("%批量%",)
        if mk in ("camera", "monitor"):
            return "WHERE COALESCE(task_type, '') LIKE ?", ("%监控%",)
        return "", tuple()

    def count_filtered(self, mode_key: str = "all") -> int:
        where_sql, params = self._task_type_filter_sql(mode_key)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                f"SELECT COUNT(*) FROM history {where_sql}",
                params,
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def list_page_filtered(self, page: int, page_size: int, mode_key: str = "all") -> List[Row]:
        ps = max(1, min(int(page_size), 50_000))
        pg = max(1, int(page))
        off = (pg - 1) * ps
        where_sql, params = self._task_type_filter_sql(mode_key)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                f"""
                SELECT id, created_at,
                       COALESCE(started_at, '') AS started_at,
                       COALESCE(task_type, '') AS task_type,
                       COALESCE(source, '') AS source,
                       COALESCE(model, '') AS model,
                       COALESCE(objects, 0) AS objects,
                       COALESCE(inference_s, 0) AS inference_s,
                       COALESCE(note, '') AS note,
                       COALESCE(det_status, '') AS det_status,
                       COALESCE(avg_proc_fps, 0) AS avg_proc_fps,
                       COALESCE(frames_done, 0) AS frames_done,
                       COALESCE(frames_total, 0) AS frames_total,
                       COALESCE(sum_infer_s, 0) AS sum_infer_s
                FROM history
                {where_sql}
                ORDER BY datetime(created_at) DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (*params, ps, off),
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
