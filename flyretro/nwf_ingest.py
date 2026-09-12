"""Ingest real nano-workforce retros from the app's SQLite database.

Reads the post-implementation material the retro pipeline persists
(app/retro.ts): the `learning` blackboard entries an epic accrued, its
constraint/scope changes, the retro summary, and decomposition counts. Each
epic becomes one RetroRecord the featurizer can turn into PN activity.

Read-only: opens the DB in immutable mode so a running app is never disturbed.
"""

from __future__ import annotations

import os
import sqlite3

from .featurizer import RetroRecord

DEFAULT_DB = os.environ.get(
    "FLY_NWF_DB", os.path.expanduser("~/workspace/nano-workforce/app.db")
)


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    return con


def load_retros(db_path: str = DEFAULT_DB, filed_only: bool = True) -> list[RetroRecord]:
    con = _connect(db_path)
    try:
        where = "WHERE r.status = 'filed'" if filed_only else ""
        rows = con.execute(
            f"""
            SELECT r.plan_key, r.status, r.summary, r.report, r.learnings,
                   p.repo, p.issue_number, p.title, p.task_count
            FROM plan_retros r JOIN plans p ON p.plan_key = r.plan_key
            {where}
            ORDER BY r.created_at
            """
        ).fetchall()

        records: list[RetroRecord] = []
        for r in rows:
            pk = r["plan_key"]
            learnings = [
                x["body"]
                for x in con.execute(
                    "SELECT body FROM plan_blackboard WHERE plan_key=? AND kind='learning' ORDER BY id",
                    (pk,),
                )
            ]
            deviations = [
                x["body"]
                for x in con.execute(
                    "SELECT body FROM plan_blackboard WHERE plan_key=? AND kind IN ('constraint-change','scope-change') ORDER BY id",
                    (pk,),
                )
            ]
            counts = con.execute(
                "SELECT COUNT(*) AS total, SUM(status='blocked') AS blocked FROM plan_tasks WHERE plan_key=?",
                (pk,),
            ).fetchone()

            head = [t for t in [r["title"], r["summary"]] if t]
            records.append(
                RetroRecord(
                    repo=r["repo"] or "?",
                    epic=f"#{r['issue_number']}",
                    learnings=head + learnings,
                    deviations=deviations,
                    summary=r["summary"] or "",
                    tasks_total=int(counts["total"] or 0),
                    tasks_blocked=int(counts["blocked"] or 0),
                    tasks_skipped=0,
                    rounds=0,
                )
            )
        return records
    finally:
        con.close()
