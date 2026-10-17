"""SQLite storage for settings, scan runs, queries, listings, observations and emails."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger TEXT NOT NULL,
    provider TEXT NOT NULL,
    send_email INTEGER NOT NULL DEFAULT 1,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    query_count INTEGER NOT NULL DEFAULT 0,
    query_errors INTEGER NOT NULL DEFAULT 0,
    result_count INTEGER NOT NULL DEFAULT 0,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    new_match_count INTEGER NOT NULL DEFAULT 0,
    email_status TEXT,
    error TEXT,
    log TEXT
);
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    tracker TEXT NOT NULL,
    query TEXT NOT NULL,
    provider TEXT NOT NULL,
    request TEXT,
    started_at TEXT NOT NULL,
    duration_ms INTEGER,
    result_count INTEGER,
    error TEXT
);
CREATE TABLE IF NOT EXISTS listings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    dedupe_key TEXT NOT NULL,
    tracker TEXT NOT NULL,
    title TEXT NOT NULL,
    merchant TEXT,
    merchant_domain TEXT,
    link TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    last_run_id INTEGER,
    times_seen INTEGER NOT NULL DEFAULT 0,
    last_price REAL,
    last_unit_price REAL,
    best_unit_price REAL,
    last_decision TEXT,
    last_reasons TEXT,
    notified_at TEXT,
    notified_unit_price REAL
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    query_id INTEGER REFERENCES queries(id) ON DELETE SET NULL,
    listing_id INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    tracker TEXT NOT NULL,
    title TEXT NOT NULL,
    merchant TEXT,
    merchant_canonical TEXT,
    merchant_domain TEXT,
    link TEXT,
    price REAL,
    shipping REAL,
    total_price REAL,
    units INTEGER,
    unit_price REAL,
    decision TEXT NOT NULL,
    reasons TEXT NOT NULL,
    notes TEXT NOT NULL,
    attributes TEXT NOT NULL,
    tier TEXT,
    notify_reason TEXT,
    notified INTEGER NOT NULL DEFAULT 0,
    raw TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    recipient TEXT,
    subject TEXT,
    status TEXT NOT NULL,
    error TEXT,
    listing_count INTEGER NOT NULL DEFAULT 0,
    body_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_run ON observations(run_id);
CREATE INDEX IF NOT EXISTS idx_obs_listing ON observations(listing_id);
CREATE INDEX IF NOT EXISTS idx_obs_created ON observations(created_at);
CREATE INDEX IF NOT EXISTS idx_listings_tracker ON listings(tracker, last_decision);
CREATE INDEX IF NOT EXISTS idx_listings_dedupe ON listings(dedupe_key);
CREATE INDEX IF NOT EXISTS idx_queries_run ON queries(run_id);
"""

JSON_COLUMNS = ("reasons", "notes", "attributes", "last_reasons")


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    data = dict(row)
    for column in JSON_COLUMNS:
        if data.get(column):
            data[column] = json.loads(data[column])
    return data


def _rows(rows) -> list[dict]:
    return [_row(r) for r in rows]


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- settings ---

    def get_settings_blob(self) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT data FROM settings WHERE id = 1").fetchone()
        return row["data"] if row else None

    def save_settings_blob(self, data: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO settings (id, data, updated_at) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
                (data, utcnow()),
            )

    # --- runs ---

    def create_run(self, trigger: str, provider: str, send_email: bool) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (trigger, provider, send_email, started_at, status) VALUES (?, ?, ?, ?, 'running')",
                (trigger, provider, int(send_email), utcnow()),
            )
            return cur.lastrowid

    def update_run(self, run_id: int, **fields) -> None:
        if not fields:
            return
        columns = ", ".join(f"{name} = ?" for name in fields)
        with self.connect() as conn:
            conn.execute(f"UPDATE runs SET {columns} WHERE id = ?", (*fields.values(), run_id))

    def get_run(self, run_id: int) -> dict | None:
        with self.connect() as conn:
            return _row(conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone())

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return _rows(rows)

    def count_runs(self) -> int:
        with self.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    def latest_completed_run(self) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE status != 'running' ORDER BY id DESC LIMIT 1").fetchone()
        return _row(row)

    def latest_successful_run(self) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE status IN ('ok', 'partial') ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return _row(row)

    def running_run(self) -> dict | None:
        with self.connect() as conn:
            return _row(conn.execute("SELECT * FROM runs WHERE status = 'running' ORDER BY id DESC LIMIT 1").fetchone())

    def mark_interrupted_runs(self) -> int:
        """Runs left 'running' by a previous process (crash or restart) are marked failed."""
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE runs SET status = 'failed', finished_at = ?, "
                "error = 'Interrupted: the app stopped while this scan was running' WHERE status = 'running'",
                (utcnow(),),
            )
            return cur.rowcount

    # --- queries ---

    def create_query(self, run_id: int, tracker: str, query: str, provider: str, request: str) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO queries (run_id, tracker, query, provider, request, started_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, tracker, query, provider, request, utcnow()),
            )
            return cur.lastrowid

    def finish_query(self, query_id: int, duration_ms: int, result_count: int | None, error: str | None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE queries SET duration_ms = ?, result_count = ?, error = ? WHERE id = ?",
                (duration_ms, result_count, error, query_id),
            )

    def queries_for_run(self, run_id: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT q.*, "
                "(SELECT COUNT(*) FROM observations o "
                "  WHERE o.query_id = q.id AND o.decision = 'accepted') AS accepted, "
                "(SELECT COUNT(*) FROM observations o WHERE o.query_id = q.id) AS kept "
                "FROM queries q WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return _rows(rows)

    # --- listings & observations ---

    def get_listing_by_key(self, key: str) -> dict | None:
        with self.connect() as conn:
            return _row(conn.execute("SELECT * FROM listings WHERE key = ?", (key,)).fetchone())

    def get_listing(self, listing_id: int) -> dict | None:
        with self.connect() as conn:
            return _row(conn.execute("SELECT * FROM listings WHERE id = ?", (listing_id,)).fetchone())

    def upsert_listing(self, key: str, dedupe_key: str, tracker: str, run_id: int, obs: dict) -> int:
        now = utcnow()
        reasons = json.dumps(obs["reasons"])
        accepted_price = obs["unit_price"] if obs["decision"] == "accepted" else None
        with self.connect() as conn:
            existing = conn.execute("SELECT id FROM listings WHERE key = ?", (key,)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE listings SET title = ?, merchant = ?, merchant_domain = ?, link = ?, last_seen_at = ?, "
                    "last_run_id = ?, times_seen = times_seen + 1, last_price = ?, last_unit_price = ?, "
                    "best_unit_price = CASE WHEN ? IS NULL THEN best_unit_price "
                    "  WHEN best_unit_price IS NULL OR ? < best_unit_price THEN ? ELSE best_unit_price END, "
                    "last_decision = ?, last_reasons = ? WHERE id = ?",
                    (
                        obs["title"], obs["merchant"], obs["merchant_domain"], obs["link"], now, run_id,
                        obs["price"], obs["unit_price"], accepted_price, accepted_price, accepted_price,
                        obs["decision"], reasons, existing["id"],
                    ),
                )  # fmt: skip
                return existing["id"]
            cur = conn.execute(
                "INSERT INTO listings (key, dedupe_key, tracker, title, merchant, merchant_domain, link, "
                "first_seen_at, last_seen_at, last_run_id, times_seen, last_price, last_unit_price, best_unit_price, "
                "last_decision, last_reasons) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
                (
                    key, dedupe_key, tracker, obs["title"], obs["merchant"], obs["merchant_domain"], obs["link"],
                    now, now, run_id, obs["price"], obs["unit_price"], accepted_price, obs["decision"], reasons,
                ),
            )  # fmt: skip
            return cur.lastrowid

    def insert_observation(self, run_id: int, query_id: int, listing_id: int, obs: dict) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO observations (run_id, query_id, listing_id, tracker, title, merchant, merchant_canonical, "
                "merchant_domain, link, price, shipping, total_price, units, unit_price, decision, reasons, notes, "
                "attributes, tier, notify_reason, raw, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, query_id, listing_id, obs["tracker"], obs["title"], obs["merchant"],
                    obs["merchant_canonical"], obs["merchant_domain"], obs["link"], obs["price"], obs["shipping"],
                    obs["total_price"], obs["units"], obs["unit_price"], obs["decision"], json.dumps(obs["reasons"]),
                    json.dumps(obs["notes"]), json.dumps(obs["attributes"]), obs["tier"], obs.get("notify_reason"),
                    json.dumps(obs["raw"]), utcnow(),
                ),
            )  # fmt: skip
            return cur.lastrowid

    def notified_duplicate(self, dedupe_key: str, listing_id: int, unit_price: float) -> dict | None:
        """Another listing with the same seller and title that was already emailed at this price or lower."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM listings WHERE dedupe_key = ? AND id != ? AND notified_unit_price IS NOT NULL "
                "AND notified_unit_price <= ? ORDER BY notified_at DESC LIMIT 1",
                (dedupe_key, listing_id, unit_price * 1.01),
            ).fetchone()
        return _row(row)

    def mark_notified(self, items: list[tuple[int, int, float]]) -> None:
        """items: (listing_id, observation_id, unit_price)."""
        now = utcnow()
        with self.connect() as conn:
            for listing_id, observation_id, unit_price in items:
                conn.execute(
                    "UPDATE listings SET notified_at = ?, notified_unit_price = ? WHERE id = ?",
                    (now, unit_price, listing_id),
                )
                conn.execute("UPDATE observations SET notified = 1 WHERE id = ?", (observation_id,))

    def observations_for_run(self, run_id: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT o.*, q.query AS query_text FROM observations o LEFT JOIN queries q ON q.id = o.query_id "
                "WHERE o.run_id = ? ORDER BY o.decision, o.tracker, o.unit_price IS NULL, o.unit_price",
                (run_id,),
            ).fetchall()
        return _rows(rows)

    def observations_for_listing(self, listing_id: int, limit: int = 200) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT o.*, q.query AS query_text FROM observations o LEFT JOIN queries q ON q.id = o.query_id "
                "WHERE o.listing_id = ? ORDER BY o.id DESC LIMIT ?",
                (listing_id, limit),
            ).fetchall()
        return _rows(rows)

    def get_observation(self, observation_id: int) -> dict | None:
        with self.connect() as conn:
            return _row(conn.execute("SELECT * FROM observations WHERE id = ?", (observation_id,)).fetchone())

    def list_listings(
        self,
        tracker: str = "",
        decision: str = "",
        search: str = "",
        merchant: str = "",
        sort: str = "recent",
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        where, params = [], []
        if tracker:
            where.append("tracker = ?")
            params.append(tracker)
        if decision == "notified":
            where.append("notified_at IS NOT NULL")
        elif decision:
            where.append("last_decision = ?")
            params.append(decision)
        if search:
            where.append("title LIKE ?")
            params.append(f"%{search}%")
        if merchant:
            where.append("(merchant LIKE ? OR merchant_domain LIKE ?)")
            params.extend([f"%{merchant}%", f"%{merchant}%"])
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        order = {
            "recent": "last_seen_at DESC, id DESC",
            "price": "last_unit_price IS NULL, last_unit_price ASC",
            "first_seen": "first_seen_at DESC, id DESC",
            "seen": "times_seen DESC",
        }.get(sort, "last_seen_at DESC, id DESC")
        with self.connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM listings {clause}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM listings {clause} ORDER BY {order} LIMIT ? OFFSET ?", (*params, limit, offset)
            ).fetchall()
        return _rows(rows), total

    def merchant_stats(self, since: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT merchant, merchant_domain AS domain, MAX(merchant_canonical) AS canonical, "
                "COUNT(*) AS seen, SUM(decision = 'accepted') AS accepted, SUM(decision = 'rejected') AS rejected, "
                "SUM(reasons LIKE '%\"merchant_%') AS merchant_rejections, MAX(created_at) AS last_seen, "
                "MIN(unit_price) AS min_unit_price, COUNT(DISTINCT tracker) AS trackers "
                "FROM observations WHERE created_at >= ? GROUP BY merchant, merchant_domain ORDER BY seen DESC",
                (since,),
            ).fetchall()
        return [dict(r) for r in rows]

    def dashboard_counts(self) -> dict:
        with self.connect() as conn:
            listings = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
            notified = conn.execute("SELECT COUNT(*) FROM listings WHERE notified_at IS NOT NULL").fetchone()[0]
            emails = conn.execute("SELECT COUNT(*) FROM notifications WHERE status = 'sent'").fetchone()[0]
            observations = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        return {"listings": listings, "notified": notified, "emails": emails, "observations": observations}

    # --- notifications ---

    def insert_notification(self, **fields) -> int:
        fields.setdefault("created_at", utcnow())
        columns = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        with self.connect() as conn:
            cur = conn.execute(f"INSERT INTO notifications ({columns}) VALUES ({placeholders})", tuple(fields.values()))
            return cur.lastrowid

    def list_notifications(self, limit: int = 100) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM notifications ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return _rows(rows)

    def last_notification(self) -> dict | None:
        with self.connect() as conn:
            return _row(conn.execute("SELECT * FROM notifications ORDER BY id DESC LIMIT 1").fetchone())

    # --- maintenance ---

    def prune(self, before: str) -> int:
        """Drop raw observation data older than the retention window (listings and runs are kept)."""
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM observations WHERE created_at < ?", (before,))
            return cur.rowcount
