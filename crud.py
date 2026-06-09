import sqlite3
import os
import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            filename        TEXT NOT NULL UNIQUE,
            upload_date     TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            page_count      INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn


def create_document(filename: str, file_size_bytes: int, page_count: int) -> int:
    """INSERT a new document. Returns the auto-assigned id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO documents (filename, upload_date, file_size_bytes, page_count) VALUES (?,?,?,?)",
            (filename, datetime.datetime.now().isoformat(timespec="seconds"), file_size_bytes, page_count),
        )
        return cur.lastrowid  # SQLite assigns the id; this is how you retrieve it


def read_documents() -> list[dict]:
    """SELECT all documents, newest first."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY upload_date DESC").fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: int) -> dict | None:
    """SELECT a single document by primary key. Returns None if not found."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None


def update_document_name(doc_id: int, new_filename: str) -> bool:
    """UPDATE the filename. Returns False if no row matched the id."""
    with _connect() as conn:
        cur = conn.execute("UPDATE documents SET filename = ? WHERE id = ?", (new_filename, doc_id))
        return cur.rowcount > 0  # rowcount == 0 means no row had that id


def delete_document(doc_id: int) -> dict | None:
    """DELETE a document. Returns the deleted row's data (needed to clean up disk + LanceDB)."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return dict(row)
