"""
AI Legal Chatbot - Database Manager

Handles connection and schema setup for SQLite, password hashing with SHA-256 pre-hash
followed by bcrypt, user credentials verification, and account creation.
"""

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import bcrypt


# Minimum length enforced at registration (and in create_user).
MIN_PASSWORD_LENGTH = 8


def _password_digest(password: str) -> bytes:
    """SHA-256 of UTF-8 password; bcrypt hashes this 32-byte value (no 72-byte user limit)."""
    return hashlib.sha256(password.encode("utf-8")).digest()


def _legacy_bcrypt_secret(password: str) -> bytes:
    """Older accounts used bcrypt(utf8(password)) with bcrypt's 72-byte input cap."""
    return password.encode("utf-8")[:72]


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(_password_digest(password), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    raw = password_hash.encode("utf-8")
    try:
        if bcrypt.checkpw(_password_digest(password), raw):
            return True
    except ValueError:
        pass
    try:
        return bcrypt.checkpw(_legacy_bcrypt_secret(password), raw)
    except ValueError:
        return False


@dataclass(frozen=True)
class User:
    id: int
    username: str
    role: str
    created_at: str


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db(db_path: str) -> None:
    """
    Initialize the SQLite database that stores hashed user/admin credentials.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL CHECK(role IN ('user','admin')),
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()


def create_user(db_path: str, *, username: str, password: str, role: str) -> None:
    """
    Create a new user with a hashed password.

    Role must be either 'user' or 'admin'.
    Password must be at least MIN_PASSWORD_LENGTH characters.
    """
    if role not in {"user", "admin"}:
        raise ValueError("role must be either 'user' or 'admin'")

    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        )

    password_hash = _hash_password(password)
    created_at = _utc_iso()

    with sqlite3.connect(db_path) as conn:
        try:
            conn.execute(
                """
                INSERT INTO users (username, role, password_hash, created_at)
                VALUES (?, ?, ?, ?);
                """,
                (username, role, password_hash, created_at),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            # Most commonly: username already exists.
            raise ValueError("Username already exists.") from e


def authenticate_user(
    db_path: str, *, username: str, password: str
) -> Optional[User]:
    """
    Verify username/password. Returns a User on success, otherwise None.
    """
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, username, role, password_hash, created_at
            FROM users
            WHERE username = ?;
            """,
            (username,),
        ).fetchone()

    if not row:
        return None

    user_id, u_username, role, password_hash, created_at = row
    if not _verify_password(password, password_hash):
        return None

    return User(
        id=int(user_id),
        username=str(u_username),
        role=str(role),
        created_at=str(created_at),
    )
