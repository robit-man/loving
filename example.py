#!/usr/bin/env python3
"""
Standalone Ollama Qwen3 chat app with:

- Auto .venv bootstrap and dependency install
- SQLite-backed user accounts and chat history
- system.md used as a system message for Ollama's chat API
- Sepia + pink light-mode web UI with markdown rendering
- True streaming responses in the chat interface (token chunks)

Run: python3 this_script.py
"""

# === Standard Library Imports & Virtualenv Bootstrap =======================

import os
import sys
import subprocess
from pathlib import Path

# --- Virtualenv configuration constants ------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parent
VENV_DIR: Path = BASE_DIR / ".venv"
VENV_BIN_DIR: Path = VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
VENV_PYTHON: Path = VENV_BIN_DIR / ("python.exe" if os.name == "nt" else "python")

REQUIRED_PACKAGES = ["flask", "requests"]  # Add more here if needed


def _in_our_venv() -> bool:
    """Return True if we're already running under the .venv we manage."""
    try:
        return VENV_DIR.resolve() == Path(sys.prefix).resolve()
    except Exception:
        return False


def _bootstrap_venv_if_needed() -> None:
    """
    Create and/or use .venv for this script.

    - Only runs when this file is used as a script (__main__).
    - If .venv missing, create and install REQUIRED_PACKAGES.
    - Then re-exec this script using the venv's python.
    """
    if __name__ != "__main__":
        # Don't auto-bootstrap when imported as a module
        return

    if _in_our_venv():
        return

    if not VENV_PYTHON.exists():
        print("[bootstrap] Creating virtualenv at .venv ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])

        print("[bootstrap] Upgrading pip in .venv ...")
        subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])

        if REQUIRED_PACKAGES:
            print("[bootstrap] Installing required packages into .venv ...")
            subprocess.check_call(
                [str(VENV_PYTHON), "-m", "pip", "install", *REQUIRED_PACKAGES]
            )

    # Re-exec this script using the venv python
    env = os.environ.copy()
    os.execve(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), __file__, *sys.argv[1:]],
        env,
    )


_bootstrap_venv_if_needed()

# === Third-Party & Standard Imports (inside venv) ==========================

import sqlite3
import threading
import datetime
import hashlib
import hmac
import secrets
import json
from typing import List, Dict, Optional, Tuple

import requests
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template_string,
    session,
    Response,
    stream_with_context,
    jsonify,
)

# === Application Configuration Constants ===================================

# --- Server settings -------------------------------------------------------

APP_TITLE: str = "Loving Qwen Eval"
SERVER_HOST: str = "127.0.0.1"
SERVER_PORT: int = 5111
SERVER_DEBUG: bool = False

# --- Secret key & sessions -------------------------------------------------

DEFAULT_DEV_SECRET: str = "change-this-secret-key"
FLASK_SECRET_KEY: str = os.environ.get("OLLAMA_CHAT_SECRET", DEFAULT_DEV_SECRET)

SESSION_USER_ID_KEY: str = "user_id"
SESSION_USERNAME_KEY: str = "username"

# --- SQLite configuration --------------------------------------------------

DATABASE_PATH: Path = BASE_DIR / "chat.db"
SQLITE_FOREIGN_KEYS_ON: str = "PRAGMA foreign_keys = ON;"

# --- Password hashing parameters ------------------------------------------

PASSWORD_HASH_ALGORITHM: str = "sha256"
PASSWORD_PBKDF2_ITERATIONS: int = 150_000
PASSWORD_SALT_BYTES: int = 16
PASSWORD_HASH_SEPARATOR: str = "$"

# --- Chat configuration ----------------------------------------------------

SYSTEM_PROMPT_FILE: Path = BASE_DIR / "system.md"

OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT: str = "/api/chat"
OLLAMA_MODEL_NAME: str = "qwen3:235b"
OLLAMA_REQUEST_TIMEOUT_SECONDS: int = 300
MAX_MESSAGES_PER_CHAT: int = 30  # number of most recent messages to send to model

# --- UI configuration ------------------------------------------------------

MAX_MESSAGE_LENGTH: int = 4000  # characters; oversized input will be truncated

# === Security Helpers ======================================================


def hash_password(plaintext: str) -> str:
    """
    Hash a plaintext password using PBKDF2-HMAC.

    Returns a string: "<algo>$<iterations>$<salt_hex>$<hash_hex>"
    """
    salt = secrets.token_bytes(PASSWORD_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        PASSWORD_HASH_ALGORITHM,
        plaintext.encode("utf-8"),
        salt,
        PASSWORD_PBKDF2_ITERATIONS,
    )
    return PASSWORD_HASH_SEPARATOR.join(
        [
            PASSWORD_HASH_ALGORITHM,
            str(PASSWORD_PBKDF2_ITERATIONS),
            salt.hex(),
            derived.hex(),
        ]
    )


def verify_password(plaintext: str, stored: str) -> bool:
    """
    Verify a plaintext password against a stored hash.

    Stored format: "<algo>$<iterations>$<salt_hex>$<hash_hex>"
    """
    try:
        algo, iterations_str, salt_hex, hash_hex = stored.split(PASSWORD_HASH_SEPARATOR)
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        # Stored hash is malformed
        return False

    candidate = hashlib.pbkdf2_hmac(
        algo,
        plaintext.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate, expected)


def utc_now_iso() -> str:
    """Timezone-aware UTC ISO timestamp."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# === Database Layer ========================================================


class Database:
    """Simple SQLite wrapper for users and messages."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    # --- schema -------------------------------------------------------------

    def _initialize_schema(self) -> None:
        with self._lock:
            cur = self._connection.cursor()
            cur.execute(SQLITE_FOREIGN_KEYS_ON)

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,        -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )

            self._connection.commit()

    # --- users --------------------------------------------------------------

    def create_user(self, username: str, password: str) -> bool:
        """Create a user; return True on success, False if username exists."""
        now = utc_now_iso()
        password_hash = hash_password(password)
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?);",
                    (username, password_hash, now),
                )
                self._connection.commit()
                return True
            except sqlite3.IntegrityError:
                # likely UNIQUE constraint on username
                return False

    def authenticate_user(self, username: str, password: str) -> Optional[Tuple[int, str]]:
        """Return (user_id, username) if correct credentials; otherwise None."""
        with self._lock:
            cur = self._connection.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?;",
                (username,),
            )
            row = cur.fetchone()

        if row is None:
            return None

        if not verify_password(password, row["password_hash"]):
            return None

        return row["id"], row["username"]

    def get_user_by_id(self, user_id: int) -> Optional[Tuple[int, str]]:
        with self._lock:
            cur = self._connection.execute(
                "SELECT id, username FROM users WHERE id = ?;",
                (user_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return row["id"], row["username"]

    # --- messages -----------------------------------------------------------

    def append_message(self, user_id: int, role: str, content: str) -> None:
        now = utc_now_iso()
        with self._lock:
            self._connection.execute(
                "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?);",
                (user_id, role, content, now),
            )
            self._connection.commit()

    def get_recent_messages(
        self,
        user_id: int,
        limit: int,
    ) -> List[Dict[str, str]]:
        """
        Return recent messages for a user as a list of dicts:
        [{"role": "user"/"assistant", "content": "..."}], oldest -> newest.
        """
        with self._lock:
            cur = self._connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?;
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()

        ordered = list(reversed(rows))  # oldest first
        return [{"role": row["role"], "content": row["content"]} for row in ordered]


# === System Prompt Loader ==================================================


class SystemPromptLoader:
    """Load and cache the system.md file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cached_mtime: Optional[float] = None
        self._cached_text: Optional[str] = None

    def get_prompt(self) -> str:
        if not self._path.exists():
            return "You are a helpful assistant."

        mtime = self._path.stat().st_mtime
        if self._cached_text is None or self._cached_mtime != mtime:
            self._cached_text = self._path.read_text(encoding="utf-8")
            self._cached_mtime = mtime
        return self._cached_text


# === Ollama Client (Streaming) ============================================


class OllamaClient:
    """Thin wrapper around the Ollama /api/chat endpoint with streaming."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        endpoint: str = OLLAMA_CHAT_ENDPOINT,
        timeout_seconds: int = OLLAMA_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds

    def stream(self, messages: List[Dict[str, str]]):
        """
        Yield content chunks from Ollama's streaming chat endpoint.
        Each yielded value is a string delta.
        """
        url = f"{self._base_url}{self._endpoint}"
        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": True,
        }

        response = requests.post(
            url, json=payload, timeout=self._timeout_seconds, stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = data.get("message") or {}
            delta = message.get("content") or ""
            if delta:
                yield delta
            if data.get("done"):
                break

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Convenience: return full response text by consuming the stream."""
        parts: List[str] = []
        try:
            for delta in self.stream(messages):
                parts.append(delta)
        except Exception as exc:  # noqa: BLE001
            print(f"[OllamaClient] Error calling Ollama: {exc}", file=sys.stderr)
            return "Sorry, I could not reach the language model. Please try again."
        text = "".join(parts).strip()
        return text or "Sorry, I could not reach the language model. Please try again."


# === Flask App Setup =======================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = FLASK_SECRET_KEY

db = Database(DATABASE_PATH)
system_prompts = SystemPromptLoader(SYSTEM_PROMPT_FILE)
ollama_client = OllamaClient(
    base_url=OLLAMA_BASE_URL,
    model_name=OLLAMA_MODEL_NAME,
)

# === HTML Templates (Sepia + Pink, Markdown via Marked.js) ================

BASE_CSS = """
:root {
  color-scheme: light;
  --bg-page: #f8f0e6;       /* warm sepia */
  --bg-card: #fff9f4;       /* off-white coral */
  --bg-input: #fdf3ea;
  --border-subtle: #f0d7c2;
  --text-main: #3d2b26;
  --text-muted: #8b6f63;
  --accent: #ec4899;        /* pink */
  --accent-soft: #fce7f3;
  --danger: #b91c1c;
  --radius-card: 14px;
  --shadow-card: 0 10px 30px rgba(60, 30, 20, 0.16);
}
*,
*::before,
*::after {
  box-sizing: border-box;
}
body {
  margin: 0;
  padding: 0;
  min-height: 100vh;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: radial-gradient(circle at top left, #fff1f7 0, var(--bg-page) 38%, #f4e6d7 100%);
  color: var(--text-main);
}
body,
.card,
.chat-bubble {
  color: var(--text-main);
  opacity: 0.96;
}
.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.app-header {
  padding: 12px 18px;
  background: linear-gradient(90deg, #fff7fb 0, #fff9f4 40%, #ffeef8 100%);
  border-bottom: 1px solid rgba(239, 200, 184, 0.8);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.app-header-title {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #be185d;
}
.app-header-sub {
  font-size: 12px;
  color: var(--text-muted);
}
.app-header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}
.app-header-user {
  font-size: 13px;
  color: rgba(139, 111, 99, 0.9);
}
.app-header-link {
  font-size: 13px;
  color: var(--accent);
  text-decoration: none;
  border-radius: 999px;
  padding: 4px 10px;
  background: rgba(252, 231, 243, 0.85);
}
.app-header-link:hover {
  text-decoration: none;
  background: var(--accent);
  color: #fff5f7;
}
.app-main {
  flex: 1;
  padding: 24px 16px 16px;
  display: flex;
  justify-content: center;
}
.card {
  width: 100%;
  max-width: 780px;
  background: var(--bg-card);
  border-radius: var(--radius-card);
  box-shadow: var(--shadow-card);
  padding: 22px 22px 18px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  border: 1px solid rgba(248, 219, 204, 0.7);
}
.card-title {
  font-size: 20px;
  font-weight: 600;
}
.card-subtitle {
  font-size: 13px;
  color: var(--text-muted);
}
.form-grid {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.form-field label {
  display: block;
  font-size: 13px;
  margin-bottom: 4px;
}
.form-input {
  width: 100%;
  padding: 8px 10px;
  border-radius: 9px;
  border: 1px solid var(--border-subtle);
  background: var(--bg-input);
  font-size: 14px;
}
.form-input:focus {
  outline: none;
  border-color: var(--accent);
  background: #fffdf9;
}
.form-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 4px;
  gap: 8px;
  flex-wrap: wrap;
}
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 84px;
  padding: 8px 16px;
  border-radius: 999px;
  border: 1px solid transparent;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  background: linear-gradient(135deg, #ec4899, #fb7185);
  color: #fff7fc;
  box-shadow: 0 6px 18px rgba(236, 72, 153, 0.35);
}
.btn-secondary {
  background: #fff7fb;
  color: var(--accent);
  border-color: var(--accent-soft);
  box-shadow: none;
}
.btn:hover {
  filter: brightness(1.04);
}
.alert {
  border-radius: 10px;
  padding: 8px 10px;
  font-size: 13px;
}
.alert-error {
  background: #fef2f2;
  color: #991b1b;
  border: 1px solid #fee2e2;
}
.alert-info {
  background: #fef3f8;
  color: #9d174d;
  border: 1px solid #f9a8d4;
}
.text-link {
  color: var(--accent);
  text-decoration: none;
  font-size: 13px;
}
.text-link:hover {
  text-decoration: underline;
}
.chat-messages {
  max-height: calc(100vh - 21.5rem);
  overflow-y: auto;
  padding: 12px 10px;
  border-radius: 12px;
  border: 1px solid var(--border-subtle);
  background: radial-gradient(circle at top left, #fff7fb 0, #fdf3ea 40%, #fef7f1 100%);
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.chat-message {
  display: flex;
  flex-direction: column;
  max-width: 90%;
}
.chat-message-user {
  align-self: flex-end;
  text-align: right;
}
.chat-message-assistant {
  align-self: flex-start;
}
.chat-bubble {
  border-radius: 18px;
  padding: 9px 12px;
  font-size: 14px;
  line-height: 1.5;
}
.chat-bubble-user {
  background: linear-gradient(135deg, #ec4899, #f97373);
  color: #fff7fb;
  border-bottom-right-radius: 6px;
}
.chat-bubble-assistant {
  background: #fffdf9;
  color: var(--text-main);
  border: 1px solid rgba(244, 213, 193, 0.9);
  border-bottom-left-radius: 6px;
}
.chat-meta {
  font-size: 11px;
  color: rgba(139, 111, 99, 0.86);
  margin-bottom: 2px;
}
.chat-input-wrap {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.chat-textarea {
  width: 100%;
  min-height: 72px;
  max-height: 160px;
  resize: vertical;
  border-radius: 12px;
  border: 1px solid var(--border-subtle);
  padding: 8px 10px;
  font-size: 14px;
  background: var(--bg-input);
}
.chat-textarea:focus {
  outline: none;
  border-color: var(--accent);
  background: #fffdf9;
}
.chat-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.chat-hint {
  font-size: 12px;
  color: var(--text-muted);
}
.markdown-body {
  font-size: 14px;
}
.markdown-body p {
  margin: 0.2em 0 0.45em;
}
.markdown-body ul,
.markdown-body ol {
  padding-left: 1.2em;
  margin: 0.3em 0 0.5em;
}
.markdown-body code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.9em;
  background: rgba(254, 242, 242, 0.9);
  padding: 1px 3px;
  border-radius: 4px;
}
.markdown-body pre {
  background: #111827;
  color: #f9fafb;
  padding: 8px 10px;
  border-radius: 8px;
  overflow-x: auto;
  font-size: 12px;
}
@media (max-width: 640px) {
  .card {
    padding: 16px 14px 14px;
  }
  .app-header {
    padding-inline: 12px;
  }
}
"""

LOGIN_TEMPLATE = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{{{ title }}}}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <style>{BASE_CSS}</style>
</head>
<body>
  <div class="app-shell">
    <header class="app-header">
      <div>
        <div class="app-header-title">{APP_TITLE}</div>
        <div class="app-header-sub">Compassionate Discourse with <code>{OLLAMA_MODEL_NAME}</code></div>
      </div>
      <div class="app-header-right">
        <span class="app-header-user">Not signed in</span>
        <a class="app-header-link" href="{{{{ url_for('register') }}}}">Create account</a>
      </div>
    </header>
    <main class="app-main">
      <section class="card">
        <div>
          <div class="card-title">Sign in</div>
          <div class="card-subtitle">Use your local account to start a loving evaluation with Qwen.</div>
        </div>

        {{% if error %}}
        <div class="alert alert-error">{{{{ error }}}}</div>
        {{% endif %}}

        <form method="post" class="form-grid">
          <div class="form-field">
            <label for="username">Username</label>
            <input class="form-input" id="username" name="username" required autofocus>
          </div>
          <div class="form-field">
            <label for="password">Password</label>
            <input class="form-input" id="password" type="password" name="password" required>
          </div>

          <div class="form-footer">
            <button class="btn" type="submit">Sign in</button>
            <a class="text-link" href="{{{{ url_for('register') }}}}">Need an account? Register</a>
          </div>
        </form>
      </section>
    </main>
  </div>
</body>
</html>
"""

REGISTER_TEMPLATE = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{{{ title }}}}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <style>{BASE_CSS}</style>
</head>
<body>
  <div class="app-shell">
    <header class="app-header">
      <div>
        <div class="app-header-title">{APP_TITLE}</div>
        <div class="app-header-sub">Compassionate Discourse with <code>{OLLAMA_MODEL_NAME}</code></div>
      </div>
      <div class="app-header-right">
        <span class="app-header-user">Not signed in</span>
        <a class="app-header-link" href="{{{{ url_for('login') }}}}">Back to sign in</a>
      </div>
    </header>
    <main class="app-main">
      <section class="card">
        <div>
          <div class="card-title">Create account</div>
          <div class="card-subtitle">A minimal local user record stored in SQLite.</div>
        </div>

        {{% if error %}}
        <div class="alert alert-error">{{{{ error }}}}</div>
        {{% elif info %}}
        <div class="alert alert-info">{{{{ info }}}}</div>
        {{% endif %}}

        <form method="post" class="form-grid">
          <div class="form-field">
            <label for="username">Username</label>
            <input class="form-input" id="username" name="username" required autofocus>
          </div>
          <div class="form-field">
            <label for="password">Password</label>
            <input class="form-input" id="password" type="password" name="password" required>
          </div>

          <div class="form-footer">
            <button class="btn" type="submit">Create</button>
            <a class="text-link" href="{{{{ url_for('login') }}}}">Already have an account? Sign in</a>
          </div>
        </form>
      </section>
    </main>
  </div>
</body>
</html>
"""

CHAT_TEMPLATE = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{{{ title }}}}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <style>{BASE_CSS}</style>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
  <div class="app-shell">
    <header class="app-header">
      <div>
        <div class="app-header-title">{APP_TITLE}</div>
        <div class="app-header-sub">Compassionate Discourse with <code>{OLLAMA_MODEL_NAME}</code></div>
      </div>
      <div class="app-header-right">
        <span class="app-header-user">Signed in as {{{{ username }}}}</span>
        <a class="app-header-link" href="{{{{ url_for('logout') }}}}">Sign out</a>
      </div>
    </header>
    <main class="app-main">
      <section class="card">
        <div>
          <div class="card-title">NOVA-Q</div>
          <div class="card-subtitle">
            System prompt is read from <code>system.md</code> and sent as the <strong>seed</strong> for nova-q's approach to communication.
          </div>
        </div>

        {{% if info %}}
        <div class="alert alert-info">{{{{ info }}}}</div>
        {{% elif error %}}
        <div class="alert alert-error">{{{{ error }}}}</div>
        {{% endif %}}

        <div class="chat-messages" id="chat-messages">
          {{% for msg in messages %}}
          <div class="chat-message chat-message-{{{{ msg.role }}}}">
            <div class="chat-meta">
              {{% if msg.role == 'user' %}}You{{% else %}}Assistant{{% endif %}}
            </div>
            <div class="chat-bubble chat-bubble-{{{{ msg.role }}}}">
              {{{{ msg.content }}}}
            </div>
          </div>
          {{% endfor %}}
          {{% if not messages %}}
          {{% endif %}}
        </div>

        <form id="chat-form" class="chat-input-wrap" onsubmit="return sendMessage(event);">
          <textarea
            class="chat-textarea"
            id="chat-input"
            name="message"
            required
            placeholder="Offer a question or reflection for NOVA-Q..."
          >{{{{ draft or '' }}}}</textarea>
          <div class="chat-actions">
            <button class="btn" type="submit">Send</button>
            <div class="chat-hint">
              Model: <code>{OLLAMA_MODEL_NAME}</code> · Context: last {{ max_messages }} messages
            </div>
          </div>
        </form>
      </section>
    </main>
  </div>

  <script>
    // Auto-scroll on load for existing messages
    (function () {{
      const container = document.getElementById('chat-messages');
      if (container) {{
        container.scrollTop = container.scrollHeight;
      }}
    }})();

    // Render Markdown for existing assistant replies using marked.js
    (function () {{
      if (typeof marked === 'undefined') return;
      const bubbles = document.querySelectorAll('.chat-bubble-assistant');
      bubbles.forEach((bubble) => {{
        const raw = bubble.textContent || '';
        const html = marked.parse(raw);
        bubble.innerHTML = '<div class="markdown-body">' + html + '</div>';
      }});
    }})();

    // Streaming send
    function sendMessage(event) {{
      event.preventDefault();
      const textarea = document.getElementById('chat-input');
      const text = (textarea.value || '').trim();
      if (!text) return false;

      const container = document.getElementById('chat-messages');

      // User bubble
      const userWrapper = document.createElement('div');
      userWrapper.className = 'chat-message chat-message-user';

      const userMeta = document.createElement('div');
      userMeta.className = 'chat-meta';
      userMeta.textContent = 'You';

      const userBubble = document.createElement('div');
      userBubble.className = 'chat-bubble chat-bubble-user';
      userBubble.textContent = text;

      userWrapper.appendChild(userMeta);
      userWrapper.appendChild(userBubble);
      container.appendChild(userWrapper);

      // Assistant bubble
      const assistantWrapper = document.createElement('div');
      assistantWrapper.className = 'chat-message chat-message-assistant';

      const assistantMeta = document.createElement('div');
      assistantMeta.className = 'chat-meta';
      assistantMeta.textContent = 'Assistant';

      const assistantBubble = document.createElement('div');
      assistantBubble.className = 'chat-bubble chat-bubble-assistant';

      const assistantBody = document.createElement('div');
      assistantBody.className = 'markdown-body';
      assistantBubble.appendChild(assistantBody);

      assistantWrapper.appendChild(assistantMeta);
      assistantWrapper.appendChild(assistantBubble);
      container.appendChild(assistantWrapper);

      container.scrollTop = container.scrollHeight;

      // Clear input
      textarea.value = '';

      // Streaming fetch
      let accumulated = '';
      let buffer = '';

      fetch('/stream_chat', {{
        method: 'POST',
        headers: {{
          'Content-Type': 'application/json'
        }},
        body: JSON.stringify({{ message: text }})
      }}).then((resp) => {{
        if (!resp.ok || !resp.body) {{
          assistantBody.textContent = 'Error contacting server (HTTP ' + resp.status + ').';
          return;
        }}
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();

        function readChunk() {{
          reader.read().then(({{
            done,
            value
          }}) => {{
            if (done) {{
              return;
            }}
            const chunk = decoder.decode(value, {{ stream: true }});
            buffer += chunk;
            const lines = buffer.split('\\n');
            buffer = lines.pop(); // keep partial line

            for (const line of lines) {{
              const trimmed = line.trim();
              if (!trimmed) continue;
              let data;
              try {{
                data = JSON.parse(trimmed);
              }} catch (e) {{
                continue;
              }}
              if (data.delta) {{
                accumulated += data.delta;
                if (typeof marked !== 'undefined') {{
                  assistantBody.innerHTML = marked.parse(accumulated);
                }} else {{
                  assistantBody.textContent = accumulated;
                }}
                container.scrollTop = container.scrollHeight;
              }}
            }}

            readChunk();
          }}).catch((err) => {{
            assistantBody.textContent = 'Error reading stream: ' + String(err);
          }});
        }}

        readChunk();
      }}).catch((err) => {{
        assistantBody.textContent = 'Error contacting server: ' + String(err);
      }});

      return false;
    }}
  </script>
</body>
</html>
"""

# === Helper: session / auth ===============================================


def current_user() -> Optional[Tuple[int, str]]:
    user_id = session.get(SESSION_USER_ID_KEY)
    username = session.get(SESSION_USERNAME_KEY)
    if user_id is None or username is None:
        return None
    # Make sure user still exists
    db_user = db.get_user_by_id(int(user_id))
    if db_user is None:
        # Clean up stale session
        session.clear()
        return None
    return db_user


def require_login_redirect():
    user = current_user()
    if user is None:
        return redirect(url_for("login"))
    return None


# === Routes ================================================================


@app.route("/")
def index():
    user = current_user()
    if user is None:
        return redirect(url_for("login"))
    return redirect(url_for("chat"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            return render_template_string(
                LOGIN_TEMPLATE,
                title=f"Sign in · {APP_TITLE}",
                error="Please provide both username and password.",
            )

        auth = db.authenticate_user(username, password)
        if auth is None:
            return render_template_string(
                LOGIN_TEMPLATE,
                title=f"Sign in · {APP_TITLE}",
                error="Invalid username or password.",
            )

        user_id, uname = auth
        session[SESSION_USER_ID_KEY] = user_id
        session[SESSION_USERNAME_KEY] = uname
        return redirect(url_for("chat"))

    return render_template_string(
        LOGIN_TEMPLATE,
        title=f"Sign in · {APP_TITLE}",
        error=None,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            return render_template_string(
                REGISTER_TEMPLATE,
                title=f"Create account · {APP_TITLE}",
                error="Please provide both username and password.",
                info=None,
            )

        success = db.create_user(username, password)
        if not success:
            return render_template_string(
                REGISTER_TEMPLATE,
                title=f"Create account · {APP_TITLE}",
                error="That username is already taken.",
                info=None,
            )

        return render_template_string(
            REGISTER_TEMPLATE,
            title=f"Create account · {APP_TITLE}",
            error=None,
            info="Account created. You can now sign in.",
        )

    return render_template_string(
        REGISTER_TEMPLATE,
        title=f"Create account · {APP_TITLE}",
        error=None,
        info=None,
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/chat", methods=["GET"])
def chat():
    user = current_user()
    if user is None:
        return redirect(url_for("login"))

    user_id, username = user
    info_msg = None
    error_msg = None
    draft_text = None

    chat_history = db.get_recent_messages(
        user_id=user_id, limit=MAX_MESSAGES_PER_CHAT
    )

    return render_template_string(
        CHAT_TEMPLATE,
        title=f"Chat · {APP_TITLE}",
        username=username,
        messages=chat_history,
        info=info_msg,
        error=error_msg,
        draft=draft_text,
        max_messages=MAX_MESSAGES_PER_CHAT,
    )


@app.route("/stream_chat", methods=["POST"])
def stream_chat():
    """Streaming endpoint used by the chat UI JS to get token chunks."""
    user = current_user()
    if user is None:
        return jsonify({"error": "not_authenticated"}), 401

    user_id, _username = user
    data = request.get_json(silent=True) or {}
    raw_message = (data.get("message") or "").strip()

    if not raw_message:
        return jsonify({"error": "empty_message"}), 400

    if len(raw_message) > MAX_MESSAGE_LENGTH:
        raw_message = raw_message[:MAX_MESSAGE_LENGTH]

    system_prompt = system_prompts.get_prompt()
    prior_limit = max(0, MAX_MESSAGES_PER_CHAT - 2)  # leave room for new user + reply
    history = db.get_recent_messages(user_id=user_id, limit=prior_limit)

    messages_for_ollama: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    messages_for_ollama.extend(history)
    messages_for_ollama.append({"role": "user", "content": raw_message})

    def generate():
        # Save user message immediately so a refresh shows it
        db.append_message(user_id, "user", raw_message)

        full_reply_parts: List[str] = []

        try:
            for delta in ollama_client.stream(messages_for_ollama):
                full_reply_parts.append(delta)
                payload = {"delta": delta, "done": False}
                yield json.dumps(payload) + "\n"
        except Exception as exc:  # noqa: BLE001
            err_text = f"\n\n[Error contacting model: {exc}]"
            payload = {"delta": err_text, "done": True}
            yield json.dumps(payload) + "\n"
            return

        full_reply = "".join(full_reply_parts).strip()
        if full_reply:
            db.append_message(user_id, "assistant", full_reply)

        yield json.dumps({"delta": "", "done": True}) + "\n"

    return Response(stream_with_context(generate()), mimetype="text/plain")


# === Main Entry Point ======================================================


def main() -> None:
    print(
        f"[startup] Starting {APP_TITLE} on http://{SERVER_HOST}:{SERVER_PORT} "
        f"(Ollama model: {OLLAMA_MODEL_NAME})"
    )
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=SERVER_DEBUG)


if __name__ == "__main__" and _in_our_venv():
    main()
