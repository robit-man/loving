#!/usr/bin/env python3
"""Standalone NKN ⇄ Ollama relay (no embedded web server)."""

from __future__ import annotations

import atexit
import contextlib
import datetime
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# === Virtualenv bootstrap ====================================================

BASE_DIR: Path = Path(__file__).resolve().parent
VENV_DIR: Path = BASE_DIR / ".venv"
VENV_BIN_DIR: Path = VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
VENV_PYTHON: Path = VENV_BIN_DIR / ("python.exe" if os.name == "nt" else "python")

REQUIRED_PACKAGES = ["requests"]


def _in_managed_venv() -> bool:
    try:
        return VENV_DIR.resolve() == Path(sys.prefix).resolve()
    except Exception:  # pragma: no cover - safety fallthrough
        return False


def _bootstrap_venv_if_needed() -> None:
    if __name__ != "__main__":
        return
    if _in_managed_venv():
        return

    if not VENV_PYTHON.exists():
        print("[bootstrap] Creating .venv …")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
        if REQUIRED_PACKAGES:
            subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", *REQUIRED_PACKAGES])

    env = os.environ.copy()
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]], env)


_bootstrap_venv_if_needed()

# === Runtime imports ========================================================

import requests

# === App configuration ======================================================

APP_TITLE = "Loving Qwen Relay"
SYSTEM_PROMPT_FILE = BASE_DIR / "system.md"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_CHAT_ENDPOINT = "/api/chat"
OLLAMA_TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT", "300"))
MAX_CONTEXT_MESSAGES = int(os.environ.get("MAX_CONTEXT", "30"))
MAX_MESSAGE_LENGTH = 4000

DATABASE_PATH = BASE_DIR / "chat.db"
SQLITE_FOREIGN_KEYS_ON = "PRAGMA foreign_keys = ON;"

PASSWORD_HASH_ALGORITHM = "sha256"
PASSWORD_PBKDF2_ITERATIONS = 150_000
PASSWORD_SALT_BYTES = 16
PASSWORD_HASH_SEPARATOR = "$"

NKN_IDENTIFIER = os.environ.get("NKN_IDENTIFIER", "loving-relay")
NKN_NUM_SUBCLIENTS = int(os.environ.get("NKN_NUM_SUBCLIENTS", "10"))
LEGACY_BRIDGE_DIR = BASE_DIR / ".nkn_bridge"
DEFAULT_BRIDGE_DIR = BASE_DIR / "nkn_bridge"
_bridge_dir_env = os.environ.get("NKN_BRIDGE_DIR")
if _bridge_dir_env:
    NKN_BRIDGE_DIR = Path(_bridge_dir_env).expanduser()
else:
    if LEGACY_BRIDGE_DIR.exists() and not DEFAULT_BRIDGE_DIR.exists():
        NKN_BRIDGE_DIR = LEGACY_BRIDGE_DIR
    else:
        NKN_BRIDGE_DIR = DEFAULT_BRIDGE_DIR
NKN_JS_NAME = "bridge.js"
NKN_PACKAGE_VERSION = "1.3.6"


# === SQLite helpers =========================================================


def hash_password(plaintext: str) -> str:
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
    try:
        algo, iterations_str, salt_hex, hash_hex = stored.split(PASSWORD_HASH_SEPARATOR)
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        return False

    candidate = hashlib.pbkdf2_hmac(
        algo,
        plaintext.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate, expected)


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

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
                    created_at TEXT NOT NULL,
                    last_nkn_addr TEXT DEFAULT ''
                );
                """
            )
            try:
                cur.execute(
                    "ALTER TABLE users ADD COLUMN last_nkn_addr TEXT DEFAULT ''"
                )
                self._connection.commit()
            except sqlite3.OperationalError:
                pass
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )
            self._connection.commit()

    def create_user(self, username: str, password: str) -> Optional[int]:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        password_hash = hash_password(password)
        with self._lock:
            try:
                cur = self._connection.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?);",
                    (username, password_hash, now),
                )
                self._connection.commit()
                return int(cur.lastrowid)
            except sqlite3.IntegrityError:
                return None

    def authenticate_user(self, username: str, password: str) -> Optional[Tuple[int, str]]:
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

    def append_message(self, user_id: int, role: str, content: str) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            self._connection.execute(
                "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?);",
                (user_id, role, content, now),
            )
            self._connection.commit()

    def get_recent_messages(self, user_id: int, limit: int) -> List[Dict[str, str]]:
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
        ordered = list(reversed(rows))
        return [
            {"role": row["role"], "content": row["content"]}
            for row in ordered
        ]

    def update_last_nkn_addr(self, user_id: int, address: str) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE users SET last_nkn_addr = ? WHERE id = ?;",
                (address, user_id),
            )
            self._connection.commit()


class SessionManager:
    """Manages user sessions bound to NKN addresses with expiry."""
    def __init__(self) -> None:
        self._sessions: Dict[str, Tuple[int, str, float]] = {}  # src -> (user_id, username, timestamp)
        self._lock = threading.Lock()
        self._session_timeout_s = 3600 * 24  # 24 hours

    def login(self, src: str, user_id: int, username: str) -> None:
        """Create a new session bound to the NKN address."""
        with self._lock:
            self._sessions[src] = (user_id, username, time.time())
            print(f"[session] Created session for user_id={user_id} username={username} addr={src}")

    def logout(self, src: str) -> None:
        """Remove session for the given NKN address."""
        with self._lock:
            if src in self._sessions:
                user_id, username, _ = self._sessions[src]
                print(f"[session] Logout user_id={user_id} username={username} addr={src}")
            self._sessions.pop(src, None)

    def get(self, src: str) -> Optional[Tuple[int, str]]:
        """Get session for the given NKN address, validating it hasn't expired."""
        with self._lock:
            session = self._sessions.get(src)
            if session is None:
                return None
            user_id, username, timestamp = session
            # Check if session expired
            if time.time() - timestamp > self._session_timeout_s:
                print(f"[session] Expired session for user_id={user_id} addr={src}")
                self._sessions.pop(src, None)
                return None
            return (user_id, username)

    def refresh(self, src: str) -> None:
        """Refresh session timestamp on activity."""
        with self._lock:
            session = self._sessions.get(src)
            if session:
                user_id, username, _ = session
                self._sessions[src] = (user_id, username, time.time())

    def validate_user_address(self, db: Database, src: str, user_id: int) -> bool:
        """Verify that the user's last known NKN address matches the current one."""
        # This adds an extra layer of security by verifying the stored address
        # matches the current connection address
        return True  # For now, rely on session binding; can be enhanced later

# === Helpers: system prompt and Ollama ======================================


class SystemPromptLoader:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._cached_text: Optional[str] = None
        self._cached_mtime: Optional[float] = None

    def get_prompt(self) -> str:
        if not self._path.exists():
            return "You are a helpful assistant."
        mtime = self._path.stat().st_mtime
        if self._cached_text is None or self._cached_mtime != mtime:
            self._cached_text = self._path.read_text(encoding="utf-8")
            self._cached_mtime = mtime
        return self._cached_text


class OllamaClient:
    def __init__(self, base_url: str, endpoint: str, model_name: str, timeout_s: int) -> None:
        self._base = base_url.rstrip("/")
        self._endpoint = endpoint
        self._model = model_name
        self._timeout = timeout_s

    def stream(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        url = f"{self._base}{self._endpoint}"
        payload = {"model": self._model, "messages": messages, "stream": True}
        response = requests.post(url, json=payload, timeout=self._timeout, stream=True)
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


# === NKN Bridge =============================================================

BRIDGE_JS_SOURCE = """
'use strict';
const nkn = require('nkn-sdk');
const readline = require('readline');

const SEED_HEX = (process.env.NKN_SEED_HEX || '').toLowerCase().replace(/^0x/, '');
const IDENTIFIER = process.env.NKN_IDENTIFIER || 'loving-relay';
const SUBCLIENTS = parseInt(process.env.NKN_NUM_SUBCLIENTS || '10', 10) || 10;
const SEED_WS = (process.env.NKN_BRIDGE_SEED_WS || '').split(',').map(s => s.trim()).filter(Boolean);

function emit(obj) {
  try { process.stdout.write(JSON.stringify(obj) + '\\n'); }
  catch (err) { /* swallow */ }
}

if (!/^[0-9a-f]{64}$/.test(SEED_HEX)) {
  emit({ type: 'fatal', error: 'NKN_SEED_HEX missing or invalid' });
  process.exit(1);
}

const client = new nkn.MultiClient({
  seed: SEED_HEX,
  identifier: IDENTIFIER,
  numSubClients: SUBCLIENTS,
  seedWsAddr: SEED_WS.length ? SEED_WS : undefined,
  wsConnHeartbeatTimeout: 120000,
});

client.on('connect', () => {
  emit({ type: 'ready', address: String(client.addr || '') });
});

client.onMessage(({src, payload, payloadType}) => {
  let text = '';
  if (Buffer.isBuffer(payload)) {
    text = payload.toString('utf8');
  } else if (typeof payload === 'string') {
    text = payload;
  } else if (payload && payload.payload) {
    const inner = payload.payload;
    text = Buffer.isBuffer(inner) ? inner.toString('utf8') : String(inner);
  } else {
    text = String(payload || '');
  }
  let parsed = null;
  try { parsed = JSON.parse(text); }
  catch (err) {
    parsed = { event: 'relay.raw', raw: text };
  }
  emit({ type: 'message', src: String(src || ''), payload: parsed });
  // Return false to prevent automatic ACK - we'll send responses manually for streaming
  return false;
});

client.on('error', (err) => {
  emit({ type: 'status', level: 'error', message: String(err && err.message || err) });
});

client.on('close', () => {
  emit({ type: 'status', level: 'close' });
  process.exit(2);
});

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  let cmd;
  try { cmd = JSON.parse(line); }
  catch (err) { return; }
  if (!cmd || cmd.type !== 'send') return;
  const body = (cmd.data !== undefined) ? cmd.data : cmd.payload;
  if (!cmd.to || body === undefined) return;

  // Use noReply: true for one-way messages (responses, notifications)
  // Use noReply: false (default) for request-reply pattern
  const opts = cmd.opts !== undefined ? cmd.opts : { noReply: true };

  client.send(String(cmd.to), JSON.stringify(body), opts).catch((err) => {
    emit({ type: 'status', level: 'send_error', message: String(err && err.message || err) });
  });
});
"""


class NKNBridge:
    def __init__(self, workdir: Path) -> None:
        self._dir = workdir
        self._pkg = self._dir / "package.json"
        self._js = self._dir / NKN_JS_NAME
        self._seed_file = self._dir / "seed_hex.txt"
        self._proc: Optional[subprocess.Popen[str]] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._send_lock = threading.Lock()
        self._listeners: List[Callable[[str, Dict[str, Any]], None]] = []
        self.address: Optional[str] = None
        self.public_key: Optional[str] = None
        self.enabled = False
        self._ready_event = threading.Event()
        self._start_lock = threading.Lock()
        self._should_run = True
        self._watchdog_thread: Optional[threading.Thread] = None

        self._node = shutil.which("node")
        self._npm = shutil.which("npm")
        if not self._node or not self._npm:
            print("[nkn] node and npm are required for the bridge; skipping startup", file=sys.stderr)
            return
        self.enabled = True
        self._ensure_bridge_artifacts()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    # ------------------------------------------------------------------ utils
    def _ensure_bridge_artifacts(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        if not self._pkg.exists():
            subprocess.check_call([self._npm, "init", "-y"], cwd=self._dir)
        node_modules = self._dir / "node_modules"
        target_pkg = node_modules / "nkn-sdk"
        if not target_pkg.exists():
            subprocess.check_call([self._npm, "install", f"nkn-sdk@{NKN_PACKAGE_VERSION}"], cwd=self._dir)
        if not self._js.exists() or self._js.read_text(encoding="utf-8") != BRIDGE_JS_SOURCE:
            self._js.write_text(BRIDGE_JS_SOURCE, encoding="utf-8")
        if not self._seed_file.exists():
            self._seed_file.write_text(secrets.token_hex(32))

    def _env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.setdefault("NKN_IDENTIFIER", NKN_IDENTIFIER)
        env.setdefault("NKN_NUM_SUBCLIENTS", str(NKN_NUM_SUBCLIENTS))
        env["NKN_SEED_HEX"] = self._seed_file.read_text().strip()
        return env

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if not self.enabled:
            return
        with self._start_lock:
            if self._proc and self._proc.poll() is None:
                return
            self._ready_event.clear()
            try:
                self._proc = subprocess.Popen(
                    [self._node, str(self._js)],
                    cwd=self._dir,
                    env=self._env(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                print("[nkn] bridge process started")
            except Exception as exc:  # pragma: no cover - startup log
                print(f"[nkn] failed to start bridge: {exc}", file=sys.stderr)
                return
            self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
            self._stdout_thread.start()
            self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self._stderr_thread.start()
            state = "unknown"
            if self._proc and self._proc.stdin:
                try:
                    self._proc.stdin.write(json.dumps({"type": "handshake"}) + "\n")
                    self._proc.stdin.flush()
                    state = "handshake"  # best effort marker
                except Exception:
                    state = "handshake_failed"
            print(f"[nkn] bridge state={state}")
            self._start_time = time.time()

    def wait_ready(self, timeout: Optional[float] = None) -> Optional[str]:
        if self._ready_event.wait(timeout):
            return self.address
        return None

    def stop(self) -> None:
        self._should_run = False
        if not self._proc:
            return
        with contextlib.suppress(Exception):
            self._proc.terminate()
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=3)
        self._proc = None
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2)

    # ----------------------------------------------------------------- comms
    def register_listener(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self._listeners.append(callback)

    def send(self, to_addr: str, payload: Dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("NKN bridge is not running")
        line = json.dumps({"type": "send", "to": to_addr, "data": payload})
        with self._send_lock:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()

    # ----------------------------------------------------------------- IO loops
    def _read_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for raw in self._proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "ready":
                self.address = event.get("address") or self.address
                self.public_key = self._extract_pubkey(self.address)
                self._ready_event.set()
                print(f"[nkn] ready at {self.address}")
                if self.public_key:
                    print(f"[nkn] pubkey {self.public_key}")
            elif etype == "message":
                src = event.get("src") or ""
                payload = event.get("payload") or {}
                for cb in list(self._listeners):
                    try:
                        cb(str(src), payload)
                    except Exception as exc:  # pragma: no cover - logging only
                        print(f"[nkn] listener error: {exc}", file=sys.stderr)
            elif etype == "status":
                level = event.get("level") or "info"
                msg = event.get("message") or ""
                print(f"[nkn] {level}: {msg}")
            elif etype == "fatal":
                print(f"[nkn] fatal: {event.get('error')}", file=sys.stderr)

    def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            text = line.strip()
            if text:
                print(f"[nkn-node] {text}", file=sys.stderr)

    @staticmethod
    def _extract_pubkey(address: Optional[str]) -> Optional[str]:
        if not address:
            return None
        addr = str(address).strip()
        if not addr:
            return None
        if "." not in addr:
            return addr
        parts = [p for p in addr.split(".") if p]
        if not parts:
            return None
        return parts[-1]

    def _watchdog_loop(self) -> None:
        while self._should_run:
            if not self.enabled:
                time.sleep(5)
                continue
            proc = self._proc
            running = proc and proc.poll() is None
            if not running:
                print("[nkn] bridge offline, attempting restart…")
                self.start()
                time.sleep(2)
                continue
            # health check: ensure stdout still active by sending ping
            try:
                if proc and proc.stdin:
                    proc.stdin.write(json.dumps({"type": "ping", "ts": time.time()}) + "\n")
                    proc.stdin.flush()
            except Exception:
                print("[nkn] ping failed, restarting bridge…")
                with contextlib.suppress(Exception):
                    proc.terminate()
                continue
            time.sleep(5)


# === NKN relay server =======================================================

class NKNRelayServer:
    def __init__(
        self,
        bridge: NKNBridge,
        ollama: OllamaClient,
        prompts: SystemPromptLoader,
        db: Database,
        sessions: SessionManager,
    ) -> None:
        self._bridge = bridge
        self._ollama = ollama
        self._prompts = prompts
        self._db = db
        self._sessions = sessions
        self._active: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._known_clients: set[str] = set()
        bridge.register_listener(self._handle_message)

    def _send(self, to_addr: str, payload: Dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            self._bridge.send(to_addr, payload)

    def _reply(self, src: str, req_id: Optional[str], event: str, **extra: Any) -> None:
        data = {"event": event}
        if req_id is not None:
            data["id"] = req_id
        data.update(extra)
        self._send(src, data)

    def _handle_message(self, src: str, payload: Dict[str, Any]) -> None:
        event = payload.get("event")
        if src and src not in self._known_clients:
            self._known_clients.add(src)
            print(f"[nkn] new client {src} (total={len(self._known_clients)})")
        if event == "relay.info":
            self._reply(
                src,
                payload.get("id"),
                "relay.info",
                model=OLLAMA_MODEL_NAME,
                address=self._bridge.address,
                ts=time.time(),
                pubkey=self._bridge.public_key,
                peer_count=len(self._known_clients),
            )
            return
        if event == "auth.register":
            self._handle_register(src, payload)
            return
        if event == "auth.login":
            self._handle_login(src, payload)
            return
        if event == "auth.logout":
            self._handle_logout(src, payload)
            return
        if event == "auth.status":
            self._handle_status(src, payload)
            return
        if event == "chat.begin":
            self._handle_chat_begin(src, payload)
            return
        if event == "chat.refresh":
            self._handle_chat_refresh(src, payload)
            return
        if event == "chat.abort":
            req_id = payload.get("id")
            if req_id:
                self._cancel(req_id)

    def _handle_register(self, src: str, payload: Dict[str, Any]) -> None:
        req_id = payload.get("id")
        username = (payload.get("username") or "").strip()
        password = payload.get("password") or ""
        if not username or not password:
            self._reply(src, req_id, "auth.register.error", message="username and password required")
            print(f"[auth] Register attempt with missing credentials from {src}")
            return
        if len(username) > 64:
            self._reply(src, req_id, "auth.register.error", message="username too long")
            return
        if len(password) < 6:
            self._reply(src, req_id, "auth.register.error", message="password must be at least 6 characters")
            return
        user_id = self._db.create_user(username, password)
        if user_id is None:
            self._reply(src, req_id, "auth.register.error", message="username already exists")
            print(f"[auth] Failed registration - username '{username}' already exists (from {src})")
            return
        self._db.update_last_nkn_addr(user_id, src)
        print(f"[auth] Registered new user '{username}' (user_id={user_id}) from {src}")
        self._reply(src, req_id, "auth.register.ok", username=username, user_id=user_id)

    def _handle_login(self, src: str, payload: Dict[str, Any]) -> None:
        req_id = payload.get("id")
        username = (payload.get("username") or "").strip()
        password = payload.get("password") or ""
        if not username or not password:
            self._reply(src, req_id, "auth.login.error", message="username and password required")
            print(f"[auth] Login attempt with missing credentials from {src}")
            return
        auth = self._db.authenticate_user(username, password)
        if auth is None:
            self._reply(src, req_id, "auth.login.error", message="invalid credentials")
            print(f"[auth] Failed login attempt for username '{username}' from {src}")
            return
        user_id, uname = auth

        # Check if user is already logged in from a different address
        existing_session = self._find_user_session(user_id)
        if existing_session and existing_session != src:
            print(f"[auth] User '{uname}' switching from {existing_session} to {src}")
            self._sessions.logout(existing_session)

        # Create new session bound to this NKN address
        self._sessions.login(src, user_id, uname)
        self._db.update_last_nkn_addr(user_id, src)
        print(f"[auth] Login OK for '{uname}' (user_id={user_id}) from {src}")

        history = self._db.get_recent_messages(user_id=user_id, limit=MAX_CONTEXT_MESSAGES)
        self._reply(
            src,
            req_id,
            "auth.login.ok",
            username=uname,
            user_id=user_id,
            messages=history,
            nkn_address=src,
        )

    def _find_user_session(self, user_id: int) -> Optional[str]:
        """Find the NKN address associated with a user's active session."""
        with self._sessions._lock:
            for addr, (uid, _, _) in self._sessions._sessions.items():
                if uid == user_id:
                    return addr
        return None

    def _handle_logout(self, src: str, payload: Dict[str, Any]) -> None:
        req_id = payload.get("id")
        self._sessions.logout(src)
        self._reply(src, req_id, "auth.logout.ok")

    def _handle_status(self, src: str, payload: Dict[str, Any]) -> None:
        req_id = payload.get("id")
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "auth.status", authenticated=False)
        else:
            _user_id, username = session
            self._reply(src, req_id, "auth.status", authenticated=True, username=username)

    def _handle_chat_refresh(self, src: str, payload: Dict[str, Any]) -> None:
        """Handle request to refresh conversation history from database."""
        req_id = payload.get("id")
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "chat.refresh.error", error="not_authenticated")
            return
        user_id, username = session

        # Refresh session timestamp on activity
        self._sessions.refresh(src)

        # Get recent messages from database
        messages = self._db.get_recent_messages(user_id=user_id, limit=MAX_CONTEXT_MESSAGES)
        self._reply(
            src,
            req_id,
            "chat.refresh.ok",
            messages=messages,
            timestamp=time.time(),
        )

    def _handle_chat_begin(self, src: str, payload: Dict[str, Any]) -> None:
        req_id = payload.get("id") or f"dm-{int(time.time() * 1000)}"
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "chat.error", error="not_authenticated")
            print(f"[chat] Rejected unauthenticated request {req_id} from {src}")
            return
        user_id, username = session

        # Refresh session timestamp on activity
        self._sessions.refresh(src)

        text = (payload.get("message") or "").strip()
        if not text:
            self._reply(src, req_id, "chat.error", error="empty_message")
            return
        text = text[:MAX_MESSAGE_LENGTH]
        print(
            f"[chat] Start req={req_id} user={username} (user_id={user_id}) src={src} chars={len(text)}"
        )
        threading.Thread(
            target=self._serve_chat,
            args=(src, req_id, user_id, text),
            daemon=True,
        ).start()

    def _cancel(self, req_id: str) -> None:
        with self._lock:
            evt = self._active.get(req_id)
            if evt:
                evt.set()

    def _serve_chat(self, src: str, req_id: str, user_id: int, message_text: str) -> None:
        cancel_flag = threading.Event()
        with self._lock:
            self._active[req_id] = cancel_flag
        history_limit = max(0, MAX_CONTEXT_MESSAGES - 1)
        history = self._db.get_recent_messages(user_id=user_id, limit=history_limit)
        self._db.append_message(user_id, "user", message_text)
        system_prompt = self._prompts.get_prompt()
        assembled: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        assembled.extend(history)
        assembled.append({"role": "user", "content": message_text})
        self._send(src, {"event": "chat.ack", "id": req_id, "model": OLLAMA_MODEL_NAME})

        # Streaming with batched deltas
        full_reply_parts: List[str] = []
        seq_num = 0
        batch_buffer: List[str] = []
        batch_start_seq = 0
        last_send_time = time.time()
        batch_max_chars = 100
        batch_max_delay_ms = 50

        def flush_batch():
            nonlocal batch_buffer, batch_start_seq, last_send_time, seq_num
            if not batch_buffer:
                return
            batch_delta = "".join(batch_buffer)
            self._send(src, {
                "event": "chat.delta",
                "id": req_id,
                "delta": batch_delta,
                "seq": batch_start_seq,
                "batch_size": len(batch_buffer),
                "timestamp": time.time()
            })
            seq_num = batch_start_seq + len(batch_buffer)
            batch_buffer = []
            batch_start_seq = seq_num
            last_send_time = time.time()

        try:
            for delta in self._ollama.stream(assembled):
                if cancel_flag.is_set():
                    break
                if delta:
                    full_reply_parts.append(delta)
                    batch_buffer.append(delta)

                    # Calculate batch metrics
                    batch_chars = sum(len(d) for d in batch_buffer)
                    time_since_send = (time.time() - last_send_time) * 1000  # ms

                    # Flush batch if it's large enough or enough time has passed
                    if batch_chars >= batch_max_chars or time_since_send >= batch_max_delay_ms:
                        flush_batch()

            # Flush any remaining batched deltas
            flush_batch()

            # Send completion marker with final sequence number and full content
            self._send(src, {
                "event": "chat.done",
                "id": req_id,
                "done": True,
                "total_seq": seq_num,
                "final_content": "".join(full_reply_parts)
            })

        except Exception as exc:  # pragma: no cover - network path
            self._send(src, {
                "event": "chat.error",
                "id": req_id,
                "error": str(exc),
                "partial": len(full_reply_parts) > 0
            })
            print(f"[chat] error req={req_id} src={src}: {exc}")
            return
        finally:
            with self._lock:
                self._active.pop(req_id, None)

        full_reply = "".join(full_reply_parts).strip()
        if full_reply:
            self._db.append_message(user_id, "assistant", full_reply)
        print(
            f"[chat] complete req={req_id} src={src} reply_chars={len(full_reply)} deltas={seq_num}"
        )


# === Entrypoint =============================================================

db = Database(DATABASE_PATH)
sessions = SessionManager()
system_prompts = SystemPromptLoader(SYSTEM_PROMPT_FILE)
ollama_client = OllamaClient(
    base_url=OLLAMA_BASE_URL,
    endpoint=OLLAMA_CHAT_ENDPOINT,
    model_name=OLLAMA_MODEL_NAME,
    timeout_s=OLLAMA_TIMEOUT_S,
)


def main() -> None:
    print(f"[startup] {APP_TITLE}")
    print(f"[startup] Ollama target: {OLLAMA_BASE_URL} (model {OLLAMA_MODEL_NAME})")
    nkn_bridge = NKNBridge(NKN_BRIDGE_DIR)
    if not nkn_bridge.enabled:
        print("[error] NKN bridge unavailable (requires node + npm)")
        sys.exit(1)

    nkn_bridge.start()
    atexit.register(nkn_bridge.stop)
    NKNRelayServer(nkn_bridge, ollama_client, system_prompts, db, sessions)

    addr = nkn_bridge.wait_ready(timeout=60)
    if addr:
        print(f"[ready] Relay NKN address: {addr}")
        if nkn_bridge.public_key:
            print(f"[ready] Relay pubkey: {nkn_bridge.public_key}")
        print("[info] Paste this address into the web UI's relay box to start chatting.")
    else:
        print("[warn] Relay not ready yet; waiting in background…")

    print("[info] Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[shutdown] Stopping relay…")


if __name__ == "__main__" and _in_managed_venv():
    main()
