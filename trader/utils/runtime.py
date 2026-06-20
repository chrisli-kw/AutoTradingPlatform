import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4


RUNTIME_ROOT = Path("runtime")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
INDICATOR_RE = re.compile(
    r"\[(Open|Close)\]\s*當前指數:\s*(?P<price>-?\d+)"
    r"(?:;\s*COST:\s*(?P<cost>[^,;]+))?"
    r".*?ATR:\s*(?P<atr>[^,;]+)"
    r".*?STD15:\s*(?P<std15>[^,;]+)"
    r".*?DTP:\s*(?P<dtp>[^,;]+)"
    r".*?TR:\s*(?P<tr>[^,;]+)"
)


def _account_dir(account: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", account or "unknown")
    path = RUNTIME_ROOT / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path, default):
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_status(account: str, status: str, message: str = "", **extra):
    data = {
        "account": account,
        "status": status,
        "message": message,
        "updated_at": now_text(),
    }
    data.update(extra)
    _write_json(_account_dir(account) / "status.json", data)


def read_status(account: str) -> dict:
    return _read_json(_account_dir(account) / "status.json", {})


def write_session(account: str, **data):
    payload = read_session(account)
    payload.update(data)
    payload["updated_at"] = now_text()
    _write_json(_account_dir(account) / "session.json", payload)


def read_session(account: str) -> dict:
    return _read_json(_account_dir(account) / "session.json", {})


def write_command(account: str, command: str, payload: dict | None = None):
    data = {
        "id": uuid4().hex,
        "account": account,
        "command": command,
        "payload": payload or {},
        "created_at": now_text(),
    }
    _write_json(_account_dir(account) / "control.json", data)
    return data


def read_command(account: str) -> dict:
    return _read_json(_account_dir(account) / "control.json", {})


def command_is_expired(command: dict, ttl_seconds: int = 300) -> bool:
    created_at = command.get("created_at")
    if not created_at:
        return False

    try:
        created = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False

    return (datetime.now() - created).total_seconds() > ttl_seconds


def clear_command(account: str, command_id: str | None = None):
    path = _account_dir(account) / "control.json"
    data = _read_json(path, {})
    if not data:
        return
    if command_id and data.get("id") != command_id:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def tail_lines(path: Path, limit: int = 300) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return [strip_ansi(line.rstrip("\n")) for line in lines[-limit:]]
    except FileNotFoundError:
        return []


def parse_latest_indicator(lines: list[str]) -> dict:
    for line in reversed(lines):
        clean = strip_ansi(line)
        match = INDICATOR_RE.search(clean)
        if not match:
            continue

        data = match.groupdict()
        data["side"] = match.group(1)
        data["raw"] = clean
        return data

    return {}
