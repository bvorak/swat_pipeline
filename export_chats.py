#!/usr/bin/env python3
"""
Export VS Code Copilot chat sessions from the current workspace to Markdown files.

Usage:
    python export_chats.py                  # full export
    python export_chats.py --incremental    # only re-export changed sessions
    python export_chats.py -o ./my_chats    # custom output folder
    python export_chats.py --list           # list sessions without exporting
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── Locate VS Code storage ──────────────────────────────────────────────

def _appdata_code_dir() -> Path:
    """Return the VS Code user-data directory for the current OS."""
    if platform.system() == "Windows":
        base = os.environ.get("APPDATA", "")
        return Path(base) / "Code" / "User"
    elif platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Code" / "User"
    else:  # Linux / WSL
        return Path.home() / ".config" / "Code" / "User"


def find_workspace_hash(workspace_folder: str | None = None) -> str | None:
    """
    Find the workspaceStorage hash that maps to *workspace_folder*.
    If *workspace_folder* is None, use the current working directory.
    """
    ws_root = Path(workspace_folder) if workspace_folder else Path.cwd()
    ws_root_uri = ws_root.as_uri()  # file:///…

    storage_root = _appdata_code_dir() / "workspaceStorage"
    if not storage_root.is_dir():
        return None

    for entry in storage_root.iterdir():
        ws_json = entry / "workspace.json"
        if ws_json.is_file():
            try:
                data = json.loads(ws_json.read_text("utf-8"))
                folder_uri = data.get("folder", "")
                # Normalize for comparison
                if _uris_match(folder_uri, ws_root_uri):
                    return entry.name
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _uris_match(a: str, b: str) -> bool:
    """Case-insensitive, percent-decoded URI comparison."""
    from urllib.parse import unquote
    def norm(u: str) -> str:
        return unquote(u).rstrip("/").replace("\\", "/").lower()
    return norm(a) == norm(b)


# ── JSONL session reconstruction ─────────────────────────────────────────

def _set_nested(obj, keys, value):
    """Set a value in a nested dict/list following *keys* path."""
    for k in keys[:-1]:
        if isinstance(k, int):
            while len(obj) <= k:
                obj.append(None)
            if obj[k] is None:
                obj[k] = {}
            obj = obj[k]
        else:
            if k not in obj:
                obj[k] = {}
            obj = obj[k]
    last = keys[-1]
    if isinstance(last, int):
        while len(obj) <= last:
            obj.append(None)
        obj[last] = value
    else:
        obj[last] = value


def _walk_to(obj, keys):
    """Navigate into *obj* following *keys*, returning the final container and last key."""
    for k in keys[:-1]:
        if isinstance(k, int):
            if isinstance(obj, list) and k < len(obj):
                obj = obj[k]
            else:
                return None, None
        else:
            if isinstance(obj, dict) and k in obj:
                obj = obj[k]
            else:
                return None, None
    return obj, keys[-1]


def reconstruct_session(jsonl_path: Path) -> dict:
    """
    Replay a VS Code chat JSONL file and return the reconstructed session state.

    Handles three JSONL operation kinds:
      kind=0 : full initial snapshot
      kind=1 : scalar property set   {k: path, v: value}
      kind=2 : array operation        {k: path, v: items_to_append}
               or splice/delete       {k: path, i: index_to_delete}
    """
    state = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = entry.get("kind")

            if kind == 0:
                # Initial state snapshot
                state = entry["v"]

            elif kind == 1:
                # Scalar property update
                keys = entry.get("k", [])
                if keys and "v" in entry:
                    _set_nested(state, keys, entry["v"])

            elif kind == 2:
                keys = entry.get("k", [])

                if "i" in entry and "v" not in entry:
                    # Splice / delete operation: remove item at index i
                    container, last_key = _walk_to(state, keys + ["__placeholder"])
                    # container should be the array pointed to by keys
                    target, tkey = _walk_to(state, keys)
                    if target is not None and tkey is not None:
                        arr = target[tkey] if isinstance(target, dict) else (
                            target[tkey] if isinstance(tkey, int) and isinstance(target, list) and tkey < len(target) else None
                        )
                        if isinstance(arr, list):
                            idx = entry["i"]
                            if 0 <= idx < len(arr):
                                arr.pop(idx)
                    continue

                value = entry.get("v")
                if value is None:
                    continue

                if keys == ["requests"] and isinstance(value, list):
                    # Append new request(s)
                    state.setdefault("requests", []).extend(value)
                elif len(keys) >= 2:
                    # Patch a sub-field, e.g. ["requests", 0, "response"]
                    container, last_key = _walk_to(state, keys)
                    if container is not None and last_key is not None:
                        if isinstance(value, list):
                            # Append to existing array
                            existing = None
                            if isinstance(container, dict):
                                existing = container.get(last_key)
                            elif isinstance(container, list) and isinstance(last_key, int) and last_key < len(container):
                                existing = container[last_key]
                            if isinstance(existing, list):
                                existing.extend(value)
                            else:
                                # Set directly
                                _set_nested(state, keys, value)
                        else:
                            _set_nested(state, keys, value)
                else:
                    if "v" in entry:
                        _set_nested(state, keys, value)

    return state


# ── Response item → Markdown text ────────────────────────────────────────

def _extract_text_from_response(response_items: list) -> str:
    """Convert response items into readable Markdown text."""
    parts: list[str] = []
    for item in response_items:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue

        kind = item.get("kind")

        # Plain markdown text fragment (no explicit kind)
        if kind is None and "value" in item:
            val = item["value"]
            if isinstance(val, str):
                parts.append(val)
            continue

        if kind == "markdownContent":
            content = item.get("content", {})
            if isinstance(content, dict):
                parts.append(content.get("value", ""))
            elif isinstance(content, str):
                parts.append(content)

        elif kind == "thinking":
            meta = item.get("metadata", {})
            if isinstance(meta, dict) and meta.get("vscodeReasoningDone"):
                continue  # skip the empty "done" marker
            val = item.get("value", "")
            if not isinstance(val, str):
                val = str(val) if val else ""
            if val.strip():
                parts.append(
                    "\n<details><summary>Thinking</summary>\n\n"
                    + val
                    + "\n\n</details>\n"
                )

        elif kind == "toolInvocationSerialized":
            msg = item.get("pastTenseMessage") or item.get("invocationMessage") or {}
            label = msg.get("value", "") if isinstance(msg, dict) else str(msg)
            if label.strip():
                parts.append(f"\n> **Tool:** {label.strip()}\n")

        elif kind == "textEditGroup":
            # File edits – just note them
            uri = item.get("uri", {})
            fpath = uri.get("path", "") if isinstance(uri, dict) else ""
            if fpath:
                parts.append(f"\n> *Edited file:* `{fpath}`\n")

        elif kind == "codeblockUri":
            uri = item.get("uri", {})
            fpath = uri.get("path", "") if isinstance(uri, dict) else ""
            if fpath:
                parts.append(f"\n> *Code block file:* `{fpath}`\n")

        elif kind == "progressMessage":
            content = item.get("content", {})
            val = content.get("value", "") if isinstance(content, dict) else ""
            if val.strip():
                parts.append(f"\n> *Progress:* {val.strip()}\n")

        # Silently skip unknown kinds (mcpServersStarting, etc.)

    return "".join(parts)


# ── Format one session to Markdown ───────────────────────────────────────

def _ts_to_iso(ts_ms: int | None) -> str:
    if ts_ms is None:
        return "unknown"
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
    except (OSError, ValueError):
        return str(ts_ms)


def _sanitize_filename(name: str, max_len: int = 120) -> str:
    """Turn a session title into a safe filename stem."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"_+", "_", name).strip("_ ")
    if len(name) > max_len:
        name = name[:max_len].rsplit("_", 1)[0]
    return name or "untitled"


def session_to_markdown(state: dict, session_id: str, source_mtime: float) -> str:
    """Render a reconstructed session state as a Markdown document."""
    title = state.get("customTitle") or f"Chat {session_id[:8]}"
    creation = state.get("creationDate")
    requests = state.get("requests", [])

    # Determine model used (from inputState or first request)
    model = ""
    input_state = state.get("inputState", {})
    sel_model = input_state.get("selectedModel", {})
    if isinstance(sel_model, dict):
        model = sel_model.get("identifier", "")
    if not model and requests:
        model = requests[0].get("modelId", "")

    # YAML front-matter
    lines = [
        "---",
        f"title: \"{title}\"",
        f"session_id: {session_id}",
        f"created: {_ts_to_iso(creation)}",
        f"exported: {datetime.now(tz=timezone.utc).isoformat()}",
        f"source_modified: {datetime.fromtimestamp(source_mtime, tz=timezone.utc).isoformat()}",
        f"model: {model}",
        f"turns: {len(requests)}",
        f"mode: {input_state.get('mode', {}).get('kind', 'unknown') if isinstance(input_state.get('mode'), dict) else 'unknown'}",
        "---",
        "",
        f"# {title}",
        "",
    ]

    for i, req in enumerate(requests):
        # User message
        msg = req.get("message", {})
        user_text = ""
        if isinstance(msg, dict):
            user_text = msg.get("text", "")
        elif isinstance(msg, str):
            user_text = msg

        lines.append(f"## Turn {i + 1}")
        lines.append("")
        lines.append("### User")
        lines.append("")
        lines.append(user_text.strip() if user_text else "*[empty message]*")
        lines.append("")

        # Assistant response
        response = req.get("response", [])
        if isinstance(response, list) and response:
            assistant_text = _extract_text_from_response(response)
            lines.append("### Assistant")
            lines.append("")
            lines.append(assistant_text.strip() if assistant_text.strip() else "*[no text response]*")
            lines.append("")

        # Separator
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ── Incremental tracking ────────────────────────────────────────────────

MANIFEST_NAME = ".chat_export_manifest.json"


def load_manifest(output_dir: Path) -> dict:
    mf = output_dir / MANIFEST_NAME
    if mf.is_file():
        try:
            return json.loads(mf.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_manifest(output_dir: Path, manifest: dict):
    mf = output_dir / MANIFEST_NAME
    mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _file_hash(path: Path) -> str:
    """Fast hash of file contents for change detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ── Main export logic ───────────────────────────────────────────────────

def export_workspace_chats(
    workspace_folder: str | None = None,
    output_dir: str | None = None,
    incremental: bool = False,
    list_only: bool = False,
    verbose: bool = False,
):
    ws_root = Path(workspace_folder) if workspace_folder else Path.cwd()
    ws_hash = find_workspace_hash(str(ws_root))

    if ws_hash is None:
        print(f"ERROR: Could not find VS Code workspace storage for:\n  {ws_root}")
        print("Make sure you've opened this folder in VS Code at least once.")
        sys.exit(1)

    storage = _appdata_code_dir() / "workspaceStorage" / ws_hash
    sessions_dir = storage / "chatSessions"

    if not sessions_dir.is_dir():
        print(f"No chatSessions folder found in {storage}")
        sys.exit(1)

    # Collect JSONL session files
    session_files = sorted(sessions_dir.glob("*.jsonl"))
    if not session_files:
        print("No chat sessions found.")
        return

    # Output directory
    if output_dir:
        out = Path(output_dir)
    else:
        out = ws_root / "chat_exports"
    out.mkdir(parents=True, exist_ok=True)

    # Incremental manifest
    manifest = load_manifest(out) if incremental else {}

    if list_only:
        print(f"Found {len(session_files)} chat session(s) in workspace storage:\n")

    exported = 0
    skipped = 0

    for sf in session_files:
        session_id = sf.stem
        file_hash = _file_hash(sf)
        mtime = sf.stat().st_mtime

        if incremental and manifest.get(session_id, {}).get("hash") == file_hash:
            skipped += 1
            if verbose:
                print(f"  SKIP (unchanged): {session_id}")
            continue

        # Reconstruct session
        try:
            state = reconstruct_session(sf)
        except Exception as exc:
            print(f"  WARN: Could not parse {sf.name}: {exc}")
            continue

        title = state.get("customTitle") or f"Chat {session_id[:8]}"
        creation = state.get("creationDate")
        n_turns = len(state.get("requests", []))

        if list_only:
            created_str = _ts_to_iso(creation)
            size_kb = sf.stat().st_size / 1024
            print(f"  {title}")
            print(f"    id={session_id}  turns={n_turns}  created={created_str}  size={size_kb:.0f}KB")
            continue

        # Generate markdown
        md = session_to_markdown(state, session_id, mtime)

        # Filename: sanitized title + short hash to avoid collisions
        safe_name = _sanitize_filename(title)
        fname = f"{safe_name}__{session_id[:8]}.md"
        out_path = out / fname

        out_path.write_text(md, encoding="utf-8")
        exported += 1

        # Update manifest
        manifest[session_id] = {
            "hash": file_hash,
            "file": fname,
            "title": title,
            "turns": n_turns,
            "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        if verbose:
            print(f"  EXPORTED: {fname} ({n_turns} turns)")

    if not list_only:
        save_manifest(out, manifest)
        print(f"\nExported {exported} session(s) to {out}")
        if skipped:
            print(f"Skipped {skipped} unchanged session(s)")
        print(f"Total sessions: {len(session_files)}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export VS Code Copilot chat sessions to Markdown files."
    )
    parser.add_argument(
        "-w", "--workspace",
        help="Path to the workspace folder (default: current directory).",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory for markdown files (default: <workspace>/chat_exports).",
    )
    parser.add_argument(
        "--incremental", "-i",
        action="store_true",
        help="Only re-export sessions that changed since last export.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_only",
        help="List sessions without exporting.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-session details.",
    )
    args = parser.parse_args()

    export_workspace_chats(
        workspace_folder=args.workspace,
        output_dir=args.output,
        incremental=args.incremental,
        list_only=args.list_only,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
