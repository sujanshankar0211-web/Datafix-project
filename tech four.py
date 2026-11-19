DataFix - complete source code
Project root: datafix_web
Files: 22

==============================================================================
FILE: app.py
==============================================================================
"""DataFix web app (Flask).

Upload an Excel/CSV file, review the detected issues, apply or reject
corrections and download the cleaned workbook plus a correction report.

The uploaded file is stored untouched next to a pickled working copy; every
change lives in state.json so it can be undone and exported at any time.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import pickle
import re
import secrets
import shutil
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, session

import engine
from engine import (
    CATEGORY_LABELS,
    STATUS_LABELS,
    analyze,
    as_text,
    build_cleaned_workbook,
    build_report_workbook,
    coerce_value,
)
from datafix import config as workspace_config
from datafix.api import bp as workspace_api

BASE_DIR = Path(__file__).resolve().parent
WORKSPACES = BASE_DIR / "workspaces"
WORKSPACES.mkdir(exist_ok=True)
PREFERENCES = BASE_DIR / "preferences"
PREFERENCES.mkdir(exist_ok=True)
PAGE_SIZE = 25
MAX_MB = workspace_config.MAX_UPLOAD_MB
ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".csv"}
OPEN_STATUSES = {"pending", "missing", "duplicate"}

APPEARANCE_MODES = ("light", "dark", "system")
THEME_IDS = (
    "ocean-blue",
    "royal-purple",
    "emerald-green",
    "sunset-orange",
    "rose-pink",
    "midnight-navy",
    "teal-breeze",
    "crimson-red",
    "golden-amber",
    "sky-blue",
    "lavender-dream",
    "graphite-grey",
)
DEFAULT_PREFERENCES = {"appearance": "system", "theme": "royal-purple"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.update(
    MAX_CONTENT_LENGTH=MAX_MB * 1024 * 1024,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(days=365),
    JSON_SORT_KEYS=False,
)
app.register_blueprint(workspace_api)


def _secret_key() -> str:
    path = BASE_DIR / ".secret_key"
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
    key = secrets.token_hex(32)
    path.write_text(key, encoding="utf-8")
    return key


app.secret_key = _secret_key()

_frame_cache: dict[str, tuple[float, pd.DataFrame]] = {}


# --------------------------------------------------------------------------- #
# Storage helpers
# --------------------------------------------------------------------------- #

def _sid() -> str | None:
    sid = session.get("sid")
    if isinstance(sid, str) and re.fullmatch(r"[0-9a-f]{32}", sid):
        workspace = WORKSPACES / sid
        if workspace.is_dir():
            return sid
    return None


def _workspace(sid: str) -> Path:
    return WORKSPACES / sid


def _state_path(sid: str) -> Path:
    return _workspace(sid) / "state.json"


def _load_state(sid: str) -> dict | None:
    path = _state_path(sid)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_state(sid: str, state: dict) -> None:
    path = _state_path(sid)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)


# --------------------------------------------------------------------------- #
# Theme preferences
# --------------------------------------------------------------------------- #

def _pref_uid(create: bool = False) -> str | None:
    # Guest preferences are session-scoped; an authenticated build can swap
    # this for the signed-in account id without changing storage elsewhere.
    uid = session.get("pref_uid")
    if isinstance(uid, str) and re.fullmatch(r"[0-9a-f]{32}", uid):
        return uid
    if not create:
        return None
    uid = uuid.uuid4().hex
    session["pref_uid"] = uid
    session.permanent = True
    return uid


def _pref_path(uid: str) -> Path:
    return PREFERENCES / f"{uid}.json"


def _normalise_preferences(raw, fallback: dict | None = None) -> dict:
    prefs = dict(fallback or DEFAULT_PREFERENCES)
    if not isinstance(raw, dict):
        return prefs
    appearance = raw.get("appearance")
    theme = raw.get("theme")
    if appearance in APPEARANCE_MODES:
        prefs["appearance"] = appearance
    if theme in THEME_IDS:
        prefs["theme"] = theme
    return prefs


def _read_preferences() -> dict:
    prefs = dict(DEFAULT_PREFERENCES)
    saved = False
    updated = None
    uid = _pref_uid(create=False)
    if uid:
        path = _pref_path(uid)
        if path.exists():
            try:
                prefs = _normalise_preferences(json.loads(path.read_text(encoding="utf-8")))
                saved = True
                updated = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
            except (OSError, ValueError):
                prefs = dict(DEFAULT_PREFERENCES)
    return {**prefs, "saved": saved, "updated_at": updated}


def _write_preferences(prefs: dict) -> dict:
    uid = _pref_uid(create=True)
    path = _pref_path(uid)
    temp = path.with_suffix(".tmp")
    payload = {
        "appearance": prefs["appearance"],
        "theme": prefs["theme"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
    return {
        "appearance": payload["appearance"],
        "theme": payload["theme"],
        "saved": True,
        "updated_at": payload["updated_at"],
    }


def _pickle_path(sid: str) -> Path:
    return _workspace(sid) / "original.pkl"


def _load_frame(sid: str) -> pd.DataFrame:
    path = _pickle_path(sid)
    mtime = path.stat().st_mtime
    cached = _frame_cache.get(sid)
    if cached and cached[0] == mtime:
        return cached[1]
    with open(path, "rb") as handle:
        frame = pickle.load(handle)
    if len(_frame_cache) > 8:
        _frame_cache.pop(next(iter(_frame_cache)))
    _frame_cache[sid] = (mtime, frame)
    return frame


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return as_text(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, bool, int, float)):
        return value
    return as_text(value)


def _display(value) -> str:
    return as_text(value)


def _set_cell(frame: pd.DataFrame, row_id: int, column: str, value) -> None:
    try:
        frame.at[row_id, column] = value
    except (TypeError, ValueError):
        frame[column] = frame[column].astype(object)
        frame.at[row_id, column] = value


def _working_frame(sid: str, state: dict) -> pd.DataFrame:
    frame = _load_frame(sid).copy(deep=True)
    for change in state["changes"]:
        row_id, column = change["row_id"], change["column"]
        if column not in frame.columns or row_id >= len(frame):
            continue
        _set_cell(frame, row_id, column, change["new"])
    return frame


def _changes_map(state: dict) -> dict[tuple[int, str], dict]:
    return {(change["row_id"], change["column"]): change for change in state["changes"]}


def _find_issue(state: dict, issue_id: str) -> dict | None:
    for issue in state["issues"]:
        if issue["id"] == issue_id:
            return issue
    return None


def _status_after_revert(issue: dict) -> str:
    if issue.get("category") == "missing":
        return "missing"
    if issue.get("category") == "duplicate" and not issue.get("suggestion"):
        return "duplicate"
    return "pending"


# --------------------------------------------------------------------------- #
# Upload / initial analysis
# --------------------------------------------------------------------------- #

def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    names: list[str] = []
    used: set[str] = set()
    for index, column in enumerate(frame.columns):
        name = str(column).strip()
        if not name or name.lower() == "nan" or name.lower().startswith("unnamed:"):
            name = f"Column {index + 1}"
        candidate = name
        suffix = 2
        while candidate in used:
            candidate = f"{name} ({suffix})"
            suffix += 1
        used.add(candidate)
        names.append(candidate)
    frame.columns = names
    return frame.reset_index(drop=True)


def _read_upload(path: Path, extension: str) -> tuple[pd.DataFrame, list[str], list[str], str]:
    sheets: list[str] = []
    sheet_name = "Sheet1"
    raw_headers: list[str] = []

    if extension == ".csv":
        frame = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                frame = pd.read_csv(path, encoding=encoding)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        if frame is None:
            raise ValueError("The CSV file could not be read.")
        raw_headers = [str(column) for column in frame.columns]
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                first_row = next(csv.reader(handle), [])
            if first_row:
                raw_headers = [str(value) for value in first_row]
        except (OSError, UnicodeDecodeError, StopIteration):
            pass
        return _clean_columns(frame), ["CSV"], raw_headers, "CSV"

    engine_name = "xlrd" if extension == ".xls" else "openpyxl"
    workbook = pd.ExcelFile(path, engine=engine_name)
    sheets = workbook.sheet_names
    sheet_name = sheets[0]
    frame = workbook.parse(sheet_name)
    header_frame = pd.read_excel(path, header=None, nrows=1, sheet_name=sheet_name,
                                 engine=engine_name)
    if not header_frame.empty:
        raw_headers = [as_text(value) for value in header_frame.iloc[0].tolist()]
    return _clean_columns(frame), sheets, raw_headers, sheet_name


def _initial_status(issue: dict) -> str:
    if issue.get("category") == "missing":
        return "missing"
    if issue.get("category") == "duplicate" and not issue.get("suggestion"):
        return "duplicate"
    return "pending"


def _bootstrap_state(sid: str, frame: pd.DataFrame, meta: dict,
                     result: dict) -> dict:
    kinds = result["kinds"]
    issues = result["issues"]
    changes: list[dict] = []
    deleted: list[int] = []

    for issue in issues:
        kind = kinds.get(issue["column"], "text")
        if issue.get("auto") and issue.get("action") == "fix" and issue.get("suggestion"):
            row_id, column = issue["row_id"], issue["column"]
            changes.append({
                "row_id": row_id,
                "column": column,
                "old": as_text(frame.at[row_id, column]),
                "new": coerce_value(kind, issue["suggestion"]),
                "status": "auto",
                "issue_id": issue["id"],
            })
            issue["status"] = "auto"
        elif issue.get("auto") and issue.get("action") == "remove_row":
            deleted.append(issue["row_id"])
            issue["status"] = "removed"
        else:
            issue["status"] = _initial_status(issue)

    return {
        "sid": sid,
        "created": datetime.now().isoformat(timespec="seconds"),
        "meta": meta,
        "kinds": kinds,
        "labels": result["labels"],
        "profile": result["profile"],
        "issues": issues,
        "changes": changes,
        "deleted_rows": sorted(set(deleted)),
        "history": [],
    }


# --------------------------------------------------------------------------- #
# Stats / payloads
# --------------------------------------------------------------------------- #

def _compute_stats(frame: pd.DataFrame, state: dict) -> dict:
    deleted = set(state["deleted_rows"])
    active_rows = [row_id for row_id in range(len(frame)) if row_id not in deleted]
    active_set = set(active_rows)

    open_rows: set[int] = set()
    missing = needs_review = duplicates = ignored = 0
    for issue in state["issues"]:
        status = issue.get("status")
        row_id = issue.get("row_id")
        if status in OPEN_STATUSES and row_id is not None:
            open_rows.add(row_id)
        if status in OPEN_STATUSES and issue.get("category") == "missing":
            missing += 1
        if status == "pending" and (issue.get("suggestion") or issue.get("action") == "remove_row"):
            needs_review += 1
        if (status in OPEN_STATUSES and issue.get("category") == "duplicate"
                and row_id is not None and issue.get("action") == "remove_row"):
            duplicates += 1
        if status == "rejected":
            ignored += 1

    open_rows &= active_set
    changes = state["changes"]
    auto = sum(1 for change in changes if change["status"] == "auto")
    approved = sum(1 for change in changes if change["status"] == "accepted")
    manual = sum(1 for change in changes if change["status"] == "manual")

    rows = len(active_rows)
    columns = len(frame.columns)
    correct = rows - len(open_rows)

    return {
        "rows": rows,
        "columns": columns,
        "cells": rows * columns,
        "errors_detected": len(state["issues"]),
        "correct_records": correct,
        "incorrect_records": len(open_rows),
        "duplicates": duplicates,
        "missing": missing,
        "auto_corrected": auto,
        "approved": approved,
        "manual": manual,
        "corrected_total": len(changes),
        "needs_review": needs_review,
        "ignored": ignored,
        "deleted_rows": len(deleted),
        "health": round(100 * correct / max(1, rows)),
    }


def _category_counts(state: dict) -> list[dict]:
    counts: dict[str, int] = {}
    for issue in state["issues"]:
        if issue.get("status") not in OPEN_STATUSES:
            continue
        counts[issue["category"]] = counts.get(issue["category"], 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"key": key, "label": CATEGORY_LABELS.get(key, key.title()), "count": count}
            for key, count in ordered]


def _issue_payload(issue: dict) -> dict:
    status = issue.get("status", "pending")
    return {
        "id": issue["id"],
        "row": issue.get("row"),
        "row_id": issue.get("row_id"),
        "column": issue.get("column", ""),
        "category": issue.get("category"),
        "category_label": CATEGORY_LABELS.get(issue.get("category", ""), ""),
        "error": issue.get("error", ""),
        "original": issue.get("original", ""),
        "suggestion": issue.get("suggestion"),
        "confidence": issue.get("confidence"),
        "severity": issue.get("severity", "error"),
        "reason": issue.get("reason", ""),
        "action": issue.get("action", "fix"),
        "status": status,
        "status_label": STATUS_LABELS.get(status, status.title()),
    }


def _rows_payload(frame: pd.DataFrame, state: dict, page: int = 1,
                  mode: str = "all", search: str = "") -> dict:
    deleted = set(state["deleted_rows"])
    changes = _changes_map(state)
    open_by_row: dict[int, int] = {}
    open_by_cell: dict[tuple[int, str], dict] = {}
    for issue in state["issues"]:
        if issue.get("status") not in OPEN_STATUSES or issue.get("row_id") is None:
            continue
        row_id = issue["row_id"]
        open_by_row[row_id] = open_by_row.get(row_id, 0) + 1
        cell_key = (row_id, issue.get("column", ""))
        current = open_by_cell.get(cell_key)
        if current is None or (not current.get("suggestion") and issue.get("suggestion")):
            open_by_cell[cell_key] = issue

    columns = [str(column) for column in frame.columns]
    search_text = search.strip().lower()

    selected: list[int] = []
    for row_id in range(len(frame)):
        if row_id in deleted:
            continue
        if mode == "problems" and row_id not in open_by_row:
            continue
        if mode == "review":
            has_review = any(
                key[0] == row_id
                and (issue.get("suggestion") or issue.get("action") == "remove_row")
                for key, issue in open_by_cell.items()
            )
            if not has_review:
                continue
        if mode == "corrected" and not any(key[0] == row_id for key in changes):
            continue
        if search_text:
            haystack = " ".join(_display(frame.at[row_id, column]) for column in columns).lower()
            haystack += " " + " ".join(
                str(change["old"]).lower() for key, change in changes.items() if key[0] == row_id
            )
            if search_text not in haystack:
                continue
        selected.append(row_id)

    total = len(selected)
    pages = max(1, math.ceil(total / PAGE_SIZE))
    page = max(1, min(page, pages))
    start = (page - 1) * PAGE_SIZE
    slice_rows = selected[start:start + PAGE_SIZE]

    items = []
    for row_id in slice_rows:
        cells = {}
        for column in columns:
            change = changes.get((row_id, column))
            issue = open_by_cell.get((row_id, column))
            value = frame.at[row_id, column]
            cells[column] = {
                "value": _display(value),
                "old": change["old"] if change else None,
                "status": change["status"] if change else None,
                "issue_id": issue["id"] if issue else None,
                "suggestion": issue.get("suggestion") if issue else None,
                "confidence": issue.get("confidence") if issue else None,
                "error": issue.get("error") if issue else None,
                "action": issue.get("action") if issue else None,
            }
        items.append({
            "row_id": row_id,
            "excel_row": row_id + 2,
            "open_issues": open_by_row.get(row_id, 0),
            "corrected": sum(1 for key in changes if key[0] == row_id),
            "cells": cells,
        })

    return {
        "items": items,
        "page": page,
        "pages": pages,
        "total": total,
        "size": PAGE_SIZE,
        "mode": mode,
        "search": search,
    }


def _payload(sid: str, page: int = 1, mode: str = "all", search: str = "") -> dict:
    state = _load_state(sid)
    frame = _load_frame(sid)
    working = _working_frame(sid, state)

    stats = _compute_stats(working, state)
    profile = []
    open_counts: dict[str, int] = {}
    for issue in state["issues"]:
        if issue.get("status") in OPEN_STATUSES:
            open_counts[issue["column"]] = open_counts.get(issue["column"], 0) + 1
    for entry in state["profile"]:
        item = dict(entry)
        item["issues"] = open_counts.get(entry["name"], 0)
        item["label"] = state["labels"].get(entry["name"], entry["label"])
        item["kind"] = state["kinds"].get(entry["name"], entry["kind"])
        profile.append(item)

    meta = dict(state["meta"])
    meta["active_rows"] = stats["rows"]

    return {
        "empty": False,
        "meta": meta,
        "stats": stats,
        "categories": _category_counts(state),
        "profile": profile,
        "columns": [{"name": column, "kind": state["kinds"].get(column, "text"),
                     "label": state["labels"].get(column, "Text")}
                    for column in frame.columns],
        "issues": [_issue_payload(issue) for issue in state["issues"]],
        "rows": _rows_payload(working, state, page, mode, search),
        "can_undo": bool(state["history"]),
        "last_action": state["history"][-1]["label"] if state["history"] else "",
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.route("/")
def index():
    return render_template(
        "workspace.html",
        max_mb=MAX_MB,
        initial_prefs=_read_preferences(),
    )


@app.route("/cleaner")
def cleaner():
    return render_template(
        "index.html",
        max_mb=MAX_MB,
        initial_prefs=_read_preferences(),
    )


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/user/preferences")
def api_get_preferences():
    return jsonify(_read_preferences())


@app.post("/api/user/preferences")
def api_set_preferences():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Send a JSON object with appearance and theme."), 400

    appearance = body.get("appearance")
    theme = body.get("theme")
    if appearance is None and theme is None:
        return jsonify(error="No preferences supplied."), 400
    if appearance is not None and appearance not in APPEARANCE_MODES:
        return jsonify(
            error="Unknown appearance mode.",
            allowed=list(APPEARANCE_MODES),
        ), 400
    if theme is not None and theme not in THEME_IDS:
        return jsonify(error="Unknown colour theme.", allowed=list(THEME_IDS)), 400

    current = _read_preferences()
    prefs = {
        "appearance": current["appearance"],
        "theme": current["theme"],
    }
    if appearance is not None:
        prefs["appearance"] = appearance
    if theme is not None:
        prefs["theme"] = theme

    try:
        saved = _write_preferences(prefs)
    except OSError:
        return jsonify(error="Could not save preferences on this computer."), 500
    return jsonify(saved)


@app.post("/api/analyze")
def api_analyze():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify(error="Choose an Excel or CSV file first."), 400

    filename = os.path.basename(upload.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify(error="Unsupported file type. Use .xlsx, .xls or .csv."), 400

    sid = uuid.uuid4().hex
    workspace = _workspace(sid)
    workspace.mkdir(parents=True, exist_ok=True)
    raw_path = workspace / f"original{extension}"

    try:
        upload.save(raw_path)
        if raw_path.stat().st_size == 0:
            raise ValueError("The file is empty.")
        frame, sheets, raw_headers, sheet_name = _read_upload(raw_path, extension)
        if frame.empty or len(frame.columns) == 0:
            raise ValueError("No data rows were found in the workbook.")
        result = analyze(frame, raw_headers)
    except Exception as exc:  # noqa: BLE001 - surface a readable message
        shutil.rmtree(workspace, ignore_errors=True)
        return jsonify(error=f"Could not read this file: {exc}"), 400

    meta = {
        "file_name": filename,
        "size": raw_path.stat().st_size,
        "extension": extension,
        "sheet": sheet_name,
        "sheets": sheets,
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "truncated": result["truncated"],
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    state = _bootstrap_state(sid, frame, meta, result)
    with open(_pickle_path(sid), "wb") as handle:
        pickle.dump(frame, handle, protocol=4)
    _save_state(sid, state)
    session["sid"] = sid

    return jsonify(_payload(sid, 1, "all", ""))


@app.get("/api/state")
def api_state():
    sid = _sid()
    if not sid:
        return jsonify(empty=True)
    page = request.args.get("page", 1, type=int) or 1
    mode = request.args.get("mode", "all")
    search = request.args.get("search", "")
    return jsonify(_payload(sid, page, mode, search))


@app.get("/api/rows")
def api_rows():
    sid = _sid()
    if not sid:
        return jsonify(empty=True)
    state = _load_state(sid)
    frame = _load_frame(sid)
    working = _working_frame(sid, state)
    page = request.args.get("page", 1, type=int) or 1
    mode = request.args.get("mode", "all")
    search = request.args.get("search", "")
    return jsonify(rows=_rows_payload(working, state, page, mode, search))


def _mutation_context() -> tuple[str, dict, dict]:
    sid = _sid()
    if not sid:
        raise LookupError("No active session.")
    state = _load_state(sid)
    if state is None:
        raise LookupError("No active session.")
    body = request.get_json(silent=True) or {}
    return sid, state, body


def _respond(sid: str, body: dict):
    page = int(body.get("page") or 1)
    mode = body.get("mode") or "all"
    search = body.get("search") or ""
    return jsonify(_payload(sid, page, mode, search))


def _history_entry(label: str, ops: list | None = None, deletes: list | None = None,
                   statuses: list | None = None) -> dict:
    return {
        "label": label,
        "at": datetime.now().isoformat(timespec="seconds"),
        "ops": ops or [],
        "deletes": deletes or [],
        "statuses": statuses or [],
    }


@app.post("/api/accept")
def api_accept():
    try:
        sid, state, body = _mutation_context()
    except LookupError as exc:
        return jsonify(error=str(exc)), 400

    ids = body.get("ids") or []
    frame = _load_frame(sid)
    working = _working_frame(sid, state)
    kinds = state["kinds"]
    changes = _changes_map(state)
    ops: list[dict] = []
    deletes: list[int] = []
    statuses: list[dict] = []
    applied = 0

    for issue_id in ids:
        issue = _find_issue(state, issue_id)
        if not issue or issue.get("status") not in OPEN_STATUSES:
            continue
        statuses.append({"id": issue["id"], "status": issue.get("status")})
        if issue.get("action") == "remove_row" and issue.get("row_id") is not None:
            row_id = issue["row_id"]
            if row_id not in state["deleted_rows"]:
                state["deleted_rows"].append(row_id)
                deletes.append(row_id)
            issue["status"] = "removed"
            applied += 1
            continue
        if not issue.get("suggestion"):
            continue
        row_id, column = issue["row_id"], issue["column"]
        kind = kinds.get(column, "text")
        new_value = coerce_value(kind, issue["suggestion"])
        before_change = changes.get((row_id, column))
        ops.append({
            "row_id": row_id,
            "column": column,
            "had_change": before_change is not None,
            "before": before_change["new"] if before_change else _json_value(frame.at[row_id, column]),
            "restore_change": before_change,
        })
        change = {
            "row_id": row_id,
            "column": column,
            "old": before_change["old"] if before_change else _display(frame.at[row_id, column]),
            "new": new_value,
            "status": "accepted",
            "issue_id": issue["id"],
        }
        state["changes"] = [
            c for c in state["changes"] if not (c["row_id"] == row_id and c["column"] == column)
        ]
        state["changes"].append(change)
        _set_cell(working, row_id, column, new_value)
        issue["status"] = "accepted"
        changes[(row_id, column)] = change
        applied += 1

    if applied:
        state["deleted_rows"] = sorted(set(state["deleted_rows"]))
        state["history"].append(_history_entry(
            f"Approved {applied} correction(s)", ops, deletes, statuses))
        _save_state(sid, state)
    return _respond(sid, body)


@app.post("/api/accept_all")
def api_accept_all():
    try:
        sid, state, body = _mutation_context()
    except LookupError as exc:
        return jsonify(error=str(exc)), 400

    frame = _load_frame(sid)
    working = _working_frame(sid, state)
    kinds = state["kinds"]
    changes = _changes_map(state)
    ops: list[dict] = []
    statuses: list[dict] = []
    applied = 0

    for issue in state["issues"]:
        if issue.get("status") != "pending" or not issue.get("suggestion"):
            continue
        if issue.get("action") != "fix" or issue.get("row_id") is None:
            continue
        row_id, column = issue["row_id"], issue["column"]
        new_value = coerce_value(kinds.get(column, "text"), issue["suggestion"])
        before_change = changes.get((row_id, column))
        statuses.append({"id": issue["id"], "status": issue.get("status")})
        ops.append({
            "row_id": row_id,
            "column": column,
            "had_change": before_change is not None,
            "before": before_change["new"] if before_change else _json_value(frame.at[row_id, column]),
            "restore_change": before_change,
        })
        change = {
            "row_id": row_id,
            "column": column,
            "old": before_change["old"] if before_change else _display(frame.at[row_id, column]),
            "new": new_value,
            "status": "accepted",
            "issue_id": issue["id"],
        }
        state["changes"] = [
            c for c in state["changes"] if not (c["row_id"] == row_id and c["column"] == column)
        ]
        state["changes"].append(change)
        issue["status"] = "accepted"
        changes[(row_id, column)] = change
        applied += 1

    if applied:
        state["history"].append(_history_entry(
            f"Approved all {applied} remaining suggestions", ops, [], statuses))
        _save_state(sid, state)
    return _respond(sid, body)


@app.post("/api/reject")
def api_reject():
    try:
        sid, state, body = _mutation_context()
    except LookupError as exc:
        return jsonify(error=str(exc)), 400

    ids = set(body.get("ids") or [])
    frame = _load_frame(sid)
    ops: list[dict] = []
    statuses: list[dict] = []
    rejected = 0

    for issue in state["issues"]:
        if issue["id"] not in ids or issue.get("status") not in OPEN_STATUSES:
            continue
        statuses.append({"id": issue["id"], "status": issue.get("status")})
        row_id, column = issue.get("row_id"), issue.get("column")
        before_change = None
        if row_id is not None:
            for change in list(state["changes"]):
                if (change["row_id"] == row_id and change["column"] == column
                        and change.get("issue_id") == issue["id"]):
                    before_change = change
                    state["changes"].remove(change)
                    ops.append({
                        "row_id": row_id,
                        "column": column,
                        "had_change": True,
                        "before": change["new"],
                        "restore_change": change,
                    })
                    break
        issue["status"] = "rejected"
        rejected += 1

    if rejected:
        state["history"].append(_history_entry(
            f"Ignored {rejected} suggestion(s)", ops, [], statuses))
        _save_state(sid, state)
    return _respond(sid, body)


@app.post("/api/edit")
def api_edit():
    try:
        sid, state, body = _mutation_context()
    except LookupError as exc:
        return jsonify(error=str(exc)), 400

    frame = _load_frame(sid)
    try:
        row_id = int(body.get("row_id"))
        column = str(body.get("column"))
        raw_value = body.get("value", "")
    except (TypeError, ValueError):
        return jsonify(error="Invalid edit request."), 400

    if column not in frame.columns or not (0 <= row_id < len(frame)):
        return jsonify(error="Unknown cell."), 400
    if row_id in state["deleted_rows"]:
        return jsonify(error="This row has been removed."), 400

    kind = state["kinds"].get(column, "text")
    text_value = str(raw_value).strip()
    if len(text_value) > 2000:
        return jsonify(error="That value is too long for a spreadsheet cell."), 400

    if text_value == "":
        new_value = None
    elif kind == "numeric":
        number = engine.parse_number(text_value)
        new_value = (int(number) if number is not None and float(number).is_integer()
                     else number if number is not None else text_value)
    elif kind == "date":
        parsed = engine.parse_date_value(text_value)
        if parsed and not parsed["ambiguous"]:
            stamp = parsed["stamp"]
            new_value = (stamp.strftime("%Y-%m-%d %H:%M:%S") if parsed["has_time"]
                         else stamp.strftime("%Y-%m-%d"))
        else:
            new_value = text_value
    else:
        new_value = text_value

    original_display = _display(frame.at[row_id, column])
    changes = _changes_map(state)
    current = changes.get((row_id, column))
    current_display = _display(current["new"]) if current else original_display

    if _display(new_value) == current_display:
        return _respond(sid, body)

    cell_issues = [issue for issue in state["issues"]
                   if issue.get("row_id") == row_id and issue.get("column") == column
                   and issue.get("status") in OPEN_STATUSES]
    before_statuses = [{"id": issue["id"], "status": issue["status"]} for issue in cell_issues]

    if _display(new_value) == original_display:
        state["changes"] = [c for c in state["changes"]
                            if not (c["row_id"] == row_id and c["column"] == column)]
        for issue in cell_issues:
            issue["status"] = _status_after_revert(issue)
    else:
        change = {
            "row_id": row_id,
            "column": column,
            "old": original_display,
            "new": new_value,
            "status": "manual",
            "issue_id": cell_issues[0]["id"] if cell_issues else None,
        }
        state["changes"] = [c for c in state["changes"]
                            if not (c["row_id"] == row_id and c["column"] == column)]
        state["changes"].append(change)
        for issue in cell_issues:
            issue["status"] = "manual"

    state["history"].append(_history_entry(
        f"Manual edit: {column} (row {row_id + 2})",
        [{
            "row_id": row_id,
            "column": column,
            "had_change": current is not None,
            "before": current["new"] if current else _json_value(frame.at[row_id, column]),
            "restore_change": current,
        }],
        [],
        before_statuses,
    ))
    _save_state(sid, state)
    return _respond(sid, body)


@app.post("/api/revert")
def api_revert():
    try:
        sid, state, body = _mutation_context()
    except LookupError as exc:
        return jsonify(error=str(exc)), 400

    issue = _find_issue(state, str(body.get("id", "")))
    if not issue:
        return jsonify(error="Correction not found."), 404

    row_id, column = issue.get("row_id"), issue.get("column")
    if row_id is None:
        issue["status"] = _status_after_revert(issue)
        _save_state(sid, state)
        return _respond(sid, body)

    frame = _load_frame(sid)
    existing = next((change for change in state["changes"]
                     if change["row_id"] == row_id and change["column"] == column), None)
    ops = []
    if existing:
        state["changes"].remove(existing)
        ops.append({
            "row_id": row_id,
            "column": column,
            "had_change": True,
            "before": existing["new"],
            "restore_change": existing,
        })
    previous_status = issue.get("status")
    issue["status"] = _status_after_revert(issue)
    state["history"].append(_history_entry(
        f"Reverted correction: {column or 'row'} (row {row_id + 2})", ops, [],
        [{"id": issue["id"], "status": previous_status}]))
    _save_state(sid, state)
    return _respond(sid, body)


@app.post("/api/remove_row")
def api_remove_row():
    try:
        sid, state, body = _mutation_context()
    except LookupError as exc:
        return jsonify(error=str(exc)), 400

    try:
        row_id = int(body.get("row_id"))
    except (TypeError, ValueError):
        return jsonify(error="Invalid row."), 400

    removable = any(
        issue.get("row_id") == row_id and issue.get("action") == "remove_row"
        for issue in state["issues"]
    )
    if not removable:
        return jsonify(error="This row cannot be removed automatically."), 400

    if row_id not in state["deleted_rows"]:
        state["deleted_rows"].append(row_id)
    statuses = []
    for issue in state["issues"]:
        if issue.get("row_id") == row_id and issue.get("status") in OPEN_STATUSES:
            statuses.append({"id": issue["id"], "status": issue["status"]})
            issue["status"] = "removed"
    state["deleted_rows"] = sorted(set(state["deleted_rows"]))
    state["history"].append(_history_entry(
        f"Removed duplicate row {row_id + 2}", [], [row_id], statuses))
    _save_state(sid, state)
    return _respond(sid, body)


@app.post("/api/undo")
def api_undo():
    try:
        sid, state, body = _mutation_context()
    except LookupError as exc:
        return jsonify(error=str(exc)), 400

    if not state["history"]:
        return jsonify(error="Nothing to undo."), 400

    entry = state["history"].pop()
    for op in reversed(entry["ops"]):
        row_id, column = op["row_id"], op["column"]
        state["changes"] = [c for c in state["changes"]
                            if not (c["row_id"] == row_id and c["column"] == column)]
        if op.get("had_change"):
            if op.get("restore_change"):
                state["changes"].append(op["restore_change"])
            else:
                state["changes"].append({
                    "row_id": row_id,
                    "column": column,
                    "old": _display(_load_frame(sid).at[row_id, column]),
                    "new": op["before"],
                    "status": "accepted",
                    "issue_id": None,
                })
    for row_id in entry["deletes"]:
        if row_id in state["deleted_rows"]:
            state["deleted_rows"].remove(row_id)
    for status in entry["statuses"]:
        issue = _find_issue(state, status["id"])
        if issue:
            issue["status"] = status["status"]
    _save_state(sid, state)
    return _respond(sid, body)


@app.post("/api/reset")
def api_reset():
    try:
        sid, state, body = _mutation_context()
    except LookupError as exc:
        return jsonify(error=str(exc)), 400

    frame = _load_frame(sid)
    for issue in state["issues"]:
        kind = state["kinds"].get(issue["column"], "text")
        if issue.get("auto") and issue.get("action") == "fix" and issue.get("suggestion"):
            issue["status"] = "auto"
        elif issue.get("auto") and issue.get("action") == "remove_row":
            issue["status"] = "removed"
        else:
            issue["status"] = _initial_status(issue)

    state["changes"] = []
    state["deleted_rows"] = []
    state["history"] = []

    for issue in state["issues"]:
        if issue.get("status") != "auto" or issue.get("row_id") is None:
            continue
        kind = state["kinds"].get(issue["column"], "text")
        state["changes"].append({
            "row_id": issue["row_id"],
            "column": issue["column"],
            "old": as_text(frame.at[issue["row_id"], issue["column"]]),
            "new": coerce_value(kind, issue["suggestion"]),
            "status": "auto",
            "issue_id": issue["id"],
        })
    for issue in state["issues"]:
        if issue.get("status") == "removed" and issue.get("row_id") is not None:
            state["deleted_rows"].append(issue["row_id"])
    state["deleted_rows"] = sorted(set(state["deleted_rows"]))
    _save_state(sid, state)
    return _respond(sid, body)


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #

def _download_stem(state: dict) -> str:
    name = Path(state["meta"]["file_name"]).stem
    name = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip() or "workbook"
    return name


@app.get("/api/download/cleaned")
def download_cleaned():
    sid = _sid()
    if not sid:
        return "No active session.", 404
    state = _load_state(sid)
    working = _working_frame(sid, state)

    # Deleted rows are excluded, but row order is preserved.
    kinds = state["kinds"]
    changes = {(change["row_id"], change["column"]): change for change in state["changes"]}
    deleted = set(state["deleted_rows"])
    payload = build_cleaned_workbook(working, deleted, changes, kinds,
                                     state["meta"].get("sheet") or "Cleaned data")
    filename = f"cleaned_{_download_stem(state)}.xlsx"
    return send_file(
        io.BytesIO(payload),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename,
    )


@app.get("/api/download/report")
def download_report():
    sid = _sid()
    if not sid:
        return "No active session.", 404
    state = _load_state(sid)
    working = _working_frame(sid, state)
    stats = _compute_stats(working, state)

    open_counts: dict[str, int] = {}
    for issue in state["issues"]:
        if issue.get("status") in OPEN_STATUSES:
            open_counts[issue["column"]] = open_counts.get(issue["column"], 0) + 1
    profile = []
    for entry in state["profile"]:
        item = dict(entry)
        item["issues"] = open_counts.get(entry["name"], 0)
        profile.append(item)

    payload = build_report_workbook(state["issues"], stats, state["meta"], profile)
    filename = f"datafix_report_{_download_stem(state)}.xlsx"
    return send_file(
        io.BytesIO(payload),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename,
    )


@app.get("/api/download/original")
def download_original():
    sid = _sid()
    if not sid:
        return "No active session.", 404
    state = _load_state(sid)
    extension = state["meta"]["extension"]
    path = _workspace(sid) / f"original{extension}"
    if not path.exists():
        return "Original file is missing.", 404
    return send_file(path, as_attachment=True,
                     download_name=state["meta"]["file_name"])


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f"The file exceeds the {MAX_MB} MB limit."), 413


@app.errorhandler(500)
def server_error(error):  # pragma: no cover - safety net
    app.logger.exception("Unexpected error: %s", error)
    return jsonify(error="Something went wrong while processing the file."), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("DATAFIX_DEBUG") == "1",
    )

==============================================================================
FILE: datafix/__init__.py
==============================================================================
"""DataFix workspace package: permanent workbooks, storage, search and exports."""

==============================================================================
FILE: datafix/api.py
==============================================================================
"""REST API for the permanent workbook workspace.

Every route is JSON except downloads and the print view. Errors are returned
as ``{"ok": false, "error": "..."}`` with a matching HTTP status.
"""

from __future__ import annotations

import json
import os
import subprocess
from functools import wraps
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, send_file

from . import config, exporters, workbooks

bp = Blueprint("workspace", __name__, url_prefix="/api/workspace")


def _payload() -> dict:
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else {}


def _json_arg(name: str):
    raw = request.args.get(name)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"The {name} parameter is not valid JSON.") from exc


def _int_arg(name: str, default: int = 0) -> int:
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"The {name} parameter must be a number.") from exc


def api(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, tuple):
                data, status = result
            else:
                data, status = result, 200
            payload = {"ok": True}
            if isinstance(data, dict):
                payload.update(data)
            elif data is not None:
                payload["data"] = data
            return jsonify(payload), status
        except LookupError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (ValueError, workbooks.OpError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except PermissionError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 403
        except Exception as exc:  # noqa: BLE001 - never leak internals to the browser
            current_app.logger.exception("Workspace API failure: %s", exc)
            return jsonify({
                "ok": False,
                "error": "The server could not complete that action. Your saved work was not changed.",
            }), 500
    return wrapper


# --------------------------------------------------------------------------- #
# Storage and settings
# --------------------------------------------------------------------------- #

def _limits() -> dict:
    return {
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "max_cells_per_sheet": config.MAX_CELLS_PER_SHEET,
        "version_retention": config.get_version_retention(),
        "extensions": sorted(config.ALLOWED_EXTENSIONS),
        "export_formats": list(config.EXPORT_FORMATS),
    }


@bp.get("/state")
@api
def state():
    return {
        "storage": config.describe_storage(),
        "limits": _limits(),
        "formats": exporters.limitations(),
        "workbooks": workbooks.list_workbooks("all"),
        "starred": workbooks.list_workbooks("starred"),
        "archived": workbooks.list_workbooks("archived"),
    }


def _workspace_numbers(reports: list[dict]) -> dict:
    books = workbooks.list_workbooks("all")
    archived = workbooks.list_workbooks("archived")
    return {
        "files": len(books),
        "archived": len(archived),
        "records": sum(int(book.get("record_count") or 0) for book in books),
        "sheets": sum(int(book.get("sheet_count") or 0) for book in books),
        "issues": sum(int(report.get("total") or 0) for report in reports),
        "corrected": sum(int(report.get("applied") or 0) for report in reports),
        "rejected": sum(int(report.get("rejected") or 0) for report in reports),
        "analysed": len(reports),
    }


@bp.get("/dashboard")
@api
def dashboard():
    reports = workbooks.analysis_reports()
    books = workbooks.list_workbooks("all")
    return {
        "totals": _workspace_numbers(reports),
        "recent": books[:8],
        "unanalysed": [book for book in books if not book.get("analysis_at")][:8],
        "reports": reports[:8],
        "storage": config.describe_storage(),
    }


@bp.get("/reports")
@api
def reports():
    items = workbooks.analysis_reports()
    return {"reports": items, "totals": _workspace_numbers(items)}


@bp.post("/storage")
@api
def set_storage():
    body = _payload()
    if body.get("reset"):
        root = config.reset_storage_root()
        return {"storage": config.describe_storage(), "root": str(root)}
    root, error = config.set_storage_root(str(body.get("root") or ""))
    if root is None:
        raise ValueError(error or "That folder cannot be used.")
    return {"storage": config.describe_storage(), "root": str(root)}


@bp.post("/storage/retention")
@api
def set_retention():
    try:
        value = int(_payload().get("value") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("Keep between 5 and 1000 versions.") from exc
    return {"version_retention": config.set_version_retention(value)}


@bp.post("/dialog")
@api
def native_dialog():
    kind = str(_payload().get("kind") or "file").lower()
    if kind not in ("file", "folder"):
        raise ValueError("Choose a file or folder dialog.")
    path = _native_pick(kind)
    return {"path": path, "cancelled": path == ""}


def _native_pick(kind: str) -> str:
    if os.name != "nt":
        raise ValueError("The desktop picker is available on Windows only. Type the path instead.")
    if kind == "folder":
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
            "$d.Description='Choose a DataFix storage folder';"
            "if($d.ShowDialog() -eq 'OK'){[Console]::Out.Write($d.SelectedPath)}"
        )
    else:
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d=New-Object System.Windows.Forms.OpenFileDialog;"
            "$d.Title='Choose a spreadsheet for DataFix';"
            "$d.Filter='Spreadsheets|*.xlsx;*.xlsm;*.xls;*.csv;*.ods|All files|*.*';"
            "if($d.ShowDialog() -eq 'OK'){[Console]::Out.Write($d.FileName)}"
        )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", script],
            capture_output=True, text=True, timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"The picker could not open: {exc}") from exc
    return (result.stdout or "").strip()


# --------------------------------------------------------------------------- #
# Workbooks
# --------------------------------------------------------------------------- #

@bp.get("/workbooks")
@api
def list_workbooks():
    scope = request.args.get("scope", "all")
    query = request.args.get("query", "")
    if scope not in ("all", "recent", "starred", "archived"):
        scope = "all"
    return {
        "workbooks": workbooks.list_workbooks(scope, query),
        "counts": {
            "all": len(workbooks.list_workbooks("all")),
            "starred": len(workbooks.list_workbooks("starred")),
            "archived": len(workbooks.list_workbooks("archived")),
        },
    }


@bp.post("/workbooks")
@api
def create_workbook():
    name = str(_payload().get("name") or "").strip()
    return {"workbook": workbooks.create_blank(name or None)}, 201


@bp.get("/workbooks/<wb_id>")
@api
def get_workbook(wb_id):
    workbook = workbooks.require_workbook(wb_id)
    workbooks.touch_opened(wb_id)
    return {"workbook": workbook}


@bp.patch("/workbooks/<wb_id>")
@api
def update_workbook(wb_id):
    body = _payload()
    workbooks.require_workbook(wb_id)
    if "name" in body:
        workbooks.rename_workbook(wb_id, str(body.get("name") or ""))
    if "starred" in body or "archived" in body:
        workbooks.set_flags(
            wb_id,
            starred=bool(body["starred"]) if "starred" in body else None,
            archived=bool(body["archived"]) if "archived" in body else None,
        )
    return {"workbook": workbooks.require_workbook(wb_id)}


@bp.delete("/workbooks/<wb_id>")
@api
def delete_workbook(wb_id):
    return {"deleted": workbooks.delete_workbook(wb_id)}


@bp.post("/workbooks/<wb_id>/duplicate")
@api
def duplicate(wb_id):
    name = str(_payload().get("name") or "").strip() or None
    return {"workbook": workbooks.duplicate_workbook(wb_id, name)}, 201


@bp.post("/workbooks/<wb_id>/save-as")
@api
def save_as(wb_id):
    body = _payload()
    copy = workbooks.duplicate_workbook(wb_id, str(body.get("name") or "").strip() or None)
    response = {"workbook": copy}
    if body.get("download"):
        record = exporters.export(copy["id"], "xlsx")
        response["export"] = record
    return response, 201


@bp.get("/workbooks/<wb_id>/grid")
@api
def grid(wb_id):
    data = workbooks.read_grid(
        wb_id,
        sheet_id=request.args.get("sheet_id") or None,
        start_row=_int_arg("start_row", 1),
        rows=_int_arg("rows", 100),
        start_col=_int_arg("start_col", 1),
        cols=_int_arg("cols", 30),
        filters=_json_arg("filters"),
        sort=_json_arg("sort"),
        search=request.args.get("search", ""),
    )
    return data


@bp.post("/workbooks/<wb_id>/ops")
@api
def ops(wb_id):
    body = _payload()
    meta = workbooks.apply_ops(
        wb_id, body.get("ops") or [],
        label=str(body.get("label") or "Edit")[:80],
        version=bool(body.get("version")),
    )
    return meta


@bp.post("/workbooks/<wb_id>/undo")
@api
def undo(wb_id):
    return {"result": workbooks.undo(wb_id)}


@bp.post("/workbooks/<wb_id>/redo")
@api
def redo(wb_id):
    return {"result": workbooks.redo(wb_id)}


@bp.post("/workbooks/<wb_id>/recalculate")
@api
def recalculate(wb_id):
    return workbooks.recalculate(wb_id)


@bp.get("/workbooks/<wb_id>/versions")
@api
def versions(wb_id):
    workbooks.require_workbook(wb_id)
    items, undoable = workbooks.list_versions(wb_id)
    history = _read_batches(wb_id)
    return {
        "versions": items, "pending_edits": undoable, "history": history,
        **workbooks.undo_state(wb_id),
    }


def _read_batches(wb_id: str) -> list[dict]:
    from . import db
    with db.read() as conn:
        rows = conn.execute(
            "SELECT id, label, undone, created_at FROM batches WHERE workbook_id=?"
            " ORDER BY id DESC LIMIT 100", (wb_id,)
        ).fetchall()
    return [dict(row) for row in rows]


@bp.post("/workbooks/<wb_id>/versions")
@api
def create_checkpoint(wb_id):
    from . import db
    label = str(_payload().get("label") or "Manual checkpoint")[:80]
    with db.transaction() as conn:
        record = workbooks.create_version(conn, wb_id, label, force=True)
    if not record:
        raise LookupError("Workbook not found.")
    return {"version": record}, 201


@bp.post("/workbooks/<wb_id>/versions/<int:version_id>/restore")
@api
def restore(wb_id, version_id):
    workbook = workbooks.restore_version(wb_id, version_id)
    return {"workbook": workbook}


@bp.get("/workbooks/<wb_id>/search")
@api
def search_workbook(wb_id):
    query = request.args.get("q", "")
    if not query.strip():
        return {"results": []}
    results = workbooks.search_in_workbook(
        wb_id, query,
        sheet_id=request.args.get("sheet_id") or None,
        column=_int_arg("column", 0),
        limit=min(500, max(1, _int_arg("limit", 200))),
    )
    return {"results": results, "query": query}


@bp.get("/search")
@api
def search_everywhere():
    query = request.args.get("q", "").strip()
    if not query:
        return {"results": [], "workbooks": []}
    return _global_search(query)


def _global_search(query: str) -> dict:
    from . import db
    tokens = [token for token in query.split() if token][:6]
    clauses, params = [], []
    for token in tokens:
        clauses.append("(c.value LIKE ? OR c.formula LIKE ?)")
        params.extend([f"%{token}%", f"%{token}%"])
    sql = (
        "SELECT c.row, c.col, c.value, c.formula, s.id AS sheet_id, s.name AS sheet_name,"
        " w.id AS workbook_id, w.name AS workbook_name, w.archived"
        " FROM cells c JOIN sheets s ON s.id=c.sheet_id"
        " JOIN workbooks w ON w.id=s.workbook_id"
        " WHERE " + " AND ".join(clauses) +
        " AND w.archived=0 ORDER BY w.opened_at DESC, s.position, c.row, c.col LIMIT 300"
    )
    with db.read() as conn:
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
        headers: dict[int, dict] = {}
        for result in rows:
            sheet_id = result["sheet_id"]
            if sheet_id not in headers:
                header_rows = conn.execute(
                    "SELECT col, value FROM cells WHERE sheet_id=? AND row=1", (sheet_id,)
                ).fetchall()
                headers[sheet_id] = {item["col"]: item["value"] for item in header_rows}
        name_rows = conn.execute(
            "SELECT id, name, archived FROM workbooks WHERE archived=0 AND (name LIKE ?"
            " OR original_name LIKE ?) ORDER BY opened_at DESC LIMIT 50",
            (f"%{query}%", f"%{query}%"),
        ).fetchall()
    for result in rows:
        result["header"] = headers.get(result["sheet_id"], {}).get(
            result["col"]) or f"Column {result['col']}"
        result["ref"] = f"{_column_letter(result['col'])}{result['row']}"
    return {
        "results": rows,
        "workbooks": [dict(row) for row in name_rows],
        "query": query,
        "tokens": tokens,
    }


def _column_letter(col: int) -> str:
    name = ""
    value = int(col)
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        name = chr(65 + remainder) + name
    return name


# --------------------------------------------------------------------------- #
# Analysis (cleaning engine)
# --------------------------------------------------------------------------- #

def _run_analysis(wb_id, sheet_id=None):
    import pandas as pd
    import engine

    workbook = workbooks.require_workbook(wb_id)
    sheet_id = sheet_id or workbook["sheets"][0]["id"]
    sheet = next((item for item in workbook["sheets"] if int(item["id"]) == int(sheet_id)), None)
    if sheet is None:
        raise LookupError("Worksheet not found.")
    view = workbooks.view_data(wb_id, sheet["id"], include_hidden=True)
    rows = [row for row in view["rows"] if row != 1]
    cols = view["columns"]
    header_cells = [view["cells"].get((1, col)) for col in cols]
    raw_headers = [str(cell["v"]) if cell and cell.get("v") is not None else "" for cell in header_cells]
    names, seen = [], {}
    for index, header in enumerate(raw_headers, start=1):
        base = header.strip() or f"Column {index}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        names.append(base if count == 1 else f"{base} ({count})")
    data = []
    for row in rows:
        line = []
        for col in cols:
            cell = view["cells"].get((row, col))
            line.append("" if not cell or cell.get("v") is None else cell["v"])
        data.append(line)
    frame = pd.DataFrame(data, columns=names)
    result = engine.analyze(frame, raw_headers)
    name_to_col = {name: index + 1 for index, name in enumerate(names)}
    for issue in result.get("issues", []):
        row_id = issue.get("row_id")
        issue["sheet_row"] = None if row_id is None else int(row_id) + 2
        issue["col"] = name_to_col.get(issue.get("column"), 0)
    payload = {
        **result,
        "sheet_id": sheet["id"],
        "sheet_name": sheet["name"],
        "analyzed_rows": len(rows),
        "columns": [{"name": name, "col": cols[index]}
                    for index, name in enumerate(names)],
    }
    workbooks.save_analysis(wb_id, sheet["id"], payload)
    return payload


@bp.post("/workbooks/<wb_id>/analysis")
@api
def analyze(wb_id):
    payload = _run_analysis(wb_id, _payload().get("sheet_id"))
    return {"analysis": payload}


@bp.get("/workbooks/<wb_id>/analysis")
@api
def get_analysis(wb_id):
    workbooks.require_workbook(wb_id)
    return {"analysis": workbooks.get_analysis(wb_id)}


@bp.delete("/workbooks/<wb_id>/analysis")
@api
def clear_analysis(wb_id):
    workbooks.require_workbook(wb_id)
    workbooks.clear_analysis(wb_id)
    return {"cleared": True}


@bp.post("/workbooks/<wb_id>/analysis/apply")
@api
def apply_analysis(wb_id):
    body = _payload()
    corrections = body.get("corrections")
    if corrections is not None:
        if not isinstance(corrections, list) or not corrections:
            raise ValueError("Choose at least one correction to apply.")
        if len(corrections) > 400:
            raise ValueError("Apply at most 400 corrections at a time.")
    else:
        corrections = [body]
    ops, cells = [], []
    for item in corrections:
        if not isinstance(item, dict):
            raise ValueError("One of the corrections is not valid.")
        try:
            sheet_id = int(item.get("sheet_id") or body.get("sheet_id") or 0)
            row = int(item["row"])
            col = int(item["col"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Choose a detected cell to correct.") from exc
        if not sheet_id:
            raise ValueError("Choose the worksheet that holds the correction.")
        ops.append({
            "op": "set_cell", "sheet": sheet_id, "row": row, "col": col,
            "value": item.get("value"),
        })
        cells.append((row, col))
    meta = workbooks.apply_ops(wb_id, ops, label="Cleaning correction")
    workbooks.set_analysis_status(wb_id, "applied", cells=cells)
    return {**meta, "stale": True, "applied": len(ops)}


@bp.post("/workbooks/<wb_id>/analysis/status")
@api
def analysis_status(wb_id):
    workbooks.require_workbook(wb_id)
    body = _payload()
    status = str(body.get("status") or "").lower()
    ids = body.get("ids") or []
    cells = body.get("cells") or []
    if not isinstance(ids, list) or not isinstance(cells, list):
        raise ValueError("The finding list is not valid.")
    if len(ids) > 2000 or len(cells) > 2000:
        raise ValueError("Update at most 2000 findings at a time.")
    clean_cells = []
    for item in cells:
        try:
            clean_cells.append((int(item[0]), int(item[1])))
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("A finding cell reference is not valid.") from exc
    changed = workbooks.set_analysis_status(wb_id, status, ids=ids, cells=clean_cells)
    return {"changed": changed}


# --------------------------------------------------------------------------- #
# Exports and printing
# --------------------------------------------------------------------------- #

@bp.get("/export/formats")
@api
def export_formats():
    return {"formats": exporters.limitations(), "default_root": config.describe_storage()}


@bp.post("/workbooks/<wb_id>/export")
@api
def export(wb_id):
    body = _payload()
    options = {
        "filters": body.get("filters"),
        "sort": body.get("sort"),
        "search": body.get("search", ""),
        "rows": body.get("rows"),
        "columns": body.get("columns"),
        "row_start": body.get("row_start"),
        "row_end": body.get("row_end"),
        "include_hidden": body.get("include_hidden", True),
    }
    record = exporters.export(wb_id, body.get("format", "xlsx"),
                              body.get("sheet_ids") or None, options)
    return {"export": record}, 201


@bp.get("/exports")
@api
def export_history():
    return {"exports": exporters.history(request.args.get("workbook_id") or None)}


@bp.get("/export/<int:export_id>/download")
def download_export(export_id):
    record = exporters.get_export(export_id)
    return send_file(record["path"], as_attachment=True,
                     download_name=record["filename"], conditional=True)


@bp.get("/workbooks/<wb_id>/original")
def download_original(wb_id):
    try:
        path = workbooks.original_file(wb_id)
        workbook = workbooks.require_workbook(wb_id)
    except LookupError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404
    if not path:
        return jsonify({"ok": False, "error": "This workbook has no uploaded original."}), 404
    return send_file(path, as_attachment=True,
                     download_name=workbook.get("original_name") or path.name,
                     conditional=True)


@bp.get("/workbooks/<wb_id>/print")
def print_view(wb_id):
    try:
        workbook = workbooks.require_workbook(wb_id)
        sheet_id = request.args.get("sheet_id") or workbook["sheets"][0]["id"]
        filters = _json_arg("filters")
        sort = _json_arg("sort")
        view = workbooks.view_data(
            wb_id, sheet_id, filters=filters, sort=sort,
            search=request.args.get("search", ""),
        )
    except LookupError as exc:
        return f"<h1>Not found</h1><p>{exc}</p>", 404
    except ValueError as exc:
        return f"<h1>Print view</h1><p>{exc}</p>", 400
    rows = []
    for row in view["rows"]:
        cells = []
        for col in view["columns"]:
            cell = view["cells"].get((row, col))
            cells.append({
                "value": "" if not cell or cell.get("v") is None else cell["v"],
                "style": cell.get("s") if cell else None,
            })
        rows.append({"number": row, "cells": cells})
    labels = []
    for col in view["columns"]:
        head = view["cells"].get((1, col))
        labels.append(str(head["v"]) if head and head.get("v") is not None
                      else _column_letter(col))
    return render_template(
        "workspace_print.html", workbook=workbook, sheet=view["sheet"],
        rows=rows, labels=labels, columns=view["columns"], filters=filters,
        query=request.args.get("search", ""), total=len(rows),
    )


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #

@bp.post("/upload")
@api
def upload():
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        raise ValueError("Choose at least one file to upload.")
    if len(files) > 20:
        raise ValueError("Upload up to 20 files at a time.")
    created, errors = [], []
    for upload_file in files:
        filename = Path(str(upload_file.filename or "")).name
        if not filename:
            errors.append({"file": "(unnamed)", "error": "The file has no name."})
            continue
        extension = Path(filename).suffix.lower()
        if extension not in config.ALLOWED_EXTENSIONS:
            errors.append({"file": filename,
                           "error": "Unsupported type. Use .xlsx, .xls, .xlsm, .csv or .ods."})
            continue
        try:
            stream = upload_file.stream
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(0)
            if size == 0:
                raise ValueError("The file is empty.")
            if size > config.MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError(f"The file exceeds the {config.MAX_UPLOAD_MB} MB limit.")
            workbook = workbooks.create_from_stream(stream, filename)
            try:
                analysis = _run_analysis(workbook["id"], workbook["sheets"][0]["id"])
                workbook["analysis_summary"] = {
                    "issues": len(analysis.get("issues", [])),
                    "analysed_rows": analysis.get("analyzed_rows", 0),
                    "sheet_name": analysis.get("sheet_name", ""),
                }
            except Exception as exc:  # noqa: BLE001 - import must still succeed
                current_app.logger.exception("Automatic analysis failed for %s: %s", filename, exc)
                workbook["analysis_summary"] = None
            created.append(workbook)
        except ValueError as exc:
            errors.append({"file": filename, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - report per-file failures
            current_app.logger.exception("Upload failed for %s", filename)
            errors.append({"file": filename,
                           "error": f"This file could not be read: {type(exc).__name__}"})
    if not created and errors:
        return {"ok": False, "created": [], "errors": errors,
                "error": errors[0]["error"]}, 400
    return {"created": created, "errors": errors}, 201


@bp.post("/import-path")
@api
def import_path():
    path = str(_payload().get("path") or "").strip()
    if not path:
        raise ValueError("Choose a file first.")
    return {"workbook": workbooks.import_local_path(path)}, 201

==============================================================================
FILE: datafix/config.py
==============================================================================
"""Configuration for the permanent workspace.

The storage root can be changed at runtime (Settings -> Storage location). Each
workbook remembers the root it was created in, so old files stay reachable
after the setting changes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG_DIR.mkdir(exist_ok=True)
STORAGE_CONFIG = CONFIG_DIR / "storage.json"
DEFAULT_STORAGE_ROOT = BASE_DIR / "storage"

DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(exist_ok=True)
DATABASE_PATH = Path(os.environ.get("DATAFIX_DB", DATABASE_DIR / "datafix.db"))

MAX_UPLOAD_MB = int(os.environ.get("DATAFIX_MAX_MB", "128"))
MAX_CELLS_PER_SHEET = int(os.environ.get("DATAFIX_MAX_CELLS", "300000"))
DEFAULT_VERSION_RETENTION = int(os.environ.get("DATAFIX_VERSION_RETENTION", "120"))
VERSION_INTERVAL_SECONDS = int(os.environ.get("DATAFIX_VERSION_INTERVAL", "300"))

ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".csv", ".ods"}
EXPORT_FORMATS = ("xlsx", "xlsm", "xls", "csv", "ods", "pdf", "docx", "json")

SUBDIRS = ("workbooks", "backups", "exports", "reports")


def ensure_tree(root: Path) -> Path:
    root = Path(root)
    for name in SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def get_storage_root() -> Path:
    try:
        payload = json.loads(STORAGE_CONFIG.read_text(encoding="utf-8"))
        raw = payload.get("root")
        if isinstance(raw, str) and raw.strip():
            root = Path(raw).expanduser()
            if root.is_absolute():
                return ensure_tree(root)
    except (OSError, ValueError):
        pass
    return ensure_tree(DEFAULT_STORAGE_ROOT)


def describe_storage() -> dict:
    root = get_storage_root()
    return {
        "root": str(root),
        "default": str(DEFAULT_STORAGE_ROOT),
        "exists": root.is_dir(),
        "writable": os.access(root, os.W_OK),
        "database": str(DATABASE_PATH),
    }


def validate_directory(raw: str) -> tuple[Path | None, str]:
    """Accept only an absolute local folder that can hold DataFix files."""
    text = (raw or "").strip().strip('"')
    if not text:
        return None, "Enter a folder path first."
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return None, "Use a full path, for example C:\\DataFix or /home/me/DataFix."
    try:
        candidate = candidate.resolve()
    except OSError as exc:
        return None, f"That path cannot be resolved: {exc}"
    if candidate.exists() and not candidate.is_dir():
        return None, "That path is a file, not a folder."
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        ensure_tree(candidate)
        probe = candidate / ".datafix-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return None, f"DataFix cannot write to that folder: {exc}"
    return candidate, ""


def set_storage_root(raw: str) -> tuple[Path | None, str]:
    candidate, error = validate_directory(raw)
    if candidate is None:
        return None, error
    STORAGE_CONFIG.write_text(
        json.dumps({"root": str(candidate)}, indent=2), encoding="utf-8"
    )
    return candidate, ""


def reset_storage_root() -> Path:
    try:
        STORAGE_CONFIG.unlink()
    except OSError:
        pass
    return get_storage_root()


def get_version_retention() -> int:
    try:
        payload = json.loads(STORAGE_CONFIG.read_text(encoding="utf-8"))
        value = int(payload.get("version_retention", DEFAULT_VERSION_RETENTION))
        return max(5, value)
    except (OSError, ValueError, TypeError):
        return DEFAULT_VERSION_RETENTION


def set_version_retention(value: int) -> int:
    retention = max(5, min(1000, int(value)))
    payload = {}
    try:
        payload = json.loads(STORAGE_CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    payload["version_retention"] = retention
    STORAGE_CONFIG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return retention

==============================================================================
FILE: datafix/db.py
==============================================================================
"""SQLite storage for workbooks, cells, edit batches, versions and exports."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS workbooks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'upload',
    original_name TEXT,
    extension TEXT,
    storage_root TEXT NOT NULL,
    starred INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    analysis_at TEXT,
    analysis_stale INTEGER NOT NULL DEFAULT 1,
    version_count INTEGER NOT NULL DEFAULT 0,
    save_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sheets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workbook_id TEXT NOT NULL REFERENCES workbooks(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    position INTEGER NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 100,
    col_count INTEGER NOT NULL DEFAULT 20,
    frozen_rows INTEGER NOT NULL DEFAULT 0,
    frozen_cols INTEGER NOT NULL DEFAULT 0,
    hidden_rows TEXT NOT NULL DEFAULT '[]',
    hidden_cols TEXT NOT NULL DEFAULT '[]',
    merges TEXT NOT NULL DEFAULT '[]',
    UNIQUE(workbook_id, name)
);

CREATE TABLE IF NOT EXISTS cells (
    sheet_id INTEGER NOT NULL REFERENCES sheets(id) ON DELETE CASCADE,
    row INTEGER NOT NULL,
    col INTEGER NOT NULL,
    value TEXT,
    num REAL,
    formula TEXT,
    style TEXT,
    PRIMARY KEY (sheet_id, row, col)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS cells_value_idx ON cells(value);
CREATE INDEX IF NOT EXISTS cells_formula_idx ON cells(formula);
CREATE INDEX IF NOT EXISTS sheets_workbook_idx ON sheets(workbook_id, position);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workbook_id TEXT NOT NULL,
    label TEXT NOT NULL,
    forward TEXT NOT NULL,
    inverse TEXT NOT NULL,
    undone INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS batches_workbook_idx ON batches(workbook_id, id);

CREATE TABLE IF NOT EXISTS versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workbook_id TEXT NOT NULL,
    number INTEGER NOT NULL,
    label TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    cell_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS versions_workbook_idx ON versions(workbook_id, number);

CREATE TABLE IF NOT EXISTS exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workbook_id TEXT NOT NULL,
    format TEXT NOT NULL,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edits_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workbook_id TEXT NOT NULL,
    sheet_id INTEGER,
    ref TEXT,
    kind TEXT NOT NULL,
    before TEXT,
    after TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS edits_workbook_idx ON edits_log(workbook_id, id);

CREATE TABLE IF NOT EXISTS analyses (
    workbook_id TEXT PRIMARY KEY,
    sheet_id INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    Path(config.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


@contextmanager
def transaction():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@contextmanager
def read():
    connection = _connect()
    try:
        yield connection
    finally:
        connection.close()


def init_db() -> None:
    with transaction() as connection:
        connection.executescript(SCHEMA)


init_db()

==============================================================================
FILE: datafix/exporters.py
==============================================================================
"""Export a workbook (or a filtered/selected view) to Excel, CSV, ODS, PDF,
DOCX or JSON and keep a permanent record of every export.

All exporters read from the SQLite store through :mod:`datafix.workbooks`; the
original uploaded file is never touched.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
from pathlib import Path

from . import config, db, workbooks

FORMAT_INFO = {
    "xlsx": {
        "label": "Excel workbook (.xlsx)",
        "limits": "Full data, formulas, cell styles, merged cells, frozen panes and hidden "
                  "rows/columns. Macros are never included.",
    },
    "xlsm": {
        "label": "Macro-enabled Excel workbook (.xlsm)",
        "limits": "Same content as .xlsx. Macros from uploaded files are never copied or "
                  "executed, so the exported file contains no VBA project.",
    },
    "xls": {
        "label": "Excel 97-2003 workbook (.xls)",
        "limits": "Values, formulas and basic styling. Maximum 65,536 rows and 256 columns; "
                  "unhidden rows only and no merged/hidden layout is preserved.",
    },
    "csv": {
        "label": "Comma-separated values (.csv)",
        "limits": "One worksheet only, plain values (formulas are written as their last "
                  "computed value), UTF-8 with BOM. Styling and layout are not included.",
    },
    "ods": {
        "label": "OpenDocument spreadsheet (.ods)",
        "limits": "Values, formulas, basic text styling and row/column counts. Complex "
                  "formatting and merged cells may be simplified.",
    },
    "pdf": {
        "label": "PDF document (.pdf)",
        "limits": "Print-style layout of the selected data. Formulas are written as their "
                  "computed values; editing is not possible.",
    },
    "docx": {
        "label": "Word document (.docx)",
        "limits": "Tables of computed values with a bold header row. Formulas and cell "
                  "styling are not preserved.",
    },
    "json": {
        "label": "DataFix JSON (.json)",
        "limits": "Complete lossless structure (values, formulas, styles, layout) for "
                  "backup or re-import to another system. Not an Excel format.",
    },
}

EXPECTED = {
    "xlsx": ".xlsx", "xlsm": ".xlsm", "xls": ".xls", "csv": ".csv",
    "ods": ".ods", "pdf": ".pdf", "docx": ".docx", "json": ".json",
}

_INVALID = re.compile(r"[^A-Za-z0-9._ -]+")


def _safe(text: str, fallback: str) -> str:
    cleaned = _INVALID.sub("_", str(text or "")).strip(" ._")
    return cleaned[:80] or fallback


def _cell_value(cell) -> str:
    if not cell:
        return ""
    value = cell.get("v")
    if value is None and cell.get("f"):
        return ""
    return "" if value is None else str(value)


def _excel_value(cell):
    if not cell:
        return None
    if cell.get("f"):
        return cell["f"] if str(cell["f"]).startswith("=") else f"={cell['f']}"
    if cell.get("num") is not None:
        number = float(cell["num"])
        return int(number) if number.is_integer() and abs(number) < 10 ** 15 else number
    return cell.get("v")


def _rgb(value, default=None):
    if not isinstance(value, str):
        return default
    text = value.strip().lstrip("#").upper()
    if re.fullmatch(r"[0-9A-F]{6}", text):
        return text
    return default


def _selected_sheets(wb: dict, sheet_ids) -> list[dict]:
    sheets = wb["sheets"]
    if not sheet_ids:
        return sheets
    wanted = {int(value) for value in sheet_ids}
    chosen = [sheet for sheet in sheets if int(sheet["id"]) in wanted]
    if not chosen:
        raise ValueError("Choose at least one worksheet to export.")
    return chosen


def _views(wb: dict, sheets: list[dict], options: dict) -> list[dict]:
    multiple = len(sheets) > 1
    views = []
    for sheet in sheets:
        view = workbooks.view_data(
            wb["id"], sheet["id"],
            filters=None if multiple else options.get("filters"),
            sort=None if multiple else options.get("sort"),
            search="" if multiple else options.get("search", ""),
            rows=None if multiple else options.get("rows"),
            row_start=None if multiple else options.get("row_start"),
            row_end=None if multiple else options.get("row_end"),
            columns=None if multiple else options.get("columns"),
            include_hidden=bool(options.get("include_hidden", True)),
        )
        views.append(view)
    return views


def _unique_path(folder: Path, stem: str, extension: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}{extension}"
    counter = 2
    while path.exists():
        path = folder / f"{stem}-{counter}{extension}"
        counter += 1
    return path


def _sheet_title(name: str, used: set[str]) -> str:
    base = _INVALID.sub(" ", str(name or "Sheet")).strip()[:31] or "Sheet"
    candidate, index = base, 2
    while candidate.lower() in used:
        tail = f" ({index})"
        candidate = base[: 31 - len(tail)] + tail
        index += 1
    used.add(candidate.lower())
    return candidate


# --------------------------------------------------------------------------- #
# XLSX / XLSM
# --------------------------------------------------------------------------- #

def _write_xlsx(views: list[dict], path: Path, workbook_name: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    book = Workbook()
    book.remove(book.active)
    used: set[str] = set()
    for view in views:
        meta = view["sheet"]
        sheet = book.create_sheet(_sheet_title(meta["name"], used))
        rows, cols = view["rows"], view["columns"]
        for out_row, source_row in enumerate(rows, start=1):
            for out_col, source_col in enumerate(cols, start=1):
                cell = view["cells"].get((source_row, source_col))
                target = sheet.cell(row=out_row, column=out_col)
                target.value = _excel_value(cell)
                style = cell.get("s") if cell else None
                if style:
                    font_kwargs = {
                        "bold": bool(style.get("bold")),
                        "italic": bool(style.get("italic")),
                        "strike": bool(style.get("strike")),
                        "underline": "single" if style.get("underline") else None,
                    }
                    colour = _rgb(style.get("color"))
                    if colour:
                        font_kwargs["color"] = "FF" + colour
                    if style.get("font"):
                        font_kwargs["name"] = str(style["font"])[:60]
                    if style.get("size"):
                        try:
                            font_kwargs["size"] = float(style["size"])
                        except (TypeError, ValueError):
                            pass
                    target.font = Font(**font_kwargs)
                    fill = _rgb(style.get("fill"))
                    if fill:
                        target.fill = PatternFill("solid", fgColor="FF" + fill)
                    horizontal = style.get("align")
                    vertical = style.get("valign")
                    if horizontal in ("left", "center", "right") or vertical or style.get("wrap"):
                        target.alignment = Alignment(
                            horizontal=horizontal if horizontal in ("left", "center", "right") else None,
                            vertical=vertical if vertical in ("top", "center", "bottom") else None,
                            wrap_text=bool(style.get("wrap")),
                        )
                    if style.get("numfmt"):
                        target.number_format = str(style["numfmt"])[:60]
        if view["complete"]:
            if meta.get("frozen_rows") or meta.get("frozen_cols"):
                row = int(meta.get("frozen_rows") or 0) + 1
                col = int(meta.get("frozen_cols") or 0) + 1
                sheet.freeze_panes = f"{get_column_letter(min(col, 16384))}{min(row, 1048576)}"
            for merged in meta.get("merges") or []:
                try:
                    sheet.merge_cells(
                        start_row=int(merged["r1"]), start_column=int(merged["c1"]),
                        end_row=int(merged["r2"]), end_column=int(merged["c2"]),
                    )
                except (KeyError, ValueError, TypeError):
                    continue
            for row in meta.get("hidden_rows") or []:
                sheet.row_dimensions[int(row)].hidden = True
            for col in meta.get("hidden_cols") or []:
                sheet.column_dimensions[get_column_letter(int(col))].hidden = True
    if not book.worksheets:
        book.create_sheet(_sheet_title(workbook_name, used))
    book.save(path)


# --------------------------------------------------------------------------- #
# XLS
# --------------------------------------------------------------------------- #

def _write_xls(views: list[dict], path: Path) -> None:
    import xlwt

    book = xlwt.Workbook(encoding="utf-8")
    styles: dict[str, object] = {}

    def make_style(style):
        key = json.dumps(style, sort_keys=True)
        if key in styles:
            return styles[key]
        xf = xlwt.XFStyle()
        font = xlwt.Font()
        font.bold = bool(style.get("bold"))
        font.italic = bool(style.get("italic"))
        font.strikeout = bool(style.get("strike"))
        font.underline = (xlwt.Font.UNDERLINE_SINGLE if style.get("underline")
                          else xlwt.Font.UNDERLINE_NONE)
        colour = _rgb(style.get("color"))
        if colour:
            try:
                font.colour_index = xlwt.Style.colour_map.get(
                    "white" if colour == "FFFFFF" else "black")
            except Exception:  # noqa: BLE001
                pass
        if style.get("fill") and _rgb(style.get("fill")):
            pattern = xlwt.Pattern()
            pattern.pattern = xlwt.Pattern.SOLID_PATTERN
            pattern.pattern_fore_colour = 0x16  # light grey fallback
            xf.pattern = pattern
        xf.font = font
        alignment = xlwt.Alignment()
        horizontal = {"left": 1, "center": 2, "right": 3}.get(style.get("align"))
        if horizontal:
            alignment.horz = horizontal
        alignment.wrap = 1 if style.get("wrap") else 0
        xf.alignment = alignment
        styles[key] = xf
        return xf

    used: set[str] = set()
    written = 0
    for view in views:
        meta = view["sheet"]
        sheet = book.add_sheet(_sheet_title(meta["name"], used))
        written += 1
        rows, cols = view["rows"], view["columns"]
        if len(rows) > 65536:
            rows = rows[:65536]
        if len(cols) > 256:
            cols = cols[:256]
        for out_row, source_row in enumerate(rows):
            for out_col, source_col in enumerate(cols):
                cell = view["cells"].get((source_row, source_col))
                if not cell:
                    continue
                style = make_style(cell["s"]) if cell.get("s") else xlwt.Style.default_style
                formula = cell.get("f")
                if formula:
                    try:
                        sheet.write(out_row, out_col,
                                    xlwt.Formula(str(formula).lstrip("=")), style)
                        continue
                    except Exception:  # noqa: BLE001 - fall back to cached value
                        pass
                value = _excel_value(cell)
                try:
                    sheet.write(out_row, out_col, value, style)
                except Exception:  # noqa: BLE001
                    sheet.write(out_row, out_col, _cell_value(cell), style)
        if view["complete"] and (meta.get("frozen_rows") or meta.get("frozen_cols")):
            sheet.set_panes_frozen(True)
            if meta.get("frozen_rows"):
                sheet.set_horz_split_pos(int(meta["frozen_rows"]))
            if meta.get("frozen_cols"):
                sheet.set_vert_split_pos(int(meta["frozen_cols"]))
    if not written:  # pragma: no cover - defensive
        book.add_sheet("Sheet")
    book.save(str(path))


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #

def _write_csv(view: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        for row in view["rows"]:
            values = [_cell_value(view["cells"].get((row, col)))
                      for col in view["columns"]]
            while values and values[-1] == "":
                values.pop()
            writer.writerow(values)


# --------------------------------------------------------------------------- #
# ODS
# --------------------------------------------------------------------------- #

def _write_ods(views: list[dict], path: Path) -> None:
    from odf import opendocument, style, table, text

    document = opendocument.OpenDocumentSpreadsheet()
    used: set[str] = set()
    style_cache: dict[str, object] = {}

    def make_style(style_data):
        key = json.dumps(style_data, sort_keys=True)
        if key in style_cache:
            return style_cache[key]
        name = f"df{len(style_cache) + 1}"
        cell_style = style.Style(name=name, family="table-cell")
        props = {}
        if style_data.get("bold"):
            props["fontweight"] = "bold"
        if style_data.get("italic"):
            props["fontstyle"] = "italic"
        colour = _rgb(style_data.get("color"))
        if colour:
            props["color"] = f"#{colour}"
        if style_data.get("size"):
            try:
                props["fontsize"] = f"{float(style_data['size'])}pt"
            except (TypeError, ValueError):
                pass
        if props:
            cell_style.addElement(style.TextProperties(**props))
        fill = _rgb(style_data.get("fill"))
        if fill:
            cell_style.addElement(style.TableCellProperties(backgroundcolor=f"#{fill}"))
        document.automaticstyles.addElement(cell_style)
        style_cache[key] = cell_style
        return cell_style

    for view in views:
        meta = view["sheet"]
        sheet = table.Table(name=_sheet_title(meta["name"], used))
        document.spreadsheet.addElement(sheet)
        for row in view["rows"]:
            row_element = table.TableRow()
            sheet.addElement(row_element)
            for col in view["columns"]:
                cell = view["cells"].get((row, col))
                value = cell.get("v") if cell else None
                attrs = {}
                style_data = cell.get("s") if cell else None
                if style_data:
                    attrs["stylename"] = make_style(style_data)
                number = cell.get("num") if cell else None
                if cell and cell.get("f") and cell.get("v") is not None:
                    attrs["formula"] = "of:=" + str(cell["f"]).lstrip("=").replace(",", ";")
                if number is not None:
                    attrs["valuetype"] = "float"
                    attrs["value"] = number
                    cell_element = table.TableCell(**attrs)
                    cell_element.addElement(text.P(text=str(value if value is not None else number)))
                elif value is None or value == "":
                    cell_element = table.TableCell(**attrs)
                elif isinstance(value, str) and value.upper() in ("TRUE", "FALSE"):
                    attrs["valuetype"] = "boolean"
                    attrs["booleanvalue"] = value.upper() == "TRUE"
                    cell_element = table.TableCell(**attrs)
                    cell_element.addElement(text.P(text=value))
                else:
                    attrs["valuetype"] = "string"
                    cell_element = table.TableCell(**attrs)
                    cell_element.addElement(text.P(text=str(value)))
                row_element.addElement(cell_element)
    if not document.spreadsheet.getElementsByType(table.Table):
        document.spreadsheet.addElement(table.Table(name="Sheet"))
    document.save(str(path))


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #

def _write_pdf(views: list[dict], path: Path, title: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    styles = getSampleStyleSheet()
    cell_style = styles["BodyText"].clone("DataFixCell", fontSize=7, leading=8.5)
    head_style = styles["BodyText"].clone("DataFixHead", fontSize=7.5, leading=9,
                                          fontName="Helvetica-Bold")
    story = []
    for index, view in enumerate(views):
        meta = view["sheet"]
        story.append(Paragraph(f"{title} — {meta['name']}", styles["Heading2"]))
        rows, cols = view["rows"], view["columns"]
        if not rows:
            story.append(Paragraph("This view has no rows.", styles["BodyText"]))
        else:
            header = []
            for col in cols:
                head = view["cells"].get((1, col))
                label = _cell_value(head) or _column_name(col)
                header.append(Paragraph(label, head_style))
            data = [header]
            for row in rows:
                if row == 1 and len(rows) > 1:
                    continue
                line = []
                for col in cols:
                    value = _cell_value(view["cells"].get((row, col)))
                    line.append(Paragraph(value.replace("\n", "<br/>"), cell_style)
                                if value else "")
                data.append(line)
            usable = landscape(A4)[0] - 40 * mm
            width = max(24, usable / max(1, len(cols)))
            table_data = Table(data, repeatRows=1,
                               colWidths=[width] * len(cols), splitByRow=1)
            table_data.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8C2D0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]))
            story.append(table_data)
        if index < len(views) - 1:
            story.append(PageBreak())
        else:
            story.append(Spacer(1, 6))
    if not story:
        story.append(Paragraph(f"{title}: no worksheets selected.", styles["BodyText"]))
    doc = SimpleDocTemplate(
        str(path), pagesize=landscape(A4),
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title=title, author="DataFix",
    )
    doc.build(story)


def _column_name(col: int) -> str:
    name = ""
    value = int(col)
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        name = chr(65 + remainder) + name
    return name


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #

def _write_docx(views: list[dict], path: Path, title: str) -> None:
    from docx import Document
    from docx.shared import Pt

    document = Document()
    document.add_heading(title, level=1)
    for view in views:
        meta = view["sheet"]
        document.add_heading(meta["name"], level=2)
        rows, cols = view["rows"], view["columns"]
        if not rows:
            document.add_paragraph("This view has no rows.")
            continue
        table = document.add_table(rows=len(rows), cols=max(1, len(cols)))
        table.style = "Table Grid"
        for out_row, row in enumerate(rows):
            for out_col, col in enumerate(cols):
                cell = table.cell(out_row, out_col)
                text = _cell_value(view["cells"].get((row, col)))
                cell.text = text
                if out_row == 0:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.bold = True
                            run.font.size = Pt(9)
                else:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(9)
    document.save(str(path))


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #

def _write_json(wb: dict, views: list[dict], path: Path) -> None:
    payload = {
        "format": "datafix-workbook",
        "version": 1,
        "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
        "workbook": {
            "id": wb["id"],
            "name": wb["name"],
            "original_name": wb.get("original_name"),
            "created_at": wb.get("created_at"),
            "updated_at": wb.get("updated_at"),
        },
        "sheets": [],
    }
    for view in views:
        meta = view["sheet"]
        cells = []
        for (row, col), cell in sorted(view["cells"].items()):
            cells.append({
                "row": row, "col": col,
                "value": cell.get("v"), "number": cell.get("num"),
                "formula": cell.get("f"), "style": cell.get("s"),
            })
        payload["sheets"].append({
            "id": meta["id"], "name": meta["name"],
            "row_count": meta["row_count"], "col_count": meta["col_count"],
            "frozen_rows": meta.get("frozen_rows", 0),
            "frozen_cols": meta.get("frozen_cols", 0),
            "hidden_rows": meta.get("hidden_rows", []),
            "hidden_cols": meta.get("hidden_cols", []),
            "merges": meta.get("merges", []),
            "exported_rows": view["rows"],
            "exported_columns": view["columns"],
            "cells": cells,
        })
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def export(wb_id: str, fmt: str, sheet_ids=None, options=None) -> dict:
    fmt = str(fmt or "").lower()
    if fmt not in config.EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {fmt}")
    options = options or {}
    wb = workbooks.require_workbook(wb_id)
    sheets = _selected_sheets(wb, sheet_ids)
    views = _views(wb, sheets, options)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    scope = "all" if len(sheets) > 1 else _safe(sheets[0]["name"], "sheet")
    stem = f"{_safe(wb['name'], 'workbook')}-{_safe(scope, 'sheet')}-{stamp}"
    extension = EXPECTED[fmt]
    folder = Path(wb["storage_root"]) / "exports" / wb["id"]
    path = _unique_path(folder, stem, extension)
    try:
        if fmt in ("xlsx", "xlsm"):
            _write_xlsx(views, path, wb["name"])
        elif fmt == "xls":
            _write_xls(views, path)
        elif fmt == "csv":
            _write_csv(views[0], path)
        elif fmt == "ods":
            _write_ods(views, path)
        elif fmt == "pdf":
            _write_pdf(views, path, wb["name"])
        elif fmt == "docx":
            _write_docx(views, path, wb["name"])
        else:
            _write_json(wb, views, path)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise

    notes = [FORMAT_INFO[fmt]["limits"]]
    if fmt in ("xls", "csv", "pdf", "docx") and len(sheets) > 1:
        notes.append(f"Only the first selected worksheet ({sheets[0]['name']}) is "
                     f"written by {EXPECTED[fmt]} exports." if fmt == "csv"
                     else "Multi-sheet export is supported; values are compacted.")
    if options.get("rows") or options.get("filters") or options.get("search") \
            or options.get("columns") or options.get("row_start"):
        notes.append("The export contains only the current filtered/selected view.")

    with db.transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO exports (workbook_id, format, filename, path, size, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (wb_id, fmt, path.name, str(path), path.stat().st_size,
             dt.datetime.now().isoformat(timespec="seconds")),
        )
        export_id = cursor.lastrowid
    return {
        "id": export_id, "format": fmt, "filename": path.name,
        "size": path.stat().st_size, "notes": notes,
        "url": f"/api/workspace/export/{export_id}/download",
    }


def history(workbook_id: str | None = None, limit: int = 200) -> list[dict]:
    with db.read() as conn:
        if workbook_id:
            rows = conn.execute(
                "SELECT id, workbook_id, format, filename, size, created_at FROM exports"
                " WHERE workbook_id=? ORDER BY id DESC LIMIT ?", (workbook_id, int(limit))
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, workbook_id, format, filename, size, created_at FROM exports"
                " ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
    return [dict(row) for row in rows]


def get_export(export_id: int) -> dict:
    with db.read() as conn:
        row = conn.execute(
            "SELECT id, workbook_id, format, filename, path, size, created_at"
            " FROM exports WHERE id=?", (int(export_id),)
        ).fetchone()
    if not row:
        raise LookupError("Export not found.")
    record = dict(row)
    path = Path(record["path"])
    if not path.is_file():
        raise LookupError("The exported file is no longer on disk.")
    record["path"] = path
    return record


def limitations() -> dict:
    return FORMAT_INFO

==============================================================================
FILE: datafix/formulas.py
==============================================================================
"""A small, safe Excel-style formula evaluator.

Only the functions listed in ``FUNCTIONS`` can run; formulas are parsed into
an AST and evaluated by hand (never passed to Python's eval). Anything that
cannot be evaluated returns a spreadsheet error string such as ``#VALUE!``.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

MAX_FORMULA_LENGTH = 4000


class FormulaError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class RangeValue:
    def __init__(self, rows: list[list]):
        self.rows = rows

    def flatten(self) -> list:
        return [cell for row in self.rows for cell in row]


TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<QREF>'(?:[^']|'')+'!\$?[A-Za-z]{1,3}\$?\d+)
  | (?P<STRING>"(?:[^"]|"")*")
  | (?P<REF>(?:'(?:[^']|'')+'|[A-Za-z_][A-Za-z0-9_. ]*)!\$?[A-Za-z]{1,3}\$?\d+
            |\$?[A-Za-z]{1,3}\$?\d+(?![\w.]))
  | (?P<NUMBER>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<OP><=|>=|<>|[-+*/^%&=<>(),:])
    """,
    re.VERBOSE,
)

CELL_RE = re.compile(r"^\$?([A-Za-z]{1,3})\$?(\d+)$")


def _col_index(letters: str) -> int:
    value = 0
    for char in letters.upper():
        value = value * 26 + (ord(char) - 64)
    return value


def col_letter(index: int) -> str:
    letters = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def parse_ref(text: str) -> tuple[str | None, int, int]:
    sheet = None
    body = text
    if "!" in text:
        head, body = text.rsplit("!", 1)
        sheet = head
        if sheet.startswith("'") and sheet.endswith("'"):
            sheet = sheet[1:-1].replace("''", "'")
    match = CELL_RE.match(body.replace("$", ""))
    if not match:
        raise FormulaError("#REF!")
    return sheet, int(match.group(2)), _col_index(match.group(1))


class Parser:
    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.pos = 0

    @staticmethod
    def _tokenize(text: str) -> list[tuple[str, str]]:
        tokens: list[tuple[str, str]] = []
        index = 0
        while index < len(text):
            match = TOKEN_RE.match(text, index)
            if not match:
                raise FormulaError("#NAME?")
            kind = match.lastgroup
            value = match.group()
            index = match.end()
            if kind == "WS":
                continue
            tokens.append((kind, value))
        tokens.append(("EOF", ""))
        return tokens

    def peek(self, offset: int = 0) -> tuple[str, str]:
        position = self.pos + offset
        return self.tokens[position] if position < len(self.tokens) else ("EOF", "")

    def take(self) -> tuple[str, str]:
        token = self.peek()
        self.pos += 1
        return token

    def expect(self, value: str) -> None:
        kind, token = self.take()
        if token != value:
            raise FormulaError("#NAME?")

    def parse(self):
        node = self.parse_expression()
        if self.peek()[0] != "EOF":
            raise FormulaError("#NAME?")
        return node

    def parse_expression(self):
        left = self.parse_concat()
        while self.peek()[1] in ("=", "<>", "<", ">", "<=", ">="):
            op = self.take()[1]
            left = ("bin", op, left, self.parse_concat())
        return left

    def parse_concat(self):
        left = self.parse_additive()
        while self.peek()[1] == "&":
            self.take()
            left = ("bin", "&", left, self.parse_additive())
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            left = ("bin", op, left, self.parse_multiplicative())
        return left

    def parse_multiplicative(self):
        left = self.parse_power()
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            left = ("bin", op, left, self.parse_power())
        return left

    def parse_power(self):
        left = self.parse_unary()
        while self.peek()[1] == "^":
            self.take()
            left = ("bin", "^", left, self.parse_unary())
        return left

    def parse_unary(self):
        if self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            return ("un", op, self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        node = self.parse_primary()
        while self.peek()[1] == "%":
            self.take()
            node = ("un", "%", node)
        return node

    def parse_primary(self):
        kind, value = self.peek()
        if kind == "NUMBER":
            self.take()
            number = float(value)
            return ("num", int(number) if number.is_integer() else number)
        if kind == "STRING":
            self.take()
            return ("str", value[1:-1].replace('""', '"'))
        if kind in ("REF", "QREF"):
            self.take()
            sheet, row, col = parse_ref(value)
            if self.peek()[1] == ":":
                self.take()
                rkind, rvalue = self.take()
                if rkind not in ("REF", "QREF"):
                    raise FormulaError("#REF!")
                sheet2, row2, col2 = parse_ref(rvalue)
                return ("range", sheet2 or sheet, row, col, row2, col2)
            return ("ref", sheet, row, col)
        if kind == "IDENT":
            self.take()
            upper = value.upper()
            if self.peek()[1] == "(":
                self.take()
                args = []
                if self.peek()[1] != ")":
                    while True:
                        args.append(self.parse_expression())
                        if self.peek()[1] == ",":
                            self.take()
                            continue
                        break
                self.expect(")")
                return ("func", upper, args)
            if upper == "TRUE":
                return ("bool", True)
            if upper == "FALSE":
                return ("bool", False)
            raise FormulaError("#NAME?")
        if value == "(":
            self.take()
            node = self.parse_expression()
            self.expect(")")
            return node
        raise FormulaError("#NAME?")


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #

def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def to_number(value):
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if is_number(value):
        return value
    text = str(value).strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        raise FormulaError("#VALUE!")


def to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if is_number(value):
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    if is_number(value):
        return value != 0
    text = str(value).strip().upper()
    if text in ("TRUE", "YES"):
        return True
    if text in ("FALSE", "NO"):
        return False
    raise FormulaError("#VALUE!")


def _round(number: float, digits: int = 0):
    factor = 10 ** digits
    return round(number * factor) / factor


def _compare(a, b) -> int:
    if is_number(a) and is_number(b):
        return (a > b) - (a < b)
    a_text, b_text = to_text(a).lower(), to_text(b).lower()
    return (a_text > b_text) - (a_text < b_text)


def _match_criteria(value, criteria) -> bool:
    if isinstance(criteria, str) and criteria[:1] in (">", "<", "="):
        match = re.match(r"^(<=|>=|<>|>|<|=)(.*)$", criteria, re.S)
        if match:
            op, raw = match.group(1), match.group(2)
            target = raw
            try:
                target = float(raw)
            except ValueError:
                pass
            if op == "=":
                return _compare(value, target) == 0
            if op == "<>":
                return _compare(value, target) != 0
            if op == ">":
                return _compare(value, target) > 0
            if op == "<":
                return _compare(value, target) < 0
            if op == ">=":
                return _compare(value, target) >= 0
            if op == "<=":
                return _compare(value, target) <= 0
    if isinstance(criteria, str) and any(ch in criteria for ch in "*?"):
        pattern = "^" + re.escape(criteria).replace(r"\*", ".*").replace(r"\?", ".") + "$"
        return re.match(pattern, to_text(value), re.I) is not None
    return _compare(value, criteria) == 0


def _flatten(args) -> list:
    flat = []
    for arg in args:
        if isinstance(arg, RangeValue):
            flat.extend(arg.flatten())
        else:
            flat.append(arg)
    return flat


def _parse_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            continue
    return None


def _format_date(day: date, fmt: str) -> str:
    replacements = (
        ("yyyy", f"{day.year:04d}"), ("yy", f"{day.year % 100:02d}"),
        ("mmmm", day.strftime("%B")), ("mmm", day.strftime("%b")),
        ("mm", f"{day.month:02d}"), ("m", str(day.month)),
        ("dddd", day.strftime("%A")), ("ddd", day.strftime("%a")),
        ("dd", f"{day.day:02d}"), ("d", str(day.day)),
    )
    result = fmt
    for token, replacement in replacements:
        result = re.sub(token, replacement, result, flags=re.I)
    return result


def format_mixed(value, fmt: str) -> str:
    text = to_text(value)
    if not fmt:
        return text
    lowered = fmt.lower()
    if any(token in lowered for token in ("yy", "dd", "mmm")):
        day = _parse_date(value)
        if day:
            return _format_date(day, fmt)
    percent = "%" in fmt
    number = to_number(value) if value not in (None, "") else None
    if number is None:
        return text
    if percent:
        number *= 100
    decimals = 0
    if "." in fmt:
        decimals = len(fmt.split(".", 1)[1].rstrip("%"))
    body = f"{number:,.{decimals}f}" if "," in fmt else f"{number:.{decimals}f}"
    if percent:
        body += "%"
    prefix = "".join(ch for ch in fmt if ch in "$€£₹")
    return prefix + body if prefix else body


class Evaluator:
    def __init__(self, sheets: dict[str, dict]):
        self.sheets = {name.lower(): (name, grid) for name, grid in sheets.items()}
        self.cache: dict[tuple, object] = {}
        self.stack: list[tuple] = []
        self.parsed: dict[str, tuple] = {}

    def resolve_sheet(self, name: str | None) -> tuple[str, dict]:
        if not name:
            raise FormulaError("#REF!")
        found = self.sheets.get(name.lower())
        if not found:
            raise FormulaError("#REF!")
        return found

    def get_cell(self, sheet: str, row: int, col: int):
        name, grid = self.resolve_sheet(sheet)
        key = (name.lower(), row, col)
        if key in self.cache:
            return self.cache[key]
        cell = grid.get((row, col))
        if not cell:
            return None
        formula = cell.get("f")
        if not formula:
            return cell.get("v")
        if key in self.stack:
            raise FormulaError("#CYCLE!")
        self.stack.append(key)
        try:
            node = self.parsed.get(formula)
            if node is None:
                node = Parser(formula[1:] if formula.startswith("=") else formula).parse()
                self.parsed[formula] = node
            value = self.evaluate(node, name)
            value = self.normalise(value)
        finally:
            self.stack.pop()
        self.cache[key] = value
        return value

    @staticmethod
    def normalise(value):
        if isinstance(value, RangeValue):
            if value.rows and value.rows[0]:
                return value.rows[0][0]
            return None
        return value

    def range_values(self, sheet, r1, c1, r2, c2) -> RangeValue:
        name, grid = self.resolve_sheet(sheet)
        rows = []
        top, bottom = sorted((r1, r2))
        left, right = sorted((c1, c2))
        if (bottom - top + 1) * (right - left + 1) > 100000:
            raise FormulaError("#REF!")
        for row in range(top, bottom + 1):
            rows.append([self.get_cell(name, row, col) for col in range(left, right + 1)])
        return RangeValue(rows)

    def evaluate(self, node, sheet: str):
        kind = node[0]
        if kind in ("num", "str", "bool"):
            return node[1]
        if kind == "ref":
            _, ref_sheet, row, col = node
            return self.get_cell(ref_sheet or sheet, row, col)
        if kind == "range":
            _, ref_sheet, r1, c1, r2, c2 = node
            return self.range_values(ref_sheet or sheet, r1, c1, r2, c2)
        if kind == "un":
            op, operand = node[1], node[2]
            value = self.evaluate(operand, sheet)
            if op == "+":
                return to_number(value)
            if op == "-":
                return -to_number(value)
            return to_number(value) / 100
        if kind == "bin":
            op = node[1]
            if op == "&":
                return to_text(self.evaluate(node[2], sheet)) + to_text(self.evaluate(node[3], sheet))
            left = self.evaluate(node[2], sheet)
            right = self.evaluate(node[3], sheet)
            if op in ("=", "<>", "<", ">", "<=", ">="):
                result = _compare(left, right)
                return {
                    "=": result == 0, "<>": result != 0, "<": result < 0,
                    ">": result > 0, "<=": result <= 0, ">=": result >= 0,
                }[op]
            if op == "^":
                return to_number(left) ** to_number(right)
            if op == "/":
                divisor = to_number(right)
                if divisor == 0:
                    raise FormulaError("#DIV/0!")
                return to_number(left) / divisor
            if op == "*":
                return to_number(left) * to_number(right)
            if op == "+":
                return to_number(left) + to_number(right)
            return to_number(left) - to_number(right)
        if kind == "func":
            return self.call(node[1], node[2], sheet)
        raise FormulaError("#NAME?")

    # -- functions -------------------------------------------------------- #

    def call(self, name: str, arg_nodes, sheet: str):
        lazy = name in ("IF", "IFS", "IFERROR", "AND", "OR", "CHOOSE")
        args = None if lazy else [self.evaluate(node, sheet) for node in arg_nodes]

        if name == "IF":
            if len(arg_nodes) < 2:
                raise FormulaError("#VALUE!")
            condition = to_bool(self.evaluate(arg_nodes[0], sheet))
            if condition:
                return self.evaluate(arg_nodes[1], sheet)
            if len(arg_nodes) > 2:
                return self.evaluate(arg_nodes[2], sheet)
            return False
        if name == "IFS":
            if len(arg_nodes) % 2:
                raise FormulaError("#VALUE!")
            for index in range(0, len(arg_nodes), 2):
                if to_bool(self.evaluate(arg_nodes[index], sheet)):
                    return self.evaluate(arg_nodes[index + 1], sheet)
            raise FormulaError("#N/A")
        if name == "IFERROR":
            if len(arg_nodes) != 2:
                raise FormulaError("#VALUE!")
            try:
                return self.evaluate(arg_nodes[0], sheet)
            except FormulaError:
                return self.evaluate(arg_nodes[1], sheet)
        if name in ("AND", "OR"):
            if not arg_nodes:
                raise FormulaError("#VALUE!")
            values = [to_bool(value) for value in _flatten(args)]
            if name == "AND":
                return all(values)
            return any(values)
        if name == "CHOOSE":
            if len(args) < 2:
                raise FormulaError("#VALUE!")
            index = int(to_number(args[0]))
            if not 1 <= index < len(args):
                raise FormulaError("#VALUE!")
            return args[index]

        flat_all = _flatten(args)
        numbers = [value for value in flat_all if is_number(value)]

        if name == "SUM":
            return sum(numbers)
        if name == "AVERAGE":
            if not numbers:
                raise FormulaError("#DIV/0!")
            return sum(numbers) / len(numbers)
        if name == "COUNT":
            return len(numbers)
        if name == "COUNTA":
            return sum(1 for value in flat_all if value not in (None, ""))
        if name == "MIN":
            return min(numbers) if numbers else 0
        if name == "MAX":
            return max(numbers) if numbers else 0
        if name == "ABS":
            return abs(to_number(args[0]))
        if name == "ROUND":
            return _round(to_number(args[0]), int(to_number(args[1])) if len(args) > 1 else 0)
        if name == "COUNTIF" or name == "SUMIF":
            if len(args) < 2:
                raise FormulaError("#VALUE!")
            test = args[0]
            criteria = args[1]
            total_range = args[2] if len(args) > 2 else test
            if not isinstance(test, RangeValue) or not isinstance(total_range, RangeValue):
                raise FormulaError("#VALUE!")
            total = 0
            totals = total_range.flatten()
            for index, value in enumerate(test.flatten()):
                if _match_criteria(value, criteria):
                    if name == "COUNTIF":
                        total += 1
                    elif index < len(totals) and is_number(totals[index]):
                        total += totals[index]
            return total
        if name == "AND" or name == "OR":
            return False
        if name in ("VLOOKUP", "XLOOKUP", "INDEX", "MATCH", "HLOOKUP"):
            return self.lookup(name, args)
        if name in ("UPPER", "LOWER", "PROPER", "TRIM", "CLEAN", "LEN"):
            text = to_text(args[0]) if args else ""
            if name == "UPPER":
                return text.upper()
            if name == "LOWER":
                return text.lower()
            if name == "PROPER":
                return text.title()
            if name == "TRIM":
                return re.sub(r"\s+", " ", text).strip()
            if name == "CLEAN":
                return re.sub(r"[\x00-\x1f\x7f]", "", text)
            return len(text)
        if name in ("LEFT", "RIGHT"):
            text = to_text(args[0]) if args else ""
            count = int(to_number(args[1])) if len(args) > 1 and args[1] not in (None, "") else 1
            count = max(0, count)
            return text[:count] if name == "LEFT" else (text[-count:] if count else "")
        if name == "MID":
            text = to_text(args[0])
            start = max(1, int(to_number(args[1])))
            count = max(0, int(to_number(args[2])))
            return text[start - 1:start - 1 + count]
        if name == "SUBSTITUTE":
            text, old, new = to_text(args[0]), to_text(args[1]), to_text(args[2])
            if not old:
                return text
            if len(args) > 3:
                instance = int(to_number(args[3]))
                if instance < 1:
                    return text
                position = -1
                for _ in range(instance):
                    position = text.find(old, position + 1)
                    if position < 0:
                        return text
                return text[:position] + new + text[position + len(old):]
            return text.replace(old, new)
        if name == "TEXT":
            return format_mixed(args[0], to_text(args[1])) if len(args) > 1 else to_text(args[0])
        if name == "VALUE":
            return to_number(args[0])
        if name == "DATE":
            try:
                return date(int(to_number(args[0])), int(to_number(args[1])),
                            int(to_number(args[2]))).isoformat()
            except ValueError:
                raise FormulaError("#VALUE!")
        if name in ("YEAR", "MONTH", "DAY"):
            day = _parse_date(args[0]) if args else None
            if not day:
                raise FormulaError("#VALUE!")
            return {"YEAR": day.year, "MONTH": day.month, "DAY": day.day}[name]
        if name == "TODAY":
            return date.today().isoformat()
        if name in ("CONCAT", "CONCATENATE"):
            return "".join(to_text(value) for value in flat_all)
        if name == "TEXTJOIN":
            delimiter = to_text(args[0])
            ignore_empty = to_bool(args[1])
            values = args[2:]
            flat = []
            for value in values:
                flat.extend(value.flatten() if isinstance(value, RangeValue) else [value])
            texts = [to_text(value) for value in flat if not (ignore_empty and value in (None, ""))]
            return delimiter.join(texts)
        raise FormulaError("#NAME?")

    def lookup(self, name: str, args):
        if name == "INDEX":
            if len(args) < 2 or not isinstance(args[0], RangeValue):
                raise FormulaError("#VALUE!")
            row = int(to_number(args[1]))
            col = int(to_number(args[2])) if len(args) > 2 else 1
            rows = args[0].rows
            if not 1 <= row <= len(rows) or not 1 <= col <= len(rows[0]):
                raise FormulaError("#REF!")
            return rows[row - 1][col - 1]
        if name == "MATCH":
            if len(args) < 2 or not isinstance(args[1], RangeValue):
                raise FormulaError("#VALUE!")
            needle = args[0]
            haystack = args[1].flatten()
            match_type = int(to_number(args[2])) if len(args) > 2 else 1
            for index, value in enumerate(haystack):
                if _compare(value, needle) == 0:
                    return index + 1
            if match_type == 0:
                raise FormulaError("#N/A")
            best = None
            for index, value in enumerate(haystack):
                relation = _compare(value, needle)
                if match_type > 0 and relation <= 0:
                    best = index + 1
                if match_type < 0 and relation >= 0:
                    best = index + 1
            if best is None:
                raise FormulaError("#N/A")
            return best
        if name in ("VLOOKUP", "HLOOKUP"):
            if len(args) < 3 or not isinstance(args[1], RangeValue):
                raise FormulaError("#VALUE!")
            needle = args[0]
            table = args[1].rows
            index = int(to_number(args[2])) - 1
            exact = len(args) < 4 or not to_bool(args[3])
            if name == "VLOOKUP":
                for row in table:
                    if index < len(row) and _compare(row[0], needle) == 0:
                        return row[index]
            else:
                header = table[0] if table else []
                for col_index, value in enumerate(header):
                    if col_index == 0 and _compare(value, needle) == 0 and index < len(table):
                        return table[index][col_index]
            if exact:
                raise FormulaError("#N/A")
            raise FormulaError("#N/A")
        if name == "XLOOKUP":
            if len(args) < 3:
                raise FormulaError("#VALUE!")
            needle = args[0]
            lookup_range = args[1]
            return_range = args[2]
            if not isinstance(lookup_range, RangeValue) or not isinstance(return_range, RangeValue):
                raise FormulaError("#VALUE!")
            haystack = lookup_range.flatten()
            results = return_range.flatten()
            for index, value in enumerate(haystack):
                if _compare(value, needle) == 0:
                    return results[index] if index < len(results) else None
            if len(args) > 3:
                return args[3]
            raise FormulaError("#N/A")
        raise FormulaError("#NAME?")


def compute(sheets: dict[str, dict]) -> dict:
    """Return computed values for every formula cell in the workbook."""
    evaluator = Evaluator(sheets)
    results = {}
    for sheet_name, grid in sheets.items():
        for (row, col), cell in grid.items():
            if not cell.get("f"):
                continue
            try:
                results[(sheet_name, row, col)] = evaluator.get_cell(sheet_name, row, col)
            except FormulaError as error:
                results[(sheet_name, row, col)] = error.code
            except Exception:  # noqa: BLE001 - a bad formula cannot break saving
                results[(sheet_name, row, col)] = "#VALUE!"
    for key, value in list(results.items()):
        results[key] = display_value(value)
    return results


def display_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    return value


def numeric_value(value) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None

==============================================================================
FILE: datafix/importer.py
==============================================================================
"""Read XLSX, XLSM, XLS, CSV and ODS files into the DataFix document model.

The original file is never modified; this module only produces an in-memory
document that is then written to the SQLite store.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import re
import zipfile
from pathlib import Path

from . import config

MAX_ROWS = 200000
MAX_COLS = 500
MIN_ROWS = 100
MIN_COLS = 20


def _clean_sheet_name(raw: str, index: int, used: set[str]) -> str:
    name = re.sub(r"[\[\]:*?/\\]", " ", str(raw or "")).strip()[:31] or f"Sheet{index}"
    candidate = name
    suffix = 2
    while candidate.lower() in used:
        tail = f" ({suffix})"
        candidate = name[: 31 - len(tail)] + tail
        suffix += 1
    used.add(candidate.lower())
    return candidate


def _column_number(letters: str) -> int:
    value = 0
    for char in str(letters).upper():
        if "A" <= char <= "Z":
            value = value * 26 + ord(char) - 64
    return value


def _blank_doc(name: str = "Sheet1", rows: int = 80, cols: int = 20) -> dict:
    return {
        "sheets": [{
            "name": name,
            "cells": {},
            "merges": [],
            "hidden_rows": [],
            "hidden_cols": [],
            "row_count": max(rows, MIN_ROWS),
            "col_count": max(cols, MIN_COLS),
        }],
        "truncated": False,
    }


def _dimensions(cells: dict, min_rows: int = MIN_ROWS, min_cols: int = MIN_COLS) -> tuple[int, int]:
    if not cells:
        return min_rows, min_cols
    rows = max(row for row, _col in cells)
    cols = max(col for _row, col in cells)
    return min(MAX_ROWS, max(min_rows, rows + 50)), min(MAX_COLS, max(min_cols, cols + 8))


def _argb(value) -> str | None:
    try:
        if value is None:
            return None
        rgb = getattr(value, "rgb", None)
        if isinstance(rgb, str):
            text = rgb.upper()
            if text in ("00000000", "FFFFFFFF"):
                return None
            if re.fullmatch(r"[0-9A-F]{8}", text):
                return "#" + text[2:]
            if re.fullmatch(r"[0-9A-F]{6}", text):
                return "#" + text
        if isinstance(value, str):
            text = value.upper().lstrip("#")
            if re.fullmatch(r"[0-9A-F]{8}", text):
                return "#" + text[2:]
            if re.fullmatch(r"[0-9A-F]{6}", text):
                return "#" + text
        if getattr(value, "type", None) == "theme":
            return None
    except Exception:  # noqa: BLE001 - colour objects vary by version
        return None
    return None


def _style_from_cell(cell) -> dict | None:
    style: dict = {}
    try:
        font = cell.font
        if font:
            if font.bold:
                style["bold"] = True
            if font.italic:
                style["italic"] = True
            underline = getattr(font, "underline", None)
            if underline and underline != "none":
                style["underline"] = True
            if font.strike:
                style["strike"] = True
            colour = _argb(getattr(font, "color", None))
            if colour:
                style["color"] = colour
            if font.name and font.name not in ("Calibri", "Arial"):
                style["font"] = font.name
            if font.sz and float(font.sz) not in (11, 12):
                style["size"] = float(font.sz)
        fill = cell.fill
        if fill and getattr(fill, "fill_type", None) not in (None, "none"):
            colour = _argb(getattr(fill, "fgColor", None))
            if colour:
                style["fill"] = colour
        alignment = cell.alignment
        if alignment:
            if alignment.horizontal and alignment.horizontal != "general":
                style["align"] = alignment.horizontal
            if alignment.vertical and alignment.vertical != "bottom":
                style["valign"] = alignment.vertical
            if alignment.wrap_text:
                style["wrap"] = True
        number_format = cell.number_format
        if number_format and number_format != "General":
            style["numfmt"] = number_format
    except Exception:  # noqa: BLE001 - never fail the import over styling
        return style or None
    return style or None


def _value_of(raw):
    if raw is None:
        return None
    if isinstance(raw, (dt.datetime, dt.date, dt.time)):
        if isinstance(raw, dt.datetime):
            return raw.strftime("%Y-%m-%d %H:%M:%S") if (raw.hour or raw.minute) else raw.strftime("%Y-%m-%d")
        if isinstance(raw, dt.time):
            return raw.strftime("%H:%M:%S")
        return raw.isoformat()
    if isinstance(raw, dt.timedelta):
        return raw.total_seconds() / 86400
    if isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and raw.is_integer() and abs(raw) < 1e15:
            return int(raw)
        return raw
    return str(raw)


def _import_xlsx(path: Path) -> dict:
    from openpyxl import load_workbook

    workbook = load_workbook(path, data_only=False)
    cached_book = None
    try:
        cached_book = load_workbook(path, data_only=True)
    except Exception:  # noqa: BLE001 - cached values are optional
        cached_book = None

    doc = {"sheets": [], "truncated": False}
    used: set[str] = set()
    for index, sheet in enumerate(workbook.worksheets, start=1):
        name = _clean_sheet_name(sheet.title, index, used)
        cells: dict = {}
        cached_sheet = None
        if cached_book is not None:
            try:
                cached_sheet = cached_book[sheet.title]
            except KeyError:
                cached_sheet = None
        count = 0
        truncated = False
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is None and cell.style_id == 0:
                    continue
                if count >= config.MAX_CELLS_PER_SHEET:
                    truncated = True
                    break
                row_index, col_index = cell.row, cell.column
                if row_index > MAX_ROWS or col_index > MAX_COLS:
                    truncated = True
                    continue
                raw = cell.value
                entry: dict = {}
                if isinstance(raw, str) and raw.startswith("=") and len(raw) > 1:
                    entry["f"] = raw[:4000]
                    cached = cached_sheet.cell(row=row_index, column=col_index).value if cached_sheet else None
                    if cached is not None:
                        entry["cache"] = _value_of(cached)
                else:
                    value = _value_of(raw)
                    if value is not None:
                        entry["v"] = value
                style = _style_from_cell(cell)
                if style:
                    entry["s"] = style
                if entry:
                    cells[(row_index, col_index)] = entry
                    count += 1
            if truncated:
                break
        doc["truncated"] = doc["truncated"] or truncated
        merges = []
        for merged in getattr(sheet, "merged_cells", []):
            try:
                merges.append({
                    "r1": merged.min_row, "c1": merged.min_col,
                    "r2": merged.max_row, "c2": merged.max_col,
                })
            except AttributeError:
                continue
        rows, cols = _dimensions(cells)
        doc["sheets"].append({
            "name": name,
            "cells": cells,
            "merges": merges,
            "hidden_rows": sorted(int(index) for index, dim in sheet.row_dimensions.items()
                                  if getattr(dim, "hidden", False)),
            "hidden_cols": sorted(n for n in (_column_number(key) for key, dim
                                   in sheet.column_dimensions.items()
                                   if getattr(dim, "hidden", False)) if n),
            "row_count": max(rows, len(sheet.row_dimensions) + 10),
            "col_count": cols,
        })
    if not doc["sheets"]:
        raise ValueError("The workbook has no worksheets.")
    return doc


def _import_xls(path: Path) -> dict:
    import xlrd

    book = xlrd.open_workbook(str(path), formatting_info=False)
    doc = {"sheets": [], "truncated": False}
    used: set[str] = set()
    for index, sheet in enumerate(book.sheets(), start=1):
        name = _clean_sheet_name(sheet.name, index, used)
        cells: dict = {}
        count = 0
        truncated = False
        for row_index in range(sheet.nrows):
            if truncated:
                break
            for col_index in range(sheet.ncols):
                if count >= config.MAX_CELLS_PER_SHEET:
                    truncated = True
                    break
                cell_type = sheet.cell_type(row_index, col_index)
                if cell_type in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
                    continue
                value = sheet.cell_value(row_index, col_index)
                if cell_type == xlrd.XL_CELL_DATE:
                    try:
                        stamp = xlrd.xldate.xldate_as_datetime(value, book.datemode)
                        value = stamp.strftime("%Y-%m-%d %H:%M:%S") if (stamp.hour or stamp.minute) else stamp.strftime("%Y-%m-%d")
                    except (ValueError, OverflowError):
                        value = _value_of(value)
                elif cell_type == xlrd.XL_CELL_ERROR:
                    value = f"#ERR{int(value)}"
                else:
                    value = _value_of(value)
                if value is None or value == "":
                    continue
                cells[(row_index + 1, col_index + 1)] = {"v": value}
                count += 1
        doc["truncated"] = doc["truncated"] or truncated
        rows, cols = _dimensions(cells)
        doc["sheets"].append({
            "name": name, "cells": cells, "merges": [],
            "hidden_rows": [], "hidden_cols": [],
            "row_count": rows, "col_count": cols,
        })
    if not doc["sheets"]:
        raise ValueError("The workbook has no worksheets.")
    return doc


def _import_csv(path: Path) -> dict:
    text = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("The CSV file could not be decoded.")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows_raw = list(csv.reader(io.StringIO(text), dialect))
    rows_raw = [row for row in rows_raw if any(str(cell).strip() for cell in row)]
    if not rows_raw:
        raise ValueError("No data rows were found in the CSV file.")
    cells: dict = {}
    truncated = False
    for row_index, row in enumerate(rows_raw, start=1):
        if row_index > MAX_ROWS:
            truncated = True
            break
        for col_index, raw in enumerate(row, start=1):
            if col_index > MAX_COLS:
                truncated = True
                break
            if raw is None:
                continue
            value = raw.strip()
            if value == "":
                continue
            number = None
            try:
                number = float(value.replace(",", ""))
            except ValueError:
                number = None
            if (number is not None
                    and re.fullmatch(r"[-+]?\d{1,3}(,\d{3})*(\.\d+)?|[-+]?\d+(\.\d+)?", value)
                    and not (value.startswith("0") and len(value) > 1 and value.isdigit())):
                value = int(number) if number.is_integer() and abs(number) < 1e15 else number
            cells[(row_index, col_index)] = {"v": value}
    rows, cols = _dimensions(cells)
    doc = {
        "sheets": [{
            "name": _clean_sheet_name(path.stem or "CSV", 1, set()), "cells": cells,
            "merges": [], "hidden_rows": [], "hidden_cols": [],
            "row_count": rows, "col_count": cols,
        }],
        "truncated": truncated,
    }
    return doc


def _import_ods(path: Path) -> dict:
    from odf import table, teletype
    from odf.namespaces import OFFICENS, TABLENS
    from odf.opendocument import load

    document = load(str(path))
    sheets = document.getElementsByType(table.Table)
    if not sheets:
        raise ValueError("The ODS file has no worksheets.")
    doc = {"sheets": [], "truncated": False}
    used: set[str] = set()
    for index, sheet in enumerate(sheets, start=1):
        name = _clean_sheet_name(sheet.getAttribute("name"), index, used)
        cells: dict = {}
        row_index = 0
        truncated = False
        for row in sheet.getElementsByType(table.TableRow):
            repeat_rows = int(row.getAttribute("numberrowsrepeated") or 1)
            row_index += repeat_rows
            col_index = 0
            for cell in row.getElementsByType(table.TableCell):
                repeat_cols = int(cell.getAttribute("numbercolumnsrepeated") or 1)
                cell_type = cell.getAttribute("valuetype")
                value = None
                formula = cell.getAttribute("formula")
                if cell_type == "float" or cell_type == "percentage" or cell_type == "currency":
                    raw = cell.getAttribute("value")
                    if raw not in (None, ""):
                        number = float(raw)
                        value = int(number) if number.is_integer() else number
                elif cell_type == "date":
                    value = cell.getAttribute("datevalue")
                elif cell_type == "boolean":
                    value = cell.getAttribute("booleanvalue") == "true"
                elif cell_type == "time":
                    value = cell.getAttribute("timevalue")
                else:
                    text = teletype.extractText(cell).strip()
                    if text:
                        value = text
                if value is not None or formula:
                    for offset in range(min(repeat_cols, MAX_COLS - col_index)):
                        entry = {}
                        if value is not None:
                            entry["v"] = value
                        if formula:
                            entry["f"] = "=" + formula.lstrip("=")
                        if not entry:
                            continue
                        cells[(row_index, col_index + offset + 1)] = entry
                col_index += repeat_cols
                if col_index > MAX_COLS:
                    truncated = True
                    break
            if row_index > MAX_ROWS:
                truncated = True
                break
            if len(cells) > config.MAX_CELLS_PER_SHEET:
                truncated = True
                break
        doc["truncated"] = doc["truncated"] or truncated
        rows, cols = _dimensions(cells)
        doc["sheets"].append({
            "name": name, "cells": cells, "merges": [],
            "hidden_rows": [], "hidden_cols": [],
            "row_count": rows, "col_count": cols,
        })
    return doc


def import_file(path: Path, extension: str) -> dict:
    extension = extension.lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type.")
    if not _looks_valid(path, extension):
        raise ValueError("This file does not look like a valid workbook of that type.")
    if extension in (".xlsx", ".xlsm"):
        return _import_xlsx(path)
    if extension == ".xls":
        return _import_xls(path)
    if extension == ".csv":
        return _import_csv(path)
    return _import_ods(path)


def _looks_valid(path: Path, extension: str) -> bool:
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
    except OSError:
        return False
    if extension in (".xlsx", ".xlsm", ".ods"):
        return head.startswith(b"PK\x03\x04")
    if extension == ".xls":
        return head.startswith(b"\xd0\xcf\x11\xe0")
    return True

==============================================================================
FILE: datafix/workbooks.py
==============================================================================
"""Permanent workbook service: storage, editing, versions and recovery."""

from __future__ import annotations

import gzip
import json
import math
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from . import config, db, formulas, importer

CELL_LIMIT = 40000


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _uid() -> str:
    return uuid.uuid4().hex


def _safe_name(name: str, fallback: str = "Untitled Workbook") -> str:
    text = re.sub(r"[\x00-\x1f<>:\"/\\|?*]+", " ", str(name or "")).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:120] or fallback


def _row_to_meta(row, sheet_rows=None) -> dict:
    meta = {
        "id": row["id"],
        "name": row["name"],
        "source": row["source"],
        "original_name": row["original_name"],
        "extension": row["extension"],
        "starred": bool(row["starred"]),
        "archived": bool(row["archived"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "opened_at": row["opened_at"],
        "analysis_at": row["analysis_at"],
        "analysis_stale": bool(row["analysis_stale"]),
        "version_count": row["version_count"],
        "save_count": row["save_count"],
    }
    sheets = list(sheet_rows or [])
    meta["sheets"] = sheets
    meta["sheet_count"] = len(sheets)
    if sheets:
        meta["row_count"] = max(sheet["row_count"] for sheet in sheets)
        meta["col_count"] = max(sheet["col_count"] for sheet in sheets)
    else:
        meta["row_count"] = 0
        meta["col_count"] = 0
    meta["storage_root"] = row["storage_root"]
    return meta


def _sheet_meta(row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "position": row["position"],
        "row_count": row["row_count"],
        "col_count": row["col_count"],
        "frozen_rows": row["frozen_rows"],
        "frozen_cols": row["frozen_cols"],
        "hidden_rows": json.loads(row["hidden_rows"] or "[]"),
        "hidden_cols": json.loads(row["hidden_cols"] or "[]"),
        "merges": json.loads(row["merges"] or "[]"),
    }


def _load_sheets(conn, wb_id) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM sheets WHERE workbook_id=? ORDER BY position, id", (wb_id,)
    ).fetchall()
    return [_sheet_meta(row) for row in rows]


def _workbook(conn, wb_id) -> dict | None:
    row = conn.execute("SELECT * FROM workbooks WHERE id=?", (wb_id,)).fetchone()
    if not row:
        return None
    return _row_to_meta(row, _load_sheets(conn, wb_id))


def get_workbook(wb_id: str) -> dict | None:
    with db.read() as conn:
        return _workbook(conn, wb_id)


def require_workbook(wb_id: str) -> dict:
    workbook = get_workbook(wb_id)
    if not workbook:
        raise LookupError("Workbook not found.")
    return workbook


def _unique_name(conn, name: str) -> str:
    base = _safe_name(name)
    candidate = base
    index = 2
    while conn.execute(
        "SELECT 1 FROM workbooks WHERE name=? AND archived=0", (candidate,)
    ).fetchone():
        candidate = f"{base} ({index})"
        index += 1
    return candidate


def list_workbooks(scope: str = "all", query: str = "") -> list[dict]:
    with db.read() as conn:
        sql = "SELECT * FROM workbooks"
        clauses = []
        params: list = []
        if scope == "archived":
            clauses.append("archived=1")
        else:
            clauses.append("archived=0")
        if scope == "starred":
            clauses.append("starred=1")
        if query.strip():
            clauses.append("(name LIKE ? OR original_name LIKE ?)")
            needle = f"%{query.strip()}%"
            params.extend([needle, needle])
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY opened_at DESC, updated_at DESC"
        rows = conn.execute(sql + " LIMIT 500", params).fetchall()
        result = []
        for row in rows:
            meta = _row_to_meta(row, _load_sheets(conn, row["id"]))
            meta.pop("storage_root", None)
            cells = conn.execute(
                "SELECT COALESCE(SUM(s.row_count * s.col_count), 0) AS total "
                "FROM sheets s WHERE s.workbook_id=?", (row["id"],)
            ).fetchone()
            meta["editable_cells"] = cells["total"] if cells else 0
            used = conn.execute(
                "SELECT COUNT(DISTINCT c.row) AS total FROM cells c"
                " JOIN sheets s ON s.id=c.sheet_id"
                " WHERE s.workbook_id=? AND c.row>1"
                " AND c.value IS NOT NULL AND TRIM(c.value)<>''", (row["id"],)
            ).fetchone()
            meta["record_count"] = used["total"] if used else 0
            result.append(meta)
        return result


def empty_listing() -> dict:
    return {"workbooks": list_workbooks("all"), "recent": list_workbooks("recent")}


# --------------------------------------------------------------------------- #
# Creation
# --------------------------------------------------------------------------- #

def _insert_workbook(conn, name, source, original_name, extension, root) -> str:
    wb_id = _uid()
    now = _now()
    conn.execute(
        "INSERT INTO workbooks (id, name, source, original_name, extension, storage_root,"
        " created_at, updated_at, opened_at, analysis_stale) VALUES (?,?,?,?,?,?,?,?,?,1)",
        (wb_id, name, source, original_name, extension, str(root), now, now, now),
    )
    return wb_id


def _insert_sheet(conn, wb_id, sheet: dict, position: int, sheet_id: int | None = None) -> int:
    if sheet_id:
        conn.execute(
            "INSERT INTO sheets (id, workbook_id, name, position, row_count, col_count,"
            " hidden_rows, hidden_cols, merges) VALUES (?,?,?,?,?,?,?,?,?)",
            (int(sheet_id), wb_id, sheet["name"], position, sheet.get("row_count", 100),
             sheet.get("col_count", 20), json.dumps(sheet.get("hidden_rows") or []),
             json.dumps(sheet.get("hidden_cols") or []),
             json.dumps(sheet.get("merges") or [])),
        )
    else:
        cursor = conn.execute(
            "INSERT INTO sheets (workbook_id, name, position, row_count, col_count,"
            " hidden_rows, hidden_cols, merges) VALUES (?,?,?,?,?,?,?,?)",
            (wb_id, sheet["name"], position, sheet.get("row_count", 100),
             sheet.get("col_count", 20), json.dumps(sheet.get("hidden_rows") or []),
             json.dumps(sheet.get("hidden_cols") or []),
             json.dumps(sheet.get("merges") or [])),
        )
        sheet_id = cursor.lastrowid
    cells = sheet.get("cells") or {}
    if cells:
        payload = []
        for key, cell in cells.items():
            if isinstance(key, str):
                row, col = key.split(",")
            else:
                row, col = key
            source = cell.get("v")
            if source is None and cell.get("f") and cell.get("cache") is not None:
                source = cell.get("cache")
            value, num = _typed(source)
            style = cell.get("s")
            payload.append((
                sheet_id, int(row), int(col), value, num,
                cell.get("f"), json.dumps(style) if style else None,
            ))
        conn.executemany(
            "INSERT OR REPLACE INTO cells (sheet_id, row, col, value, num, formula, style)"
            " VALUES (?,?,?,?,?,?,?)", payload,
        )
    return sheet_id


def _typed(raw):
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, bool):
        return ("TRUE" if raw else "FALSE"), (1.0 if raw else 0.0)
    if isinstance(raw, (int, float)):
        number = float(raw)
        text = str(int(number)) if number.is_integer() and abs(number) < 1e15 else str(number)
        return text, number
    text = str(raw)
    try:
        number = float(text.replace(",", ""))
        if math.isfinite(number):
            return text, number
    except ValueError:
        pass
    return text, None


def create_blank(name: str | None = None) -> dict:
    root = config.get_storage_root()
    with db.transaction() as conn:
        final_name = _unique_name(conn, name or _next_untitled(conn))
        wb_id = _insert_workbook(conn, final_name, "blank", None, None, root)
        (root / "workbooks" / wb_id).mkdir(parents=True, exist_ok=True)
        _insert_sheet(conn, wb_id, {
            "name": "Sheet1", "cells": {}, "row_count": 100, "col_count": 20,
        }, 0)
        create_version(conn, wb_id, "Created", force=True)
        workbook = _workbook(conn, wb_id)
    return workbook


def _next_untitled(conn) -> str:
    index = 1
    while conn.execute(
        "SELECT 1 FROM workbooks WHERE name=?", (f"Untitled Workbook {index}",)
    ).fetchone():
        index += 1
    return f"Untitled Workbook {index}"


def create_from_file(saved_path: Path, filename: str, extension: str,
                     name: str | None = None) -> dict:
    """Import an already-saved original file into a new permanent workbook."""
    document = importer.import_file(saved_path, extension)
    root = config.get_storage_root()
    wb_id = None
    try:
        with db.transaction() as conn:
            base_name = name or Path(filename).stem
            final_name = _unique_name(conn, base_name)
            wb_id = _insert_workbook(conn, final_name, "upload", filename, extension, root)
            folder = root / "workbooks" / wb_id
            folder.mkdir(parents=True, exist_ok=True)
            original = folder / f"original{extension}"
            shutil.copy2(saved_path, original)
            for position, sheet in enumerate(document["sheets"]):
                _insert_sheet(conn, wb_id, sheet, position)
            _recalculate(conn, wb_id)
            create_version(conn, wb_id, "Imported", force=True)
            conn.execute("UPDATE workbooks SET save_count=0 WHERE id=?", (wb_id,))
            workbook = _workbook(conn, wb_id)
    except Exception:
        if wb_id:
            shutil.rmtree(root / "workbooks" / wb_id, ignore_errors=True)
        raise
    workbook["truncated"] = document.get("truncated", False)
    return workbook


def create_from_stream(stream, filename: str, name: str | None = None) -> dict:
    filename = Path(filename or "workbook.xlsx").name
    extension = Path(filename).suffix.lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Use .xlsx, .xls, .xlsm, .csv or .ods.")
    temp_dir = config.get_storage_root() / "workbooks" / ".incoming"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4().hex}{extension}"
    if hasattr(stream, "save"):
        stream.save(temp_path)
    else:
        with open(temp_path, "wb") as target:
            shutil.copyfileobj(stream, target)
    try:
        if temp_path.stat().st_size == 0:
            raise ValueError("The file is empty.")
        return create_from_file(temp_path, filename, extension, name)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def import_local_path(raw_path: str) -> dict:
    """Import a workbook from any permitted local path (desktop app feature)."""
    path = Path(str(raw_path).strip().strip('"'))
    if not path.is_absolute():
        raise ValueError("Use a full path to the workbook.")
    path = path.resolve()
    if not path.is_file():
        raise ValueError("That file does not exist.")
    if path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type.")
    if path.stat().st_size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"The file exceeds the {config.MAX_UPLOAD_MB} MB limit.")
    return create_from_file(path, path.name, path.suffix.lower())


def duplicate_workbook(wb_id: str, name: str | None = None) -> dict:
    source = require_workbook(wb_id)
    doc = snapshot_doc(wb_id)
    root = config.get_storage_root()
    with db.transaction() as conn:
        base = name or f"{source['name']} copy"
        final_name = _unique_name(conn, base)
        new_id = _insert_workbook(conn, final_name, "copy", source["original_name"],
                                  source["extension"], root)
        folder = root / "workbooks" / new_id
        folder.mkdir(parents=True, exist_ok=True)
        for position, sheet in enumerate(doc["sheets"]):
            _insert_sheet(conn, new_id, sheet, position)
        _recalculate(conn, new_id)
        create_version(conn, new_id, "Copied", force=True)
        return _workbook(conn, new_id)


# --------------------------------------------------------------------------- #
# Metadata operations
# --------------------------------------------------------------------------- #

def touch_opened(wb_id: str) -> None:
    with db.transaction() as conn:
        conn.execute("UPDATE workbooks SET opened_at=? WHERE id=?", (_now(), wb_id))


def recalculate(wb_id: str) -> dict:
    with db.transaction() as conn:
        if not conn.execute("SELECT 1 FROM workbooks WHERE id=?", (wb_id,)).fetchone():
            raise LookupError("Workbook not found.")
        _recalculate(conn, wb_id)
        conn.execute("UPDATE workbooks SET updated_at=? WHERE id=?", (_now(), wb_id))
        return {
            "workbook": _workbook(conn, wb_id),
            "saved_at": _now(),
            "version": conn.execute(
                "SELECT COALESCE(MAX(number), 0) AS n FROM versions WHERE workbook_id=?",
                (wb_id,),
            ).fetchone()["n"],
        }


def rename_workbook(wb_id: str, name: str) -> dict:
    if not name.strip():
        raise ValueError("A workbook needs a name.")
    with db.transaction() as conn:
        if not conn.execute("SELECT 1 FROM workbooks WHERE id=?", (wb_id,)).fetchone():
            raise LookupError("Workbook not found.")
        conn.execute(
            "UPDATE workbooks SET name=?, updated_at=? WHERE id=?",
            (_safe_name(name), _now(), wb_id),
        )
        return _workbook(conn, wb_id)


def set_flags(wb_id: str, starred=None, archived=None) -> dict:
    with db.transaction() as conn:
        if not conn.execute("SELECT 1 FROM workbooks WHERE id=?", (wb_id,)).fetchone():
            raise LookupError("Workbook not found.")
        if starred is not None:
            conn.execute("UPDATE workbooks SET starred=? WHERE id=?",
                         (1 if starred else 0, wb_id))
        if archived is not None:
            conn.execute("UPDATE workbooks SET archived=?, updated_at=? WHERE id=?",
                         (1 if archived else 0, _now(), wb_id))
        return _workbook(conn, wb_id)


def delete_workbook(wb_id: str) -> dict:
    """Permanently remove one workbook, its data and its own stored files."""
    workbook = require_workbook(wb_id)
    root = Path(workbook["storage_root"])
    with db.transaction() as conn:
        conn.execute(
            "DELETE FROM cells WHERE sheet_id IN"
            " (SELECT id FROM sheets WHERE workbook_id=?)", (wb_id,)
        )
        conn.execute("DELETE FROM sheets WHERE workbook_id=?", (wb_id,))
        for table in ("batches", "versions", "exports", "edits_log", "analyses"):
            conn.execute(f"DELETE FROM {table} WHERE workbook_id=?", (wb_id,))  # noqa: S608
        conn.execute("DELETE FROM workbooks WHERE id=?", (wb_id,))
    for sub in config.SUBDIRS:
        shutil.rmtree(root / sub / wb_id, ignore_errors=True)
    return {"id": wb_id, "name": workbook["name"]}


def mark_analyzed(wb_id: str, stale: bool = False) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE workbooks SET analysis_at=?, analysis_stale=? WHERE id=?",
            (_now() if not stale else None, 1 if stale else 0, wb_id),
        )


def mark_stale(wb_id: str) -> None:
    with db.transaction() as conn:
        conn.execute("UPDATE workbooks SET analysis_stale=1 WHERE id=?", (wb_id,))


def workbook_folder(wb_id: str) -> Path | None:
    workbook = get_workbook(wb_id)
    if not workbook:
        return None
    return Path(workbook["storage_root"]) / "workbooks" / wb_id


def original_file(wb_id: str) -> Path | None:
    workbook = get_workbook(wb_id)
    if not workbook or not workbook.get("extension"):
        return None
    path = Path(workbook["storage_root"]) / "workbooks" / wb_id / f"original{workbook['extension']}"
    return path if path.exists() else None


# --------------------------------------------------------------------------- #
# Sheet / grid access
# --------------------------------------------------------------------------- #

def _fetch_sheet(conn, wb_id, sheet_id=None):
    if sheet_id:
        row = conn.execute(
            "SELECT * FROM sheets WHERE id=? AND workbook_id=?", (int(sheet_id), wb_id)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM sheets WHERE workbook_id=? ORDER BY position, id LIMIT 1", (wb_id,)
        ).fetchone()
    if not row:
        raise LookupError("Worksheet not found.")
    return row


def sheet_id_by_name(wb_id: str, name: str) -> int | None:
    with db.read() as conn:
        row = conn.execute(
            "SELECT id FROM sheets WHERE workbook_id=? AND lower(name)=lower(?)",
            (wb_id, name),
        ).fetchone()
    return row["id"] if row else None


def _cell_text(value) -> str:
    return "" if value is None else str(value)


def _load_cells(conn, sheet_id: int) -> dict:
    rows = conn.execute(
        "SELECT row, col, value, num, formula, style FROM cells WHERE sheet_id=?", (sheet_id,)
    ).fetchall()
    cells = {}
    for row in rows:
        cells[(row["row"], row["col"])] = {
            "v": row["value"],
            "num": row["num"],
            "f": row["formula"],
            "s": json.loads(row["style"]) if row["style"] else None,
        }
    return cells


def _filter_rows(cells: dict, filters: list[dict], row_count: int,
                 has_header: bool = True) -> list[int]:
    if not filters:
        return list(range(1, row_count + 1))
    visible = []
    for row in range(1, row_count + 1):
        if has_header and row == 1:
            visible.append(row)
            continue
        keep = True
        for spec in filters:
            col = int(spec.get("col") or 0)
            if col < 1:
                continue
            cell = cells.get((row, col))
            value = _cell_text(cell["v"] if cell else None)
            if not _match_filter(value, spec, cells, col, row):
                keep = False
                break
        if keep:
            visible.append(row)
    return visible


def _match_filter(value: str, spec: dict, cells: dict, col: int, row: int) -> bool:
    op = str(spec.get("op") or "contains").lower()
    needle = str(spec.get("value") if spec.get("value") is not None else "")
    value_l = value.lower()
    needle_l = needle.lower()
    if op == "contains":
        return needle_l in value_l
    if op == "not_contains":
        return needle_l not in value_l
    if op == "equals":
        return value_l == needle_l
    if op == "not_equals":
        return value_l != needle_l
    if op == "starts":
        return value_l.startswith(needle_l)
    if op == "ends":
        return value_l.endswith(needle_l)
    if op == "blank":
        return value.strip() == ""
    if op == "not_blank":
        return value.strip() != ""
    if op == "duplicates":
        return _is_duplicate(cells, col, row, value)
    if op == "unique":
        return not _is_duplicate(cells, col, row, value)
    if op in ("gt", "lt", "gte", "lte", "between"):
        try:
            numeric = float(value.replace(",", ""))
            target = float(str(spec.get("value") or 0).replace(",", ""))
        except ValueError:
            return False
        if op == "gt":
            return numeric > target
        if op == "lt":
            return numeric < target
        if op == "gte":
            return numeric >= target
        if op == "lte":
            return numeric <= target
        try:
            upper = float(str(spec.get("value2") or 0).replace(",", ""))
        except ValueError:
            upper = target
        low, high = sorted((target, upper))
        return low <= numeric <= high
    return True


def _is_duplicate(cells: dict, col: int, row: int, value: str) -> bool:
    if value.strip() == "":
        return False
    lowered = value.strip().lower()
    for (other_row, other_col), cell in cells.items():
        if other_col != col or other_row == row or other_row == 1:
            continue
        if _cell_text(cell.get("v")).strip().lower() == lowered:
            return True
    return False


def _visible_rows(cells: dict, sheet_meta: dict, filters=None, sort=None,
                  search: str = "", include_hidden: bool = False) -> list[int]:
    hidden_rows = set() if include_hidden else set(sheet_meta["hidden_rows"])
    visible = [row for row in _filter_rows(cells, filters or [], sheet_meta["row_count"])
               if row not in hidden_rows]
    search_text = (search or "").strip().lower()
    if search_text:
        matching = set()
        for (row, _col), cell in cells.items():
            if (search_text in _cell_text(cell.get("v")).lower()
                    or search_text in _cell_text(cell.get("f")).lower()):
                matching.add(row)
        visible = [row for row in visible if row in matching]
    if sort and sort.get("col"):
        key_col = int(sort["col"])
        reverse = str(sort.get("direction", "asc")).lower() == "desc"
        header = [row for row in visible if row == 1]
        body = [row for row in visible if row != 1]

        def sort_key(row):
            cell = cells.get((row, key_col))
            value = cell.get("num") if cell and cell.get("num") is not None else _cell_text(
                cell.get("v") if cell else None)
            is_blank = value in (None, "")
            if isinstance(value, (int, float)):
                return (is_blank, 0, float(value), "")
            return (is_blank, 1, 0.0, str(value).lower())

        try:
            body.sort(key=sort_key, reverse=reverse)
        except TypeError:
            pass
        visible = header + body
    return visible


def read_grid(wb_id: str, sheet_id=None, start_row: int = 1, rows: int = 60,
              start_col: int = 1, cols: int = 30, filters=None, sort=None,
              search: str = "") -> dict:
    with db.read() as conn:
        sheet = _fetch_sheet(conn, wb_id, sheet_id)
        cells = _load_cells(conn, sheet["id"])
        sheet_meta = _sheet_meta(sheet)

    row_count = sheet_meta["row_count"]
    col_count = sheet_meta["col_count"]
    hidden_cols = set()
    for item in sheet_meta["hidden_cols"]:
        if isinstance(item, int):
            hidden_cols.add(item)
        elif isinstance(item, str) and item.isalpha():
            hidden_cols.add(0)

    visible_rows = _visible_rows(cells, sheet_meta, filters, sort, search)

    total = len(visible_rows)
    start_row = max(1, min(int(start_row or 1), max(1, total)))
    page_rows = visible_rows[start_row - 1:start_row - 1 + max(1, min(rows, 400))]
    end_col = min(col_count, start_col + max(1, min(cols, 60)) - 1)
    page_cols = [col for col in range(start_col, end_col + 1) if col not in hidden_cols]

    output_cells = {}
    for row in page_rows:
        for col in page_cols:
            cell = cells.get((row, col))
            if not cell:
                continue
            output_cells[f"{row}:{col}"] = {
                "v": cell["v"],
                "f": cell["f"],
                "s": cell["s"],
                "n": cell["num"],
            }

    headers = {}
    for col in range(1, col_count + 1):
        cell = cells.get((1, col))
        headers[str(col)] = _cell_text(cell["v"] if cell else None)

    return {
        "sheet": sheet_meta,
        "start_row": start_row,
        "rows": page_rows,
        "columns": page_cols,
        "total_rows": total,
        "row_count": row_count,
        "col_count": col_count,
        "cells": output_cells,
        "headers": headers,
        "hidden_cols": sorted(hidden_cols),
        "visible_from": start_row,
    }


def view_data(wb_id: str, sheet_id=None, filters=None, sort=None, search: str = "",
              rows=None, row_start=None, row_end=None, columns=None,
              include_hidden: bool = False) -> dict:
    """Full ordered view of one sheet for exporting / printing."""
    with db.read() as conn:
        sheet = _fetch_sheet(conn, wb_id, sheet_id)
        cells = _load_cells(conn, sheet["id"])
        sheet_meta = _sheet_meta(sheet)

    hidden_cols = {item for item in sheet_meta["hidden_cols"] if isinstance(item, int)}
    if rows:
        wanted = sorted({int(value) for value in rows})
        order = [row for row in _visible_rows(cells, sheet_meta, None, None, "",
                                             include_hidden) if row in wanted]
        missing = [row for row in wanted if row not in order]
        order.extend(missing)
    else:
        order = _visible_rows(cells, sheet_meta, filters, sort, search, include_hidden)
        if row_start or row_end:
            low = int(row_start or 1)
            high = int(row_end or 10 ** 9)
            order = [row for row in order if low <= row <= high]

    if columns:
        cols = sorted({int(value) for value in columns})
    else:
        cols = [col for col in range(1, sheet_meta["col_count"] + 1)
                if include_hidden or col not in hidden_cols]
    selected = {}
    for row in order:
        for col in cols:
            cell = cells.get((row, col))
            if cell:
                selected[(row, col)] = cell
    complete = (not filters and not sort and not search and not rows
                and not row_start and not row_end and not columns)
    return {
        "sheet": sheet_meta,
        "rows": order,
        "columns": cols,
        "cells": selected,
        "complete": complete,
    }


def search_in_workbook(wb_id: str, query: str, sheet_id=None, column: int = 0,
                       limit: int = 200) -> list[dict]:
    needle = f"%{(query or '').strip()}%"
    if not (query or "").strip():
        return []
    with db.read() as conn:
        sql = (
            "SELECT c.row, c.col, c.value, s.id AS sheet_id, s.name AS sheet_name "
            "FROM cells c JOIN sheets s ON s.id=c.sheet_id "
            "WHERE s.workbook_id=? AND c.value LIKE ?"
        )
        params: list = [wb_id, needle]
        if sheet_id:
            sql += " AND s.id=?"
            params.append(int(sheet_id))
        if column:
            sql += " AND c.col=?"
            params.append(int(column))
        sql += " ORDER BY s.position, c.row, c.col LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        results = [dict(row) for row in rows]

        if needle:
            headers = {}
            for result in results:
                key = (result["sheet_id"])
                if key not in headers:
                    header_rows = conn.execute(
                        "SELECT col, value FROM cells WHERE sheet_id=? AND row=1", (key,)
                    ).fetchall()
                    headers[key] = {h["col"]: h["value"] for h in header_rows}
                result["header"] = headers[key].get(result["col"]) or f"Column {result['col']}"
    return results


def sheet_data(wb_id: str, sheet_id=None) -> dict:
    with db.read() as conn:
        sheet = _fetch_sheet(conn, wb_id, sheet_id)
        return {"sheet": _sheet_meta(sheet), "cells": _load_cells(conn, sheet["id"])}


# --------------------------------------------------------------------------- #
# Snapshot / versions
# --------------------------------------------------------------------------- #

def snapshot_doc(wb_id: str) -> dict:
    with db.read() as conn:
        workbook = _workbook(conn, wb_id)
        if not workbook:
            raise LookupError("Workbook not found.")
        sheets = []
        for meta in workbook["sheets"]:
            cells = _load_cells(conn, meta["id"])
            sheets.append({
                "name": meta["name"],
                "cells": {f"{r},{c}": cell for (r, c), cell in cells.items()},
                "row_count": meta["row_count"],
                "col_count": meta["col_count"],
                "frozen_rows": meta["frozen_rows"],
                "frozen_cols": meta["frozen_cols"],
                "hidden_rows": meta["hidden_rows"],
                "hidden_cols": meta["hidden_cols"],
                "merges": meta["merges"],
            })
    workbook.pop("storage_root", None)
    return {"workbook": workbook, "sheets": sheets}


def create_version(conn, wb_id: str, label: str, force: bool = False) -> dict | None:
    workbook = conn.execute("SELECT * FROM workbooks WHERE id=?", (wb_id,)).fetchone()
    if not workbook:
        return None
    if not force:
        latest = conn.execute(
            "SELECT created_at FROM versions WHERE workbook_id=? ORDER BY number DESC LIMIT 1",
            (wb_id,),
        ).fetchone()
        if latest:
            try:
                age = (datetime.now() - datetime.fromisoformat(latest["created_at"])).total_seconds()
                if age < config.VERSION_INTERVAL_SECONDS and workbook["save_count"] % 8:
                    return None
            except ValueError:
                pass
    number = (conn.execute(
        "SELECT COALESCE(MAX(number), 0) AS n FROM versions WHERE workbook_id=?", (wb_id,)
    ).fetchone()["n"] + 1)
    sheets = []
    for sheet in conn.execute("SELECT * FROM sheets WHERE workbook_id=? ORDER BY position, id", (wb_id,)).fetchall():
        cells = _load_cells(conn, sheet["id"])
        sheets.append({
            "id": sheet["id"],
            "name": sheet["name"],
            "cells": {f"{r},{c}": cell for (r, c), cell in cells.items()},
            "row_count": sheet["row_count"],
            "col_count": sheet["col_count"],
            "frozen_rows": sheet["frozen_rows"],
            "frozen_cols": sheet["frozen_cols"],
            "hidden_rows": json.loads(sheet["hidden_rows"] or "[]"),
            "hidden_cols": json.loads(sheet["hidden_cols"] or "[]"),
            "merges": json.loads(sheet["merges"] or "[]"),
        })
    workbook_row = dict(workbook)
    snapshot = {
        "workbook": {"name": workbook_row["name"], "id": wb_id},
        "sheets": sheets,
        "label": label,
        "created_at": _now(),
    }
    folder = Path(workbook_row["storage_root"]) / "backups" / wb_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"v{number:05d}.json.gz"
    payload = gzip.compress(json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))
    temp = path.with_suffix(".tmp")
    temp.write_bytes(payload)
    temp.replace(path)
    cell_count = sum(len(sheet["cells"]) for sheet in sheets)
    cursor = conn.execute(
        "INSERT INTO versions (workbook_id, number, label, path, size, cell_count, created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (wb_id, number, label, str(path), len(payload), cell_count, _now()),
    )
    conn.execute("UPDATE workbooks SET version_count=version_count+1 WHERE id=?", (wb_id,))
    _prune_versions(conn, wb_id)
    return {
        "id": cursor.lastrowid, "number": number, "label": label,
        "size": len(payload), "cells": cell_count, "created_at": _now(),
    }


def _prune_versions(conn, wb_id: str) -> None:
    retention = config.get_version_retention()
    rows = conn.execute(
        "SELECT id, path FROM versions WHERE workbook_id=? ORDER BY number DESC", (wb_id,)
    ).fetchall()
    for row in rows[retention:]:
        try:
            Path(row["path"]).unlink()
        except OSError:
            pass
        conn.execute("DELETE FROM versions WHERE id=?", (row["id"],))


def list_versions(wb_id: str) -> tuple[list[dict], int]:
    with db.read() as conn:
        rows = conn.execute(
            "SELECT id, number, label, size, cell_count, created_at FROM versions"
            " WHERE workbook_id=? ORDER BY number DESC", (wb_id,)
        ).fetchall()
        batches = conn.execute(
            "SELECT COUNT(*) AS n FROM batches WHERE workbook_id=? AND undone=0", (wb_id,)
        ).fetchone()["n"]
    return [dict(row) for row in rows], batches


def current_version_number(wb_id: str) -> int:
    with db.read() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(number), 0) AS n FROM versions WHERE workbook_id=?", (wb_id,)
        ).fetchone()
    return row["n"] if row else 0


def restore_version(wb_id: str, version_id: int) -> dict:
    with db.transaction() as conn:
        version = conn.execute(
            "SELECT * FROM versions WHERE id=? AND workbook_id=?", (int(version_id), wb_id)
        ).fetchone()
        if not version:
            raise LookupError("Version not found.")
        try:
            snapshot = json.loads(gzip.decompress(Path(version["path"]).read_bytes()))
        except (OSError, ValueError) as exc:
            raise ValueError(f"That backup cannot be read: {exc}") from exc
        create_version(conn, wb_id, "Before restore", force=True)
        conn.execute("DELETE FROM sheets WHERE workbook_id=?", (wb_id,))
        for position, sheet in enumerate(snapshot["sheets"]):
            cells = {}
            for key, cell in sheet.get("cells", {}).items():
                row, col = key.split(",")
                cells[(int(row), int(col))] = cell
            _insert_sheet(conn, wb_id, {
                "name": sheet["name"], "cells": cells,
                "row_count": sheet.get("row_count", 100),
                "col_count": sheet.get("col_count", 20),
                "hidden_rows": sheet.get("hidden_rows", []),
                "hidden_cols": sheet.get("hidden_cols", []),
                "merges": sheet.get("merges", []),
            }, position, sheet_id=sheet.get("id"))
        conn.execute(
            "UPDATE workbooks SET updated_at=?, analysis_stale=1 WHERE id=?", (_now(), wb_id)
        )
        create_version(conn, wb_id, f"Restored v{version['number']}", force=True)
        return _workbook(conn, wb_id)


def _apply_snapshot_cells(conn, sheet_id: int, cells: dict) -> None:
    conn.execute("DELETE FROM cells WHERE sheet_id=?", (sheet_id,))
    payload = []
    for (row, col), cell in cells.items():
        payload.append((
            sheet_id, int(row), int(col), cell.get("v"), cell.get("num"),
            cell.get("f"), json.dumps(cell.get("s")) if cell.get("s") else None,
        ))
    if payload:
        conn.executemany(
            "INSERT INTO cells (sheet_id, row, col, value, num, formula, style)"
            " VALUES (?,?,?,?,?,?,?)", payload,
        )


# --------------------------------------------------------------------------- #
# Editing operations
# --------------------------------------------------------------------------- #

class OpError(ValueError):
    pass


def _cell_values(cell) -> tuple:
    if not cell:
        return (None, None, None, None)
    return (cell.get("v"), cell.get("num"), cell.get("f"), cell.get("s"))


def _write_cell(conn, sheet_id: int, row: int, col: int, value, num, formula, style) -> None:
    if value in (None, "") and not formula and not style:
        conn.execute("DELETE FROM cells WHERE sheet_id=? AND row=? AND col=?",
                     (sheet_id, row, col))
        return
    conn.execute(
        "INSERT OR REPLACE INTO cells (sheet_id, row, col, value, num, formula, style)"
        " VALUES (?,?,?,?,?,?,?)",
        (sheet_id, row, col, value, num, formula,
         json.dumps(style) if style else None),
    )


def _before_cell(conn, sheet_id, row, col) -> dict | None:
    cell = conn.execute(
        "SELECT value, num, formula, style FROM cells WHERE sheet_id=? AND row=? AND col=?",
        (sheet_id, row, col),
    ).fetchone()
    if not cell:
        return None
    return {
        "v": cell["value"], "num": cell["num"], "f": cell["formula"],
        "s": json.loads(cell["style"]) if cell["style"] else None,
    }


def _set_cell_inverse(sheet_id, row, col, before) -> dict:
    if before is None:
        return {"op": "delete_cell", "sheet": sheet_id, "row": row, "col": col}
    return {
        "op": "set_cell", "sheet": sheet_id, "row": row, "col": col,
        "value": before["v"], "formula": before["f"], "style": before["s"],
        "_num": before["num"],
    }


def _normalise_style(style) -> dict | None:
    if style is None:
        return None
    if not isinstance(style, dict):
        raise OpError("Style must be an object.")
    clean = {}
    allowed = {"bold", "italic", "underline", "strike", "color", "fill", "align",
               "valign", "wrap", "numfmt", "font", "size"}
    for key, value in style.items():
        if key not in allowed:
            continue
        clean[key] = value
    return clean or None


def _parse_value(raw):
    if raw is None:
        return None, None
    if isinstance(raw, bool):
        return ("TRUE" if raw else "FALSE"), (1.0 if raw else 0.0)
    if isinstance(raw, (int, float)):
        number = float(raw)
        text = str(int(number)) if number.is_integer() and abs(number) < 1e15 else str(number)
        return text, number
    return str(raw), None


def apply_ops(wb_id: str, ops: list[dict], label: str = "Edit", version: bool = False) -> dict:
    if not isinstance(ops, list) or not ops:
        raise OpError("No edits were sent.")
    if len(ops) > 400:
        raise OpError("Too many separate edits in one save.")
    with db.transaction() as conn:
        workbook = conn.execute("SELECT * FROM workbooks WHERE id=?", (wb_id,)).fetchone()
        if not workbook:
            raise LookupError("Workbook not found.")
        forward: list[dict] = []
        inverse: list[dict] = []
        affected = set()
        for op in ops:
            if not isinstance(op, dict):
                raise OpError("Invalid edit.")
            sheet = _fetch_sheet(conn, wb_id, op.get("sheet"))
            op = dict(op)
            op["sheet"] = sheet["id"]
            fwd, inv, rows = _apply_op(conn, wb_id, sheet, op)
            forward.extend(fwd)
            inverse.extend(inv)
            affected.update(rows)
        if not forward:
            raise OpError("Nothing changed.")
        _recalculate(conn, wb_id)
        batch = conn.execute(
            "INSERT INTO batches (workbook_id, label, forward, inverse, created_at)"
            " VALUES (?,?,?,?,?)",
            (wb_id, label, json.dumps(forward), json.dumps(inverse), _now()),
        ).lastrowid
        conn.execute("DELETE FROM batches WHERE workbook_id=? AND undone=1 AND id<>?", (wb_id, batch))
        conn.execute(
            "UPDATE workbooks SET updated_at=?, analysis_stale=1, save_count=save_count+1 WHERE id=?",
            (_now(), wb_id),
        )
        saved = _workbook(conn, wb_id)
        if version:
            create_version(conn, wb_id, label, force=True)
        else:
            create_version(conn, wb_id, label, force=False)
        meta = {
            "saved_at": _now(),
            "batch": batch,
            "can_undo": _can_undo(conn, wb_id),
            "can_redo": _can_redo(conn, wb_id),
            "affected_rows": sorted(affected)[:500],
            "version": conn.execute(
                "SELECT COALESCE(MAX(number), 0) AS n FROM versions WHERE workbook_id=?", (wb_id,)
            ).fetchone()["n"],
            "workbook": saved,
        }
        return meta


def _apply_op(conn, wb_id, sheet, op) -> tuple[list[dict], list[dict], set]:
    kind = op.get("op")
    sid = sheet["id"]
    forward: list[dict] = []
    inverse: list[dict] = []
    affected: set[int] = set()

    if kind == "set_cell":
        row, col = int(op.get("row") or 0), int(op.get("col") or 0)
        if not (1 <= row <= 200000 and 1 <= col <= 500):
            raise OpError("Cell reference is outside the sheet.")
        formula = op.get("formula")
        if formula is not None:
            formula = str(formula).strip()
            if not formula:
                formula = None
            elif not formula.startswith("="):
                formula = "=" + formula
            if formula and len(formula) > formulas.MAX_FORMULA_LENGTH:
                raise OpError("That formula is too long.")
        if formula:
            value, num = None, None
        else:
            value, num = _parse_value(op.get("value"))
            if isinstance(value, str) and len(value) > 32767:
                raise OpError("That value is too long for a spreadsheet cell.")
        style = _normalise_style(op.get("style")) if "style" in op else None
        keep_style = "style" not in op
        before = _before_cell(conn, sid, row, col)
        if keep_style:
            style = before["s"] if before else None
        _write_cell(conn, sid, row, col, value, num, formula, style)
        forward.append({
            "op": "set_cell", "sheet": sid, "row": row, "col": col,
            "value": value, "formula": formula, "style": style, "_num": num,
        })
        inverse.append(_set_cell_inverse(sid, row, col, before))
        affected.add(row)
        if row > sheet["row_count"] or col > sheet["col_count"]:
            conn.execute(
                "UPDATE sheets SET row_count=?, col_count=? WHERE id=?",
                (max(sheet["row_count"], row), max(sheet["col_count"], col), sid),
            )
        return forward, inverse, affected

    if kind == "set_range":
        rng = op.get("range") or []
        if len(rng) != 4:
            raise OpError("Invalid range.")
        r1, c1, r2, c2 = (int(value) for value in rng)
        top, bottom = sorted((r1, r2))
        left, right = sorted((c1, c2))
        if (bottom - top + 1) * (right - left + 1) > CELL_LIMIT:
            raise OpError("That range is too large.")
        style_patch = _normalise_style(op.get("style")) if op.get("style") is not None else None
        value_mode = op.get("value_mode", "keep")
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                before = _before_cell(conn, sid, row, col)
                old_style = before["s"] if before else None
                style = dict(old_style or {}) if style_patch is not None else old_style
                if style_patch is not None:
                    for key, value in style_patch.items():
                        if value in (None, False, ""):
                            style.pop(key, None)
                        else:
                            style[key] = value
                value = before["v"] if before else None
                formula = before["f"] if before else None
                num = before["num"] if before else None
                if value_mode == "clear":
                    value, formula, num = None, None, None
                elif value_mode == "set":
                    value, num = _parse_value(op.get("value"))
                    formula = None
                _write_cell(conn, sid, row, col, value, num, formula, style or None)
                forward.append({
                    "op": "set_cell", "sheet": sid, "row": row, "col": col,
                    "value": value, "formula": formula, "style": style or None, "_num": num,
                })
                inverse.append(_set_cell_inverse(sid, row, col, before))
                affected.add(row)
        return forward, inverse, affected

    if kind == "clear":
        rng = op.get("range") or []
        if len(rng) != 4:
            raise OpError("Invalid range.")
        r1, c1, r2, c2 = (int(value) for value in rng)
        top, bottom = sorted((r1, r2))
        left, right = sorted((c1, c2))
        if (bottom - top + 1) * (right - left + 1) > CELL_LIMIT:
            raise OpError("That range is too large.")
        keep_style = bool(op.get("keep_style"))
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                before = _before_cell(conn, sid, row, col)
                if not before:
                    continue
                style = before["s"] if keep_style else None
                _write_cell(conn, sid, row, col, None, None, None, style)
                forward.append({
                    "op": "set_cell", "sheet": sid, "row": row, "col": col,
                    "value": None, "formula": None, "style": style,
                })
                inverse.append(_set_cell_inverse(sid, row, col, before))
                affected.add(row)
        return forward, inverse, affected

    if kind in ("insert_rows", "delete_rows", "insert_cols", "delete_cols"):
        is_row = kind.endswith("rows")
        insert = kind.startswith("insert")
        at = int(op.get("at") or 1)
        count = max(1, min(int(op.get("count") or 1), 200))
        cells = _load_cells(conn, sid)
        if is_row:
            if not 1 <= at <= sheet["row_count"] + 1:
                raise OpError("Row is outside the sheet.")
            if insert:
                _shift_rows(conn, sid, cells, at, count)
                conn.execute("UPDATE sheets SET row_count=row_count+? WHERE id=?",
                             (count, sid))
                forward.append({"op": "insert_rows", "sheet": sid, "at": at, "count": count})
                inverse.append({"op": "delete_rows", "sheet": sid, "at": at, "count": count})
            else:
                snapshot = _rows_snapshot(cells, at, count)
                _delete_rows(conn, sid, cells, at, count)
                conn.execute(
                    "UPDATE sheets SET row_count=MAX(20, row_count-?) WHERE id=?",
                    (count, sid))
                forward.append({"op": "delete_rows", "sheet": sid, "at": at, "count": count})
                inverse.append({"op": "_restore_rows", "sheet": sid, "at": at,
                                "count": count, "rows": snapshot})
            affected.update(range(at, at + 6))
            return forward, inverse, affected

        if not 1 <= at <= sheet["col_count"] + 1:
            raise OpError("Column is outside the sheet.")
        if insert:
            _shift_cols(conn, sid, cells, at, count)
            conn.execute("UPDATE sheets SET col_count=col_count+? WHERE id=?", (count, sid))
            forward.append({"op": "insert_cols", "sheet": sid, "at": at, "count": count})
            inverse.append({"op": "delete_cols", "sheet": sid, "at": at, "count": count})
        else:
            snapshot = _cols_snapshot(cells, at, count)
            _delete_cols(conn, sid, cells, at, count)
            conn.execute(
                "UPDATE sheets SET col_count=MAX(5, col_count-?) WHERE id=?", (count, sid))
            forward.append({"op": "delete_cols", "sheet": sid, "at": at, "count": count})
            inverse.append({"op": "_restore_cols", "sheet": sid, "at": at,
                            "count": count, "cols": snapshot})
        affected.update(range(1, 200))
        return forward, inverse, affected

    if kind == "sort":
        col = int(op.get("col") or 1)
        direction = "desc" if str(op.get("direction", "asc")).lower() == "desc" else "asc"
        from_row = max(2, int(op.get("from_row") or 2))
        to_row = min(sheet["row_count"], int(op.get("to_row") or sheet["row_count"]))
        if to_row < from_row:
            raise OpError("There are no data rows to sort.")
        cells = _load_cells(conn, sid)
        rows = list(range(from_row, to_row + 1))

        def key(row):
            cell = cells.get((row, col))
            value = cell.get("num") if cell and cell.get("num") is not None else (
                _cell_text(cell.get("v")) if cell else "")
            blank = value in (None, "")
            return (blank, value if isinstance(value, (int, float)) else str(value).lower())

        rows.sort(key=key, reverse=direction == "desc")
        before = _rows_snapshot(cells, from_row, to_row - from_row + 1)
        for offset in range(0, to_row - from_row + 1):
            conn.execute("DELETE FROM cells WHERE sheet_id=? AND row=?",
                         (sid, from_row + offset))
        for index, source_row in enumerate(rows, start=from_row):
            for c in range(1, sheet["col_count"] + 1):
                cell = cells.get((source_row, c))
                if cell:
                    _write_cell(conn, sid, index, c, cell["v"], cell["num"],
                                cell["f"], cell["s"])
        after_cells = _load_cells(conn, sid)
        after = _rows_snapshot(after_cells, from_row, to_row - from_row + 1)
        forward.append({"op": "_set_rows", "sheet": sid, "at": from_row, "rows": after})
        inverse.append({"op": "_set_rows", "sheet": sid, "at": from_row, "rows": before})
        affected.update(range(from_row, to_row + 3))
        return forward, inverse, affected

    if kind == "add_sheet":
        name = _unique_sheet_name(conn, wb_id, op.get("name") or "Sheet")
        position = max(0, min(int(op.get("after") or 0), len(_load_sheets(conn, wb_id))))
        conn.execute("UPDATE sheets SET position=position+1 WHERE workbook_id=? AND position>=?",
                     (wb_id, position))
        new_id = _insert_sheet(conn, wb_id, {
            "name": name, "cells": {}, "row_count": 100, "col_count": 20,
        }, position)
        forward.append({"op": "_restore_sheet", "sheet_snapshot": {
            "id": new_id, "name": name, "position": position, "row_count": 100,
            "col_count": 20, "cells": {}, "hidden_rows": [], "hidden_cols": [], "merges": [],
        }})
        inverse.append({"op": "delete_sheet", "sheet": new_id})
        return forward, inverse, {1}

    if kind == "rename_sheet":
        name = _unique_sheet_name(conn, wb_id, op.get("name") or sheet["name"], exclude=sid)
        forward.append({"op": "rename_sheet", "sheet": sid, "name": name})
        inverse.append({"op": "rename_sheet", "sheet": sid, "name": sheet["name"]})
        conn.execute("UPDATE sheets SET name=? WHERE id=?", (name, sid))
        return forward, inverse, set()

    if kind == "delete_sheet":
        sheets = _load_sheets(conn, wb_id)
        if len(sheets) <= 1:
            raise OpError("A workbook needs at least one worksheet.")
        cells = _load_cells(conn, sid)
        snapshot = {
            "id": sid,
            "name": sheet["name"], "position": sheet["position"],
            "row_count": sheet["row_count"], "col_count": sheet["col_count"],
            "hidden_rows": json.loads(sheet["hidden_rows"] or "[]"),
            "hidden_cols": json.loads(sheet["hidden_cols"] or "[]"),
            "merges": json.loads(sheet["merges"] or "[]"),
            "cells": {f"{r},{c}": cell for (r, c), cell in cells.items()},
        }
        conn.execute("DELETE FROM sheets WHERE id=?", (sid,))
        conn.execute("UPDATE sheets SET position=position-1 WHERE workbook_id=? AND position>?",
                     (wb_id, sheet["position"]))
        forward.append({"op": "delete_sheet", "sheet": sid})
        inverse.append({"op": "_restore_sheet", "sheet_snapshot": snapshot})
        return forward, inverse, set()

    if kind == "duplicate_sheet":
        name = _unique_sheet_name(conn, wb_id, op.get("name") or f"{sheet['name']} copy")
        cells = _load_cells(conn, sid)
        position = sheet["position"] + 1
        conn.execute("UPDATE sheets SET position=position+1 WHERE workbook_id=? AND position>=?",
                     (wb_id, position))
        new_id = _insert_sheet(conn, wb_id, {
            "name": name,
            "cells": {key: dict(value) for key, value in cells.items()},
            "row_count": sheet["row_count"], "col_count": sheet["col_count"],
            "hidden_rows": json.loads(sheet["hidden_rows"] or "[]"),
            "hidden_cols": json.loads(sheet["hidden_cols"] or "[]"),
            "merges": json.loads(sheet["merges"] or "[]"),
        }, position)
        snapshot = {
            "id": new_id,
            "name": name, "position": position,
            "row_count": sheet["row_count"], "col_count": sheet["col_count"],
            "hidden_rows": json.loads(sheet["hidden_rows"] or "[]"),
            "hidden_cols": json.loads(sheet["hidden_cols"] or "[]"),
            "merges": json.loads(sheet["merges"] or "[]"),
            "cells": {f"{r},{c}": cell for (r, c), cell in cells.items()},
        }
        forward.append({"op": "_restore_sheet", "sheet_snapshot": snapshot})
        inverse.append({"op": "delete_sheet", "sheet": new_id})
        return forward, inverse, {1}

    if kind == "move_sheet":
        sheets = _load_sheets(conn, wb_id)
        target = max(0, min(int(op.get("to") or 0), len(sheets) - 1))
        if target == sheet["position"]:
            return [], [], set()
        source_position = sheet["position"]
        _move_sheet(conn, wb_id, sid, target)
        forward.append({"op": "move_sheet", "sheet": sid, "to": target})
        inverse.append({"op": "move_sheet", "sheet": sid, "to": source_position})
        return forward, inverse, set()

    if kind in ("set_layout",):
        columns = {
            "frozen_rows": int(op.get("frozen_rows", sheet["frozen_rows"]) or 0),
            "frozen_cols": int(op.get("frozen_cols", sheet["frozen_cols"]) or 0),
        }
        if "hidden_rows" in op:
            columns["hidden_rows"] = json.dumps(sorted(set(
                int(value) for value in op.get("hidden_rows") or [] if str(value).isdigit())))
        if "hidden_cols" in op:
            columns["hidden_cols"] = json.dumps(sorted(set(
                int(value) for value in op.get("hidden_cols") or [] if str(value).isdigit())))
        if "row_count" in op:
            columns["row_count"] = max(20, min(200000, int(op["row_count"])))
        if "col_count" in op:
            columns["col_count"] = max(5, min(500, int(op["col_count"])))
        assignments = ", ".join(f"{key}=?" for key in columns)
        conn.execute(f"UPDATE sheets SET {assignments} WHERE id=?",
                     (*columns.values(), sid))
        inverse.append({"op": "set_layout", "sheet": sid, **{
            key: (json.loads(sheet[key]) if key.startswith("hidden") else sheet[key])
            for key in columns
        }})
        return [{"op": "set_layout", "sheet": sid, **{
            key: (json.loads(columns[key]) if key.startswith("hidden") else columns[key])
            for key in columns
        }}], inverse, set()

    if kind in ("merge", "unmerge"):
        rng = [int(value) for value in (op.get("range") or [])]
        if len(rng) != 4:
            raise OpError("Invalid merge range.")
        merges = json.loads(sheet["merges"] or "[]")
        entry = {"r1": min(rng[0], rng[2]), "c1": min(rng[1], rng[3]),
                 "r2": max(rng[0], rng[2]), "c2": max(rng[1], rng[3])}
        if kind == "merge":
            merges = [m for m in merges if not _ranges_overlap(m, entry)]
            merges.append(entry)
        else:
            merges = [m for m in merges if m != entry]
        conn.execute("UPDATE sheets SET merges=? WHERE id=?", (json.dumps(merges), sid))
        inverse.append({"op": "set_merges", "sheet": sid,
                        "merges": json.loads(sheet["merges"] or "[]")})
        return [{"op": "set_merges", "sheet": sid, "merges": merges}], inverse, set()

    if kind == "move_col":
        col = int(op.get("col") or 1)
        to = int(op.get("to") or col)
        if col == to:
            return [], [], set()
        col_count = sheet["col_count"]
        cells = _load_cells(conn, sid)
        before = _cols_snapshot(cells, 1, col_count)
        order = list(range(1, col_count + 1))
        order.remove(col)
        order.insert(max(0, min(to - 1, len(order))), col)
        conn.execute("DELETE FROM cells WHERE sheet_id=?", (sid,))
        for row in range(1, sheet["row_count"] + 1):
            for index, source_col in enumerate(order, start=1):
                cell = cells.get((row, source_col))
                if cell:
                    _write_cell(conn, sid, row, index, cell["v"], cell["num"],
                                cell["f"], cell["s"])
        after_cells = _load_cells(conn, sid)
        after = _cols_snapshot(after_cells, 1, col_count)
        forward.append({"op": "_set_cols", "sheet": sid, "at": 1, "cols": after})
        inverse.append({"op": "_set_cols", "sheet": sid, "at": 1, "cols": before})
        affected.update(range(1, 200))
        return forward, inverse, affected

    if kind == "replace":
        find = str(op.get("find") or "")
        replacement = str(op.get("replace") or "")
        if not find:
            raise OpError("Nothing to find.")
        match_case = bool(op.get("match_case"))
        whole = bool(op.get("whole"))
        has_header = bool(op.get("has_header", True))
        rng = op.get("range") or [1, 1, sheet["row_count"], sheet["col_count"]]
        r1, c1, r2, c2 = (int(value) for value in rng)
        cells = _load_cells(conn, sid)
        changed = 0
        for row in range(min(r1, r2), max(r1, r2) + 1):
            if has_header and row == 1:
                continue
            for col in range(min(c1, c2), max(c1, c2) + 1):
                cell = cells.get((row, col))
                if not cell or cell.get("f") or cell.get("v") is None:
                    continue
                text = _cell_text(cell["v"])
                if match_case:
                    found = (text == find) if whole else (find in text)
                    updated = (text.replace(find, replacement) if not whole
                               else (replacement if text == find else text))
                else:
                    pattern = re.escape(find)
                    if whole:
                        updated = replacement if re.fullmatch(pattern, text, re.I) else text
                    else:
                        updated = re.sub(pattern, replacement.replace("\\", "\\\\"), text, flags=re.I)
                    found = updated != text
                if not found:
                    continue
                value, num = _parse_value(updated)
                before = _before_cell(conn, sid, row, col)
                _write_cell(conn, sid, row, col, value, num, None, cell["s"])
                forward.append({"op": "set_cell", "sheet": sid, "row": row, "col": col,
                                "value": value, "formula": None, "style": cell["s"]})
                inverse.append(_set_cell_inverse(sid, row, col, before))
                changed += 1
                if changed >= 2000:
                    break
            if changed >= 2000:
                break
        if not changed:
            raise OpError("No cells matched that search.")
        affected.update(range(min(r1, r2), max(r1, r2) + 1))
        return forward, inverse, affected

    if kind == "reset_formula_cache":
        return forward, inverse, affected

    raise OpError(f"Unsupported edit operation: {kind}")


def _ranges_overlap(a, b) -> bool:
    return not (a["r2"] < b["r1"] or b["r2"] < a["r1"]
                or a["c2"] < b["c1"] or b["c2"] < a["c1"])


def _rows_snapshot(cells, start_row, count) -> list[dict]:
    rows = []
    for row in range(start_row, start_row + count):
        entries = []
        for (r, c), cell in cells.items():
            if r == row:
                entries.append({
                    "col": c, "value": cell.get("v"), "num": cell.get("num"),
                    "formula": cell.get("f"), "style": cell.get("s"),
                })
        rows.append({"row": row, "cells": entries})
    return rows


def _cols_snapshot(cells, start_col, count) -> list[dict]:
    cols = []
    for col in range(start_col, start_col + count):
        entries = []
        for (r, c), cell in cells.items():
            if c == col:
                entries.append({
                    "row": r, "value": cell.get("v"), "num": cell.get("num"),
                    "formula": cell.get("f"), "style": cell.get("s"),
                })
        cols.append({"col": col, "cells": entries})
    return cols


def _shift_rows(conn, sheet_id, cells, at, count) -> None:
    moving = {key: cell for key, cell in cells.items() if key[0] >= at}
    conn.execute("DELETE FROM cells WHERE sheet_id=? AND row>=?", (sheet_id, at))
    for (row, col), cell in sorted(moving.items()):
        _write_cell(conn, sheet_id, row + count, col, cell["v"], cell["num"],
                    cell["f"], cell["s"])


def _delete_rows(conn, sheet_id, cells, at, count) -> None:
    moving = {key: cell for key, cell in cells.items() if key[0] >= at}
    conn.execute("DELETE FROM cells WHERE sheet_id=? AND row>=?", (sheet_id, at))
    for (row, col), cell in sorted(moving.items()):
        if row >= at + count:
            _write_cell(conn, sheet_id, row - count, col, cell["v"], cell["num"],
                        cell["f"], cell["s"])


def _shift_cols(conn, sheet_id, cells, at, count) -> None:
    moving = {key: cell for key, cell in cells.items() if key[1] >= at}
    conn.execute("DELETE FROM cells WHERE sheet_id=? AND col>=?", (sheet_id, at))
    for (row, col), cell in sorted(moving.items()):
        _write_cell(conn, sheet_id, row, col + count, cell["v"], cell["num"],
                    cell["f"], cell["s"])


def _delete_cols(conn, sheet_id, cells, at, count) -> None:
    moving = {key: cell for key, cell in cells.items() if key[1] >= at}
    conn.execute("DELETE FROM cells WHERE sheet_id=? AND col>=?", (sheet_id, at))
    for (row, col), cell in sorted(moving.items()):
        if col >= at + count:
            _write_cell(conn, sheet_id, row, col - count, cell["v"], cell["num"],
                        cell["f"], cell["s"])


def _move_sheet(conn, wb_id, sheet_id, target) -> None:
    row = conn.execute("SELECT position FROM sheets WHERE id=?", (sheet_id,)).fetchone()
    if not row:
        raise OpError("Worksheet not found.")
    source = row["position"]
    if source == target:
        return
    conn.execute("UPDATE sheets SET position=-1000000 WHERE id=?", (sheet_id,))
    if target > source:
        conn.execute(
            "UPDATE sheets SET position=position-1 WHERE workbook_id=?"
            " AND position>? AND position<=?", (wb_id, source, target))
    else:
        conn.execute(
            "UPDATE sheets SET position=position+1 WHERE workbook_id=?"
            " AND position>=? AND position<?", (wb_id, target, source))
    conn.execute("UPDATE sheets SET position=? WHERE id=?", (target, sheet_id))


def _write_row_blocks(conn, sheet_id, rows) -> None:
    for block in rows or []:
        for entry in block.get("cells", []):
            _write_cell(conn, sheet_id, int(block["row"]), int(entry["col"]),
                        entry.get("value"), entry.get("num"), entry.get("formula"),
                        entry.get("style"))


def _write_col_blocks(conn, sheet_id, cols) -> None:
    for block in cols or []:
        for entry in block.get("cells", []):
            _write_cell(conn, sheet_id, int(entry["row"]), int(block["col"]),
                        entry.get("value"), entry.get("num"), entry.get("formula"),
                        entry.get("style"))


def _set_rows_raw(conn, sheet_id, at, rows) -> None:
    if not rows:
        return
    conn.execute("DELETE FROM cells WHERE sheet_id=? AND row>=? AND row<?",
                 (sheet_id, at, at + len(rows)))
    _write_row_blocks(conn, sheet_id, rows)


def _set_cols_raw(conn, sheet_id, at, cols) -> None:
    if not cols:
        return
    conn.execute("DELETE FROM cells WHERE sheet_id=? AND col>=? AND col<?",
                 (sheet_id, at, at + len(cols)))
    _write_col_blocks(conn, sheet_id, cols)


def _unique_sheet_name(conn, wb_id, raw, exclude=None) -> str:
    name = re.sub(r"[\[\]:*?/\\]", " ", str(raw or "")).strip()[:31] or "Sheet"
    existing = {
        row["name"].lower() for row in conn.execute(
            "SELECT name FROM sheets WHERE workbook_id=? AND id<>?", (wb_id, int(exclude or 0))
        ).fetchall()
    }
    candidate = name
    index = 2
    while candidate.lower() in existing:
        tail = f" ({index})"
        candidate = name[: 31 - len(tail)] + tail
        index += 1
    return candidate


def _recalculate(conn, wb_id) -> None:
    sheets = conn.execute(
        "SELECT id, name FROM sheets WHERE workbook_id=? ORDER BY position, id", (wb_id,)
    ).fetchall()
    grids = {}
    raw = {}
    for sheet in sheets:
        cells = _load_cells(conn, sheet["id"])
        raw[sheet["id"]] = cells
        grid = {}
        for (row, col), cell in cells.items():
            grid[(row, col)] = {"v": cell.get("v"), "f": cell.get("f")}
        grids[sheet["name"]] = grid
    results = formulas.compute(grids)
    name_to_id = {sheet["name"]: sheet["id"] for sheet in sheets}
    for (sheet_name, row, col), result in results.items():
        sheet_id = name_to_id.get(sheet_name)
        if not sheet_id:
            continue
        cell = raw[sheet_id].get((row, col))
        if not cell:
            continue
        if isinstance(result, str) and result.startswith("#"):
            cached = cell.get("v")
            if cached not in (None, "") and not (isinstance(cached, str) and cached.startswith("#")):
                continue
        value, num = _parse_value(result)
        conn.execute(
            "UPDATE cells SET value=?, num=? WHERE sheet_id=? AND row=? AND col=?",
            (value, num, sheet_id, row, col),
        )


def undo(wb_id: str) -> dict:
    with db.transaction() as conn:
        batch = conn.execute(
            "SELECT * FROM batches WHERE workbook_id=? AND undone=0 ORDER BY id DESC LIMIT 1",
            (wb_id,),
        ).fetchone()
        if not batch:
            raise OpError("Nothing to undo.")
        inverse = json.loads(batch["inverse"])
        for op in reversed(inverse):
            _apply_raw(conn, wb_id, op)
        conn.execute("UPDATE batches SET undone=1 WHERE id=?", (batch["id"],))
        _recalculate(conn, wb_id)
        conn.execute("UPDATE workbooks SET updated_at=?, analysis_stale=1 WHERE id=?",
                     (_now(), wb_id))
        return {
            "label": batch["label"], "saved_at": _now(),
            "can_undo": _can_undo(conn, wb_id), "can_redo": _can_redo(conn, wb_id),
            "workbook": _workbook(conn, wb_id),
        }


def redo(wb_id: str) -> dict:
    with db.transaction() as conn:
        batch = conn.execute(
            "SELECT * FROM batches WHERE workbook_id=? AND undone=1 ORDER BY id ASC LIMIT 1",
            (wb_id,),
        ).fetchone()
        if not batch:
            raise OpError("Nothing to redo.")
        forward = json.loads(batch["forward"])
        for op in forward:
            _apply_raw(conn, wb_id, op)
        conn.execute("UPDATE batches SET undone=0 WHERE id=?", (batch["id"],))
        _recalculate(conn, wb_id)
        conn.execute("UPDATE workbooks SET updated_at=?, analysis_stale=1 WHERE id=?",
                     (_now(), wb_id))
        return {
            "label": batch["label"], "saved_at": _now(),
            "can_undo": _can_undo(conn, wb_id), "can_redo": _can_redo(conn, wb_id),
            "workbook": _workbook(conn, wb_id),
        }


def undo_state(wb_id: str) -> dict:
    with db.read() as conn:
        return {"can_undo": _can_undo(conn, wb_id), "can_redo": _can_redo(conn, wb_id)}


def _can_undo(conn, wb_id) -> bool:
    return conn.execute(
        "SELECT 1 FROM batches WHERE workbook_id=? AND undone=0 LIMIT 1", (wb_id,)
    ).fetchone() is not None


def _can_redo(conn, wb_id) -> bool:
    return conn.execute(
        "SELECT 1 FROM batches WHERE workbook_id=? AND undone=1 LIMIT 1", (wb_id,)
    ).fetchone() is not None


def _apply_raw(conn, wb_id, op) -> None:
    kind = op.get("op")
    sheet_id = int(op.get("sheet") or 0)
    if kind == "delete_cell":
        conn.execute("DELETE FROM cells WHERE sheet_id=? AND row=? AND col=?",
                     (sheet_id, int(op["row"]), int(op["col"])))
        return
    if kind == "set_cell":
        _write_cell(conn, sheet_id, int(op["row"]), int(op["col"]), op.get("value"),
                    op.get("_num"), op.get("formula"), op.get("style"))
        return
    if kind == "insert_rows":
        cells = _load_cells(conn, sheet_id)
        _shift_rows(conn, sheet_id, cells, int(op["at"]), int(op["count"]))
        conn.execute("UPDATE sheets SET row_count=row_count+? WHERE id=?",
                     (int(op["count"]), sheet_id))
        return
    if kind == "delete_rows":
        cells = _load_cells(conn, sheet_id)
        _delete_rows(conn, sheet_id, cells, int(op["at"]), int(op["count"]))
        conn.execute("UPDATE sheets SET row_count=MAX(20, row_count-?) WHERE id=?",
                     (int(op["count"]), sheet_id))
        return
    if kind == "insert_cols":
        cells = _load_cells(conn, sheet_id)
        _shift_cols(conn, sheet_id, cells, int(op["at"]), int(op["count"]))
        conn.execute("UPDATE sheets SET col_count=col_count+? WHERE id=?",
                     (int(op["count"]), sheet_id))
        return
    if kind == "delete_cols":
        cells = _load_cells(conn, sheet_id)
        _delete_cols(conn, sheet_id, cells, int(op["at"]), int(op["count"]))
        conn.execute("UPDATE sheets SET col_count=MAX(5, col_count-?) WHERE id=?",
                     (int(op["count"]), sheet_id))
        return
    if kind == "_restore_rows":
        rows = op.get("rows") or []
        count = int(op.get("count") or len(rows))
        at = int(op["at"])
        cells = _load_cells(conn, sheet_id)
        _shift_rows(conn, sheet_id, cells, at, count)
        _write_row_blocks(conn, sheet_id, rows)
        conn.execute("UPDATE sheets SET row_count=row_count+? WHERE id=?", (count, sheet_id))
        return
    if kind == "_restore_cols":
        cols = op.get("cols") or []
        count = int(op.get("count") or len(cols))
        at = int(op["at"])
        cells = _load_cells(conn, sheet_id)
        _shift_cols(conn, sheet_id, cells, at, count)
        _write_col_blocks(conn, sheet_id, cols)
        conn.execute("UPDATE sheets SET col_count=col_count+? WHERE id=?", (count, sheet_id))
        return
    if kind == "_set_rows":
        _set_rows_raw(conn, sheet_id, int(op["at"]), op.get("rows") or [])
        return
    if kind == "_set_cols":
        _set_cols_raw(conn, sheet_id, int(op["at"]), op.get("cols") or [])
        return
    if kind == "_restore_sheet":
        snapshot = op["sheet_snapshot"]
        conn.execute("UPDATE sheets SET position=position+1 WHERE workbook_id=? AND position>=?",
                     (wb_id, int(snapshot["position"])))
        cells = {}
        for key, cell in snapshot.get("cells", {}).items():
            row, col = key.split(",")
            cells[(int(row), int(col))] = cell
        _insert_sheet(conn, wb_id, {
            "name": snapshot["name"], "cells": cells,
            "row_count": snapshot.get("row_count", 100),
            "col_count": snapshot.get("col_count", 20),
            "hidden_rows": snapshot.get("hidden_rows", []),
            "hidden_cols": snapshot.get("hidden_cols", []),
            "merges": snapshot.get("merges", []),
        }, int(snapshot["position"]), sheet_id=snapshot.get("id"))
        return
    if kind == "delete_sheet":
        row = conn.execute("SELECT position FROM sheets WHERE id=?", (sheet_id,)).fetchone()
        conn.execute("DELETE FROM sheets WHERE id=?", (sheet_id,))
        if row:
            conn.execute("UPDATE sheets SET position=position-1 WHERE workbook_id=? AND position>?",
                         (wb_id, row["position"]))
        return
    if kind == "rename_sheet":
        conn.execute("UPDATE sheets SET name=? WHERE id=?", (op.get("name"), sheet_id))
        return
    if kind == "move_sheet":
        _move_sheet(conn, wb_id, sheet_id, int(op.get("to") or 0))
        return
    if kind == "set_layout":
        columns = {}
        for key in ("frozen_rows", "frozen_cols", "row_count", "col_count"):
            if key in op:
                columns[key] = int(op[key])
        for key in ("hidden_rows", "hidden_cols"):
            if key in op:
                columns[key] = json.dumps(sorted(set(int(value) for value in op[key] or [])))
        if not columns:
            return
        assignments = ", ".join(f"{key}=?" for key in columns)
        conn.execute(f"UPDATE sheets SET {assignments} WHERE id=?",
                     (*columns.values(), sheet_id))
        return
    if kind == "set_merges":
        conn.execute("UPDATE sheets SET merges=? WHERE id=?",
                     (json.dumps(op.get("merges") or []), sheet_id))
        return
    raise OpError(f"Cannot replay operation: {kind}")


def sync_cells(wb_id: str, sheet_id: int, values: dict) -> None:
    """Mirror accepted corrections from a cleaning session into the workbook."""
    if not values:
        return
    with db.transaction() as conn:
        for (row, col), raw in values.items():
            cell = conn.execute(
                "SELECT formula, style FROM cells WHERE sheet_id=? AND row=? AND col=?",
                (sheet_id, row, col),
            ).fetchone()
            if cell and cell["formula"]:
                continue
            value, num = _typed(raw)
            style = json.loads(cell["style"]) if cell and cell["style"] else None
            _write_cell(conn, sheet_id, row, col, value, num, None, style)
        _recalculate(conn, wb_id)
        conn.execute(
            "UPDATE workbooks SET updated_at=?, analysis_at=?, analysis_stale=0 WHERE id=?",
            (_now(), _now(), wb_id),
        )


def save_analysis(wb_id: str, sheet_id: int, payload: dict) -> None:
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO analyses (workbook_id, sheet_id, payload, created_at)"
            " VALUES (?,?,?,?) ON CONFLICT(workbook_id) DO UPDATE SET"
            " sheet_id=excluded.sheet_id, payload=excluded.payload,"
            " created_at=excluded.created_at",
            (wb_id, int(sheet_id), json.dumps(payload), _now()),
        )
        conn.execute(
            "UPDATE workbooks SET analysis_at=?, analysis_stale=0 WHERE id=?", (_now(), wb_id)
        )


def get_analysis(wb_id: str) -> dict | None:
    with db.read() as conn:
        row = conn.execute(
            "SELECT a.sheet_id, a.payload, a.created_at, w.analysis_stale FROM analyses a"
            " JOIN workbooks w ON w.id=a.workbook_id WHERE a.workbook_id=?", (wb_id,)
        ).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row["payload"])
    except ValueError:
        return None
    payload["sheet_id"] = row["sheet_id"]
    payload["created_at"] = row["created_at"]
    payload["stale"] = bool(row["analysis_stale"])
    return payload


def clear_analysis(wb_id: str) -> None:
    with db.transaction() as conn:
        conn.execute("DELETE FROM analyses WHERE workbook_id=?", (wb_id,))


def set_analysis_status(wb_id: str, status: str, ids=None, cells=None) -> int:
    """Persist accepted/rejected/open state for stored findings."""
    if status not in ("applied", "rejected", "open"):
        raise ValueError("Unknown finding status.")
    wanted_ids = {str(item) for item in (ids or [])}
    wanted_cells = {(int(row), int(col)) for row, col in (cells or [])}
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT payload FROM analyses WHERE workbook_id=?", (wb_id,)
        ).fetchone()
        if not row:
            return 0
        payload = json.loads(row["payload"])
        changed = 0
        for issue in payload.get("issues", []):
            matches = issue.get("id") in wanted_ids or (
                issue.get("sheet_row"), issue.get("col")) in wanted_cells
            if not matches:
                continue
            issue["status"] = status
            issue["applied"] = status == "applied"
            issue["rejected"] = status == "rejected"
            changed += 1
        if changed:
            conn.execute(
                "UPDATE analyses SET payload=? WHERE workbook_id=?",
                (json.dumps(payload), wb_id),
            )
        return changed


def analysis_reports() -> list[dict]:
    with db.read() as conn:
        rows = conn.execute(
            "SELECT a.workbook_id, a.sheet_id, a.payload, a.created_at,"
            " w.name AS workbook_name, w.updated_at, w.analysis_stale,"
            " s.name AS sheet_name"
            " FROM analyses a JOIN workbooks w ON w.id=a.workbook_id"
            " LEFT JOIN sheets s ON s.id=a.sheet_id"
            " ORDER BY a.created_at DESC"
        ).fetchall()
    reports = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except ValueError:
            continue
        issues = payload.get("issues", [])
        applied = sum(1 for issue in issues if issue.get("applied") or issue.get("status") == "applied")
        rejected = sum(1 for issue in issues if issue.get("rejected") or issue.get("status") == "rejected")
        categories: dict[str, int] = {}
        for issue in issues:
            key = issue.get("category") or "other"
            categories[key] = categories.get(key, 0) + 1
        reports.append({
            "workbook_id": row["workbook_id"],
            "workbook_name": row["workbook_name"],
            "sheet_id": row["sheet_id"],
            "sheet_name": row["sheet_name"] or payload.get("sheet_name") or "Sheet",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "stale": bool(row["analysis_stale"]),
            "analyzed_rows": payload.get("analyzed_rows", 0),
            "total": len(issues),
            "applied": applied,
            "rejected": rejected,
            "categories": categories,
        })
    return reports

==============================================================================
FILE: engine.py
==============================================================================
"""DataFix cleaning engine.

Framework independent: takes a pandas DataFrame and returns a list of issues
(original value, suggested correction, confidence, evidence) plus helpers that
write the cleaned workbook and the correction report.

Design rules:
* Corrections are only proposed when the evidence is in the data itself
  (dominant formats, most frequent values, structural rules).
* Missing personal information is never invented.
* Duplicates and ambiguous cases are flagged, never silently changed.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

MAX_ISSUES = 8000
MISSING = "(blank)"

PLACEHOLDERS = {
    "na", "n/a", "n.a.", "none", "null", "nan", "nil", "-", "--", "---",
    "not available", "missing", "?", "unknown", "undefined", "",
}

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
KNOWN_DOMAINS = (
    "gmail", "yahoo", "outlook", "hotmail", "rediffmail", "icloud",
    "protonmail", "live", "msn", "aol", "zoho", "yandex",
)
KNOWN_TLDS = ["com", "org", "net", "in", "edu", "gov", "io", "co", "co.in", "us", "uk"]

CATEGORY_LABELS = {
    "case": "Capitalization",
    "spacing": "Spacing",
    "spelling": "Spelling",
    "duplicate": "Duplicates",
    "missing": "Missing data",
    "type": "Data type",
    "format": "Formatting",
    "sequence": "Sequence & pattern",
    "special": "Special characters",
    "structure": "Structure",
}

STATUS_LABELS = {
    "auto": "Auto-corrected",
    "accepted": "Approved",
    "manual": "Manual edit",
    "pending": "Needs review",
    "missing": "Missing",
    "duplicate": "Duplicate",
    "rejected": "Ignored",
    "removed": "Row removed",
    "reverted": "Reverted",
}

# Formats are parsed strictly so that ambiguous numeric dates are detectable.
DATE_FORMATS = [
    ("%Y-%m-%d", "iso"), ("%Y/%m/%d", "iso"),
    ("%d/%m/%Y", "dmy"), ("%m/%d/%Y", "mdy"),
    ("%d-%m-%Y", "dmy"), ("%m-%d-%Y", "mdy"),
    ("%d.%m.%Y", "dmy"), ("%Y.%m.%d", "iso"),
    ("%d %b %Y", "d-mon-y"), ("%d %B %Y", "d-month-y"),
    ("%b %d, %Y", "mon-d-y"), ("%B %d, %Y", "month-d-y"),
    ("%b %d %Y", "mon-d-y"), ("%B %d %Y", "month-d-y"),
    ("%d-%b-%Y", "d-mon-y"), ("%d-%b-%y", "d-mon-y"),
    ("%Y-%m-%d %H:%M:%S", "iso"), ("%Y-%m-%d %H:%M", "iso"),
    ("%d/%m/%Y %H:%M", "dmy"), ("%m/%d/%Y %H:%M", "mdy"),
]

FORMAT_HINTS = {
    "iso": "YYYY-MM-DD", "dmy": "DD/MM/YYYY", "mdy": "MM/DD/YYYY",
    "d-mon-y": "DD-Mon-YYYY", "d-month-y": "DD Month YYYY",
    "mon-d-y": "Mon DD, YYYY", "month-d-y": "Month DD, YYYY",
}

NON_NEGATIVE_HINTS = (
    "age", "price", "amount", "salary", "fee", "qty", "quantity", "count",
    "marks", "score", "spend", "budget", "cost", "total", "income",
    "weight", "height", "tax", "discount",
)
NEGATIVE_OK = ("balance", "change", "diff", "profit", "loss", "delta")

_CONFUSION = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "8": "b", "4": "a", "7": "t"})


# --------------------------------------------------------------------------- #
# Small text helpers
# --------------------------------------------------------------------------- #

def as_text(value: Any) -> str:
    """Convert a cell value to a stable display string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.hour or value.minute or value.second:
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        if number.is_integer():
            return str(int(number))
        return format(number, ".15g")
    return str(value)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def group_key(text: str) -> str:
    """Aggressive key used to find formatting variants of the same value."""
    return re.sub(r"[\s\-_./\\]+", "", text.strip().lower())


def id_shape(text: str) -> str:
    """Character-class run pattern, e.g. 26suubecs001 -> 2d6a3d."""
    out: list[str] = []
    previous = ""
    count = 0
    for char in text:
        kind = "d" if char.isdigit() else "a" if char.isalpha() else "s"
        if kind == previous:
            count += 1
        else:
            if previous:
                out.append(f"{count}{previous}")
            previous, count = kind, 1
    if previous:
        out.append(f"{count}{previous}")
    return "".join(out)


def probe_key(key: str) -> str:
    return key.translate(_CONFUSION)


def parse_number(text: str) -> float | None:
    """Parse a plain, comma-grouped or currency-prefixed number."""
    value = text.strip()
    if not value:
        return None
    negative = value.startswith("(") and value.endswith(")")
    value = value.strip("()")
    value = re.sub(r"(?i)^(?:rs\.?|inr|usd|eur|gbp)\s*", "", value)
    value = value.replace(",", "").replace("₹", "").replace("$", "").replace("€", "").replace("£", "")
    value = value.strip()
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        return None
    number = float(value)
    return -number if negative else number


def parse_date_value(text: str) -> dict | None:
    """Strictly parse a date-ish string and remember which formats matched."""
    value = text.strip()
    if not value or len(value) > 40:
        return None
    matches: list[str] = []
    parsed: list[datetime] = []
    for fmt, tag in DATE_FORMATS:
        try:
            stamp = datetime.strptime(value, fmt)
        except ValueError:
            continue
        matches.append(tag)
        parsed.append(stamp)
    if not matches:
        return None
    distinct_dates = {stamp.date() for stamp in parsed}
    has_time = any(stamp.hour or stamp.minute or stamp.second for stamp in parsed)
    return {
        "stamp": parsed[0],
        "tags": matches,
        "ambiguous": len(distinct_dates) > 1,
        "has_time": has_time,
    }


# --------------------------------------------------------------------------- #
# Issue construction
# --------------------------------------------------------------------------- #

def make_issue(column: str, row_id: int | None, category: str, error: str,
               original: str, suggestion: str | None = None,
               confidence: int | None = None, severity: str = "error",
               reason: str = "", action: str = "fix", auto: bool = False) -> dict:
    raw_id = f"{column}\x1f{row_id}\x1f{category}\x1f{original}\x1f{suggestion}"
    issue_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:12]
    return {
        "id": issue_id,
        "row_id": row_id,
        "row": (row_id + 2) if row_id is not None else None,
        "column": column,
        "category": category,
        "error": error,
        "original": original,
        "suggestion": suggestion,
        "confidence": confidence,
        "severity": severity,
        "reason": reason,
        "action": action,
        "auto": auto,
    }


# --------------------------------------------------------------------------- #
# Column kind detection
# --------------------------------------------------------------------------- #

def _name_tokens(name: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())


def detect_kind(name: str, values: list[str]) -> tuple[str, str]:
    """Return (kind, human label) for a column."""
    tokens = _name_tokens(name)
    sample = values[:600]
    total = len(sample) or 1

    email_hint = "email" in tokens or "mail" in tokens or "e-mail" in name.lower()
    phone_hint = bool(tokens & {"phone", "mobile", "contact", "cell", "tel", "telephone", "whatsapp"})
    date_hint = bool(tokens & {"date", "dob", "birth", "birthday", "joining", "admission", "expiry"}) \
        or "date" in name.lower()
    id_hint = bool(tokens & {"id", "roll", "rollno", "reg", "regno", "registration", "enrolment",
                             "enrollment", "code", "serial", "uid", "gid"}) or name.lower().endswith("id")
    numeric_hint = bool(tokens & {"age", "price", "amount", "salary", "fee", "qty", "quantity", "count",
                                  "marks", "score", "spend", "budget", "cost", "total", "income",
                                  "weight", "height", "tax", "discount", "rate", "number"})
    name_hint = bool(tokens & {"name", "names", "student", "customer", "employee", "fullname"})
    address_hint = bool(tokens & {"address", "city", "state", "country", "location", "pincode",
                                  "zip", "zipcode", "area", "district"})
    dob_hint = bool(tokens & {"dob", "birth", "birthday"})

    if not sample:
        return "text", "Text"

    email_ratio = sum(
        1 for value in sample
        if EMAIL_RE.match(value.strip()) or ("@" in value and len(value) > 5)
        or "#" in value.split(".")[0]
    ) / total
    if email_hint or email_ratio >= 0.6:
        return "email", "Email"

    date_parsed = sum(1 for value in sample if parse_date_value(value))
    if date_hint or (date_parsed / total >= 0.6):
        return "date", "Birth date" if dob_hint else "Date"

    if phone_hint:
        return "phone", "Phone"
    if id_hint:
        return "id", "ID / code"

    phone_like = sum(
        1 for value in sample
        if 7 <= len(re.sub(r"\D", "", value)) <= 15
        and len(re.sub(r"[^0-9+\-() .]", "", value)) / max(1, len(value)) >= 0.8
    )
    if phone_like / total >= 0.7:
        return "phone", "Phone"

    mixed_alnum = sum(
        1 for value in sample
        if 3 <= len(value) <= 32 and re.search(r"[A-Za-z]", value) and re.search(r"\d", value)
        and len(re.sub(r"[A-Za-z0-9\-_/ .]", "", value)) <= 1
    )
    unique_ratio = len(set(sample)) / total
    if mixed_alnum / total >= 0.75 and unique_ratio >= 0.7:
        return "id", "ID / code"

    numeric_ratio = sum(1 for value in sample if parse_number(value) is not None) / total
    if numeric_ratio >= 0.85 or (numeric_hint and numeric_ratio >= 0.4):
        return "numeric", "Number"

    if name_hint:
        return "name", "Name"
    if address_hint:
        return "address", "Address"

    unique = len(set(sample))
    if (unique <= 12 or unique_ratio <= 0.3) and unique >= 2:
        return "category", "Category"

    title_ratio = sum(
        1 for value in sample
        if value.split() and all(word[0].isupper() for word in value.split() if word[0].isalpha())
    ) / total
    if title_ratio >= 0.4 and unique_ratio >= 0.5:
        return "name", "Name"

    return "text", "Text"


# --------------------------------------------------------------------------- #
# Column analyzers
# --------------------------------------------------------------------------- #

def _analyze_text(column: str, kind: str, rows: dict[int, str], counts: Counter,
                  add) -> None:
    """Whitespace, casing, repeated words, punctuation and spelling fixes."""
    surfaces: dict[str, Counter] = defaultdict(Counter)
    row_ids_by_value: dict[str, list[int]] = defaultdict(list)
    for row_id, raw in rows.items():
        clean = normalize_spaces(raw)
        surfaces[group_key(clean)][clean] += 1
        row_ids_by_value[raw].append(row_id)

    dominant: dict[str, str] = {}
    for key, variants in surfaces.items():
        dominant[key] = max(
            variants.items(),
            key=lambda item: (
                not bool(re.search(r"[.,;:!*#'\"]+$", item[0])),
                item[1],
                item[0][:1].isupper(),
                -len(item[0]),
            ),
        )[0]

    dominant_counts = Counter({dominant[key]: sum(variants.values())
                               for key, variants in surfaces.items()})
    candidates = [value for value, count in dominant_counts.items()
                  if count >= 2 and len(value) >= 3]
    fuzzy_cutoff = 92 if kind == "name" else 88

    for raw, count in counts.items():
        row_list = row_ids_by_value[raw]
        stripped = normalize_spaces(raw)
        key = group_key(stripped)
        canonical = dominant.get(key, stripped)
        suggestion: str | None = None
        category = "spacing"
        error = ""
        confidence: int | None = None
        reason = ""
        auto = False
        severity = "error"

        if stripped != raw:
            suggestion = canonical if canonical else stripped
            error = "Leading/trailing spaces" if raw.strip() != raw else "Extra spaces between words"
            confidence = 98
            auto = True
            reason = "Spaces are not part of the value; the column pattern matches the cleaned form."
        elif re.search(r"(?i)\b(\w{2,})(?:\s+\1\b)+", stripped) and kind in ("name", "category"):
            suggestion = re.sub(r"(?i)\b(\w{2,})(?:\s+\1\b)+", r"\1", stripped)
            category = "spelling"
            error = "Repeated word"
            confidence = 96
            auto = True
            reason = "The same word is repeated inside the value."
        elif (punct := re.search(r"[.,;:!*#'\"]+$", stripped)) \
                and "." not in stripped[:punct.start()] and len(stripped[:punct.start()]) >= 2:
            base = stripped[:punct.start()]
            canonical = dominant.get(group_key(base), base)
            suggestion = (base if re.search(r"[.,;:!*#'\"]+$", canonical) else canonical)
            category = "special"
            error = "Trailing punctuation"
            confidence = 92
            auto = True
            reason = "Trailing punctuation is not used by other values in this column."
        elif canonical != raw:
            suggestion = canonical
            if raw.lower() == canonical.lower():
                category, error = "case", "Inconsistent capitalization"
                confidence = 96
            elif key == group_key(canonical):
                category, error = "spacing", "Inconsistent spacing"
                confidence = 94
            auto = True
            reason = f"'{canonical}' is the most common form of this value in the column."
        else:
            repeated = re.sub(r"([.,;:!?_-])\1+", r"\1", stripped)
            if repeated != stripped and kind in ("name", "category", "address"):
                suggestion = repeated
                category = "special"
                error = "Repeated punctuation"
                confidence = 90
                reason = "Punctuation is accidentally repeated."
            elif len(stripped) >= 4 and candidates:
                lowered = stripped.lower()
                ranked = process.extract(lowered, [c.lower() for c in candidates],
                                         scorer=fuzz.ratio, limit=2)
                cutoff = 94 if kind == "address" else fuzzy_cutoff
                if ranked and ranked[0][1] >= cutoff:
                    best_value = candidates[ranked[0][2]]
                    needed_margin = 4 if kind == "address" else 3
                    margin_ok = len(ranked) < 2 or ranked[0][1] - ranked[1][1] >= needed_margin
                    count_ok = dominant_counts[best_value] > count and (
                        kind != "address" or dominant_counts[best_value] >= 3)
                    name_ok = kind != "name" or (ranked[0][1] >= 92 and dominant_counts[best_value] >= 3)
                    if margin_ok and count_ok and name_ok and group_key(best_value) != key:
                        suggestion = best_value
                        category = "spelling"
                        error = "Possible spelling mistake"
                        confidence = min(92, int(ranked[0][1]))
                        reason = (f"Very similar to '{best_value}', which appears "
                                  f"{dominant_counts[best_value]} time(s) in this column.")
            if suggestion is None and len(stripped) >= 4:
                probe = probe_key(key)
                for other_key, other_value in dominant.items():
                    if other_key != key and probe_key(other_key) == probe and dominant_counts[other_value] >= 2:
                        suggestion = other_value
                        category = "spelling"
                        error = "Character substitution (e.g. 0/O, 1/l)"
                        confidence = 91
                        reason = f"Digits/letters are swapped compared with '{other_value}'."
                        break

        if suggestion is not None and suggestion != raw:
            for row_id in row_list:
                add(column, row_id, category, error, raw, suggestion, confidence,
                    reason, auto=auto, severity=severity)

    if kind == "name":
        seen_pairs: set[tuple[str, str]] = set()
        emitted = 0
        name_groups: dict[str, list[int]] = defaultdict(list)
        for row_id, raw in rows.items():
            name_groups[group_key(normalize_spaces(raw))].append(row_id)
        for row_list in name_groups.values():
            if len(row_list) < 2 or emitted >= 60:
                continue
            for row_id in row_list[1:]:
                add(column, row_id, "duplicate", "Repeated name", rows[row_id], None, None,
                    f"This name appears {len(row_list)} times, ignoring case and spaces. "
                    "Verify these are different people before changing anything.",
                    action="none", auto=False, severity="warning")
                emitted += 1
                if emitted >= 60:
                    break

        unique_names = [value for value in counts if len(value) >= 4 and counts[value] <= 3]
        all_names = [value for value in counts if len(value) >= 4]
        if 2 <= len(all_names) <= 600:
            lowered = [value.lower() for value in all_names]
            emitted = 0
            for value in unique_names:
                if emitted >= 10:
                    break
                best = process.extractOne(value.lower(), lowered, scorer=fuzz.ratio, score_cutoff=95)
                if not best:
                    continue
                other = all_names[best[2]]
                if group_key(other) == group_key(value) or counts[other] <= counts[value]:
                    continue
                pair = tuple(sorted((value.lower(), other.lower())))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                for row_id in row_ids_by_value[value][:1]:
                    add(column, row_id, "duplicate", "Similar name",
                        value, None, None,
                        f"Very similar to '{other}' elsewhere in the column. "
                        "Check whether these are the same person.",
                        action="none", auto=False, severity="warning")
                emitted += 1


def _email_candidate(raw: str, dominant_tld: str) -> tuple[str | None, str, int, str, bool]:
    """Return (candidate, error label, confidence, reason, auto)."""
    value = raw.strip()
    if not value:
        return None, "", 0, "", False

    if "#" in value:
        candidate = value.replace("#", "@")
        if EMAIL_RE.match(candidate):
            return candidate.lower(), "'#' instead of '@'", 95, \
                "The value becomes a valid email when '#' is replaced by '@'.", True

    if re.search(r"\s", value):
        candidate = re.sub(r"\s+", "", value)
        if EMAIL_RE.match(candidate):
            return candidate.lower(), "Spaces inside email", 95, \
                "Email addresses cannot contain spaces.", True

    if value.count("@") > 1:
        local, _, domain = value.rpartition("@")
        candidate = f"{local.replace('@', '')}@{domain}"
        if EMAIL_RE.match(candidate):
            return candidate.lower(), "Multiple '@' symbols", 90, \
                "Only one '@' is valid; the extra one was removed.", False

    if "@" not in value:
        lowered = value.lower()
        for provider in KNOWN_DOMAINS:
            marker = provider + "."
            index = lowered.find(marker)
            if index > 2:
                candidate = value[:index] + "@" + value[index:]
                if EMAIL_RE.match(candidate):
                    return candidate.lower(), "Missing '@'", 94, \
                        f"Split before the '{provider}' domain to restore the '@'.", True
        at_words = re.sub(r"(?i)\s+(?:at)\s+", "@", value)
        if at_words != value and EMAIL_RE.match(at_words):
            return at_words.lower(), "Missing '@'", 92, \
                "The word 'at' was used in place of '@'.", True

    if "@" in value:
        local, _, domain = value.partition("@")
        lowered_domain = domain.lower()
        if "." not in lowered_domain:
            for provider in KNOWN_DOMAINS:
                for tld in KNOWN_TLDS:
                    if lowered_domain == provider + tld:
                        candidate = f"{local}@{provider}.{tld}"
                        return candidate.lower(), "Missing dot in domain", 94, \
                            f"The domain is missing its dot (e.g. '{provider}.{tld}').", True
        if local and domain and "." not in lowered_domain:
            tld = dominant_tld or "com"
            candidate = f"{local}@{domain}.{tld}"
            return candidate.lower(), "Missing domain extension", 86, \
                f"The domain has no extension; '.{tld}' is the most common in this column.", False

    if "@" in value:
        local, _, domain = value.partition("@")
        if "." in domain:
            stem, _, tld = domain.rpartition(".")
            if tld and tld.lower() not in KNOWN_TLDS and tld.isalpha():
                best = process.extractOne(tld.lower(), KNOWN_TLDS, scorer=fuzz.ratio, score_cutoff=70)
                if best:
                    candidate = f"{local}@{stem}.{best[0]}"
                    if EMAIL_RE.match(candidate):
                        return candidate.lower(), "Invalid domain extension", 86, \
                            f"'.{tld}' looks like a typo of '.{best[0]}'.", False

    cleaned = value.rstrip(".,;:! ")
    if cleaned != value and EMAIL_RE.match(cleaned):
        return cleaned.lower(), "Trailing punctuation", 93, \
            "Trailing punctuation is not part of the email address.", True

    if value != value.lower() and EMAIL_RE.match(value):
        return value.lower(), "Capital letters in email", 96, \
            "Email addresses are case-insensitive; the column uses lowercase.", True

    return None, "Invalid email format", 0, "", False


def _analyze_email(column: str, rows: dict[int, str], add) -> None:
    tld_counts: Counter = Counter()
    for raw in rows.values():
        if "@" in raw and "." in raw:
            tld = raw.rsplit(".", 1)[-1].strip().lower()
            if tld.isalpha() and tld in KNOWN_TLDS:
                tld_counts[tld] += 1
    dominant_tld = tld_counts.most_common(1)[0][0] if tld_counts else "com"

    duplicates = Counter(raw.strip().lower() for raw in rows.values() if "@" in raw)
    first_row: dict[str, int] = {}
    for row_id, raw in rows.items():
        key = raw.strip().lower()
        if "@" in raw:
            first_row.setdefault(key, row_id)

    for row_id, raw in rows.items():
        candidate, error, confidence, reason, auto = _email_candidate(raw, dominant_tld)
        fixed = candidate is not None and candidate != raw
        if fixed:
            add(column, row_id, "format", error, raw, candidate, confidence,
                reason, auto=auto)
        elif not EMAIL_RE.match(raw):
            add(column, row_id, "format", "Invalid email format", raw, None, None,
                "The value is not a valid email address and cannot be repaired "
                "without guessing. Edit it manually.", action="none",
                severity="warning")

        key = raw.strip().lower()
        if (not fixed and "@" in raw and EMAIL_RE.match(raw) and duplicates[key] > 1
                and first_row.get(key) != row_id):
            add(column, row_id, "duplicate", "Duplicate email", raw, None, None,
                f"This email is also used in Excel row {first_row.get(key, 0) + 2}. "
                "Verify whether these are the same person; it was not changed automatically.",
                action="none", severity="warning")


def _canonical_phone(digits: str, style: str, country_code: str) -> str:
    if style == "plus" and country_code:
        if len(digits) > len(country_code):
            local = digits[len(country_code):]
            if len(local) == 10:
                return f"+{country_code} {local[:5]} {local[5:]}"
        return f"+{digits}"
    return digits


def _analyze_phone(column: str, rows: dict[int, str], add) -> None:
    digit_lengths = Counter()
    plus_values = 0
    plain_values = 0
    country_codes: Counter = Counter()
    valid_values = 0

    parsed: dict[int, str] = {}
    for row_id, raw in rows.items():
        digits = re.sub(r"\D", "", raw)
        parsed[row_id] = digits
        if 7 <= len(digits) <= 15:
            valid_values += 1
            digit_lengths[len(digits)] += 1
            stripped = raw.strip()
            if stripped.startswith("+"):
                plus_values += 1
                head = re.match(r"\+(\d{1,3})", stripped)
                if head:
                    country_codes[head.group(1)] += 1
            elif stripped.isdigit():
                plain_values += 1

    if not valid_values:
        return

    expected_length = digit_lengths.most_common(1)[0][0]
    style = "plus" if plus_values > plain_values else "plain"
    country_code = country_codes.most_common(1)[0][0] if country_codes else "91"

    duplicates = Counter(digits for digits in parsed.values() if len(digits) >= 7)
    first_row: dict[str, int] = {}
    for row_id, digits in parsed.items():
        if len(digits) >= 7:
            first_row.setdefault(digits, row_id)

    for row_id, raw in rows.items():
        digits = parsed[row_id]
        stripped = raw.strip()
        suggestion: str | None = None
        error = ""
        confidence = 0
        reason = ""
        auto = False
        category = "format"

        has_letters = bool(re.search(r"[A-Za-z]", stripped))
        length = len(digits)

        if has_letters and length == expected_length:
            suggestion = _canonical_phone(digits, style, country_code)
            error = "Letters inside phone number"
            confidence = 90
            reason = "Only digits are kept to form a usable phone number."
        elif length == 0:
            add(column, row_id, "format", "No digits in phone number", raw, None, None,
                "The value contains no digits. Enter the phone number manually.",
                action="none", severity="warning")
        elif length != expected_length:
            if length == expected_length + 2 and digits.startswith(country_code) and expected_length == 10:
                suggestion = _canonical_phone(digits[-10:], style, country_code)
                error = "Inconsistent country code"
                confidence = 93
                auto = True
                reason = (f"The number carries an extra '{country_code}' prefix; "
                          f"most numbers in this column have {expected_length} digits.")
            elif length == expected_length + 1 and digits.startswith("0"):
                suggestion = _canonical_phone(digits[1:], style, country_code)
                error = "Extra leading zero"
                confidence = 92
                auto = True
                reason = "The leading 0 is not used by the other phone numbers."
            else:
                error = f"Unusual digit count ({length})"
                confidence = None
                reason = (f"Most numbers in this column have {expected_length} digits. "
                          "The missing/extra digits were not guessed.")
        elif 7 <= length <= 15:
            canonical = _canonical_phone(digits, style, country_code)
            if stripped != canonical:
                suggestion = canonical
                error = "Inconsistent phone format"
                confidence = 92
                auto = not (stripped.startswith("+") and style == "plain")
                if auto:
                    reason = "Separators were removed to match the column's usual format."
                else:
                    confidence = 90
                    reason = "Most numbers in this column are stored without a country code."

        if suggestion and suggestion != stripped:
            add(column, row_id, category, error, raw, suggestion, confidence,
                reason, auto=auto)

        if (7 <= len(digits) <= 15 and duplicates[digits] > 1
                and first_row.get(digits) != row_id):
            add(column, row_id, "duplicate", "Duplicate phone number", raw, None, None,
                f"This number is also used in Excel row {first_row.get(digits, 0) + 2}. "
                "It may be shared by family members, so it was not changed automatically.",
                action="none", severity="warning")

        if length >= 7 and len(set(digits)) <= 2:
            add(column, row_id, "format", "Repeated digits", raw, None, None,
                "Almost every digit is identical — check for a placeholder number.",
                action="none", severity="warning")


def _analyze_id(column: str, rows: dict[int, str], add) -> None:
    keys: dict[str, list[int]] = defaultdict(list)
    surfaces: dict[str, Counter] = defaultdict(Counter)
    shapes: Counter = Counter()
    upper_share = lower_share = 0
    for row_id, raw in rows.items():
        key = group_key(raw)
        keys[key].append(row_id)
        surfaces[key][raw] += 1
        shapes[id_shape(raw)] += 1
        if raw.isupper():
            upper_share += 1
        if raw.islower():
            lower_share += 1

    dominant_surface: dict[str, str] = {}
    column_upper = upper_share >= 0.7 * len(rows)
    column_lower = lower_share >= 0.7 * len(rows)

    def surface_score(item: tuple[str, int]) -> tuple[int, int, int]:
        value, count = item
        clean = 0 if re.search(r"\s", value) else 1
        style = 0
        if column_upper and value.isupper():
            style = 2
        elif column_lower and value.islower():
            style = 2
        return clean, style, count

    for key, variants in surfaces.items():
        dominant_surface[key] = max(variants.items(), key=surface_score)[0]

    dominant_shape, dominant_shape_count = shapes.most_common(1)[0] if shapes else ("", 0)
    shape_ratio = dominant_shape_count / max(1, len(rows))
    no_separators = sum(1 for raw in rows.values() if not re.search(r"[\s\-_/.]", raw)) / max(1, len(rows))

    for row_id, raw in rows.items():
        key = group_key(raw)
        suggestion: str | None = None
        error = ""
        confidence: int | None = None
        reason = ""
        auto = False
        category = "format"

        canonical = dominant_surface.get(key)
        if canonical and canonical != raw:
            suggestion = canonical
            error = "Inconsistent ID formatting"
            confidence = 96
            auto = True
            reason = f"'{canonical}' is the most common form of this ID in the column."
        elif re.search(r"\s", raw):
            candidate = re.sub(r"\s+", "", raw).strip()
            alternative = dominant_surface.get(group_key(candidate))
            suggestion = (alternative if alternative and not re.search(r"\s", alternative)
                          and alternative != raw else candidate)
            error = "Hidden spaces in ID"
            confidence = 94
            auto = True
            reason = "Spaces inside an identifier are accidental."
        elif column_upper and raw != raw.upper() and not re.search(r"\d{4}-\d{2}", raw):
            suggestion = raw.upper()
            error = "Inconsistent capitalization"
            confidence = 92
            auto = True
            reason = "Identifiers in this column are written in uppercase."
        elif column_lower and raw != raw.lower():
            suggestion = raw.lower()
            error = "Inconsistent capitalization"
            confidence = 92
            auto = True
            reason = "Identifiers in this column are written in lowercase."
        elif no_separators >= 0.8 and re.search(r"[\s\-_]", raw):
            candidate = re.sub(r"[\s\-_]", "", raw)
            alternative = dominant_surface.get(group_key(candidate))
            suggestion = (alternative if alternative and not re.search(r"[\s\-_]", alternative)
                          and alternative != raw else candidate)
            error = "Unexpected separators"
            confidence = 93
            auto = True
            reason = "The column's IDs are written without separators."

        if suggestion and suggestion != raw:
            add(column, row_id, category, error, raw, suggestion, confidence,
                reason, auto=auto)

        if shapes[id_shape(raw)] == 1 and shape_ratio >= 0.6 and len(rows) >= 5 \
                and id_shape(raw) != dominant_shape and suggestion is None:
            add(column, row_id, "sequence", "ID does not match column pattern", raw, None, None,
                f"Most IDs follow the shape {dominant_shape} (e.g. letters + digits); "
                "this one does not. It was not changed because that could alter identity.",
                action="none", severity="warning")

    for key, row_list in keys.items():
        if len(row_list) > 1:
            for row_id in row_list[1:]:
                add(column, row_id, "duplicate", "Duplicate ID", rows[row_id], None, None,
                    f"The same ID appears {len(row_list)} times. Duplicate identifiers "
                    "affect record matching, so they were not changed automatically.",
                    action="none", severity="warning")

    # Sequence gaps: prefix + numeric suffix.
    groups: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for row_id, raw in rows.items():
        match = re.match(r"^(.*?)(\d{1,6})$", raw.strip())
        if not match:
            continue
        prefix, suffix = match.group(1).lower(), int(match.group(2))
        groups[prefix].append((row_id, suffix, raw))
    for prefix, members in groups.items():
        if len(members) < 5:
            continue
        members.sort(key=lambda item: item[1])
        for index in range(1, len(members)):
            previous_id, previous_number, _ = members[index - 1]
            current_id, current_number, current_raw = members[index]
            gap = current_number - previous_number
            if 1 < gap <= 20:
                add(column, current_id, "sequence", "Missing ID in sequence", current_raw, None, None,
                    f"No ID exists between #{previous_number} and #{current_number}. "
                    "Check whether a record is missing.",
                    action="none", severity="warning")
                break


def _analyze_date(column: str, rows: dict[int, str], add, dob_hint: bool) -> None:
    parsed: dict[int, dict] = {}
    tag_counts: Counter = Counter()
    for row_id, raw in rows.items():
        result = parse_date_value(raw)
        if result:
            parsed[row_id] = result
            if not result["ambiguous"] and len(result["tags"]) == 1:
                tag_counts[result["tags"][0]] += 1
    dominant_tag = tag_counts.most_common(1)[0][0] if tag_counts else None
    dominant_label = FORMAT_HINTS.get(dominant_tag, "YYYY-MM-DD")

    today = date.today()
    for row_id, raw in rows.items():
        result = parsed.get(row_id)
        if not result:
            add(column, row_id, "format", "Invalid date", raw, None, None,
                "This value is not a valid calendar date. Enter it manually.",
                action="none", severity="warning")
            continue
        if result["ambiguous"]:
            add(column, row_id, "format", "Ambiguous date order", raw, None, None,
                "Both DD/MM/YYYY and MM/DD/YYYY are possible. Ordering was not "
                "changed because the intended date is unknown.",
                action="none", severity="warning")
            continue

        stamp: datetime = result["stamp"]
        if dob_hint and stamp.date() > today:
            add(column, row_id, "format", "Future birth date", raw, None, None,
                "The birth date is in the future. This looks like a typo.",
                action="none", severity="warning")
            continue
        if dob_hint and (today - stamp.date()).days > 110 * 365:
            add(column, row_id, "format", "Unusual birth date", raw, None, None,
                "The birth date implies an age above 110; check the year.",
                action="none", severity="warning")
            continue

        iso = stamp.strftime("%Y-%m-%d %H:%M:%S") if result["has_time"] else stamp.strftime("%Y-%m-%d")
        if iso != raw.strip():
            reason = "Date normalised to YYYY-MM-DD."
            if dominant_tag and dominant_tag not in result["tags"]:
                reason += f" The column mostly uses {dominant_label}."
            add(column, row_id, "format", "Inconsistent date format", raw, iso, 92,
                reason, auto=True)


def _analyze_numeric(column: str, series: pd.Series, rows: dict[int, str], add) -> None:
    lowered = column.lower()
    non_negative = any(hint in lowered for hint in NON_NEGATIVE_HINTS) \
        and not any(word in lowered for word in NEGATIVE_OK)

    values: list[tuple[int, float]] = []
    converted: set[int] = set()
    for row_id, raw in rows.items():
        number = parse_number(raw)
        if number is None:
            add(column, row_id, "type", "Text in numeric column", raw, None, None,
                "This column is mostly numeric but this value cannot be read as a "
                "number. It was not changed.", action="none", severity="warning")
            continue
        values.append((row_id, number))
        original = series.iloc[row_id]
        if not _is_number(original):
            suggestion = str(int(number)) if float(number).is_integer() else format(number, ".15g")
            reason = "The value contains a number; it was converted to a numeric cell."
            if raw.strip() != suggestion:
                reason = "Currency symbols, commas or spaces were removed to make it a number."
            add(column, row_id, "type", "Number stored as text", raw, suggestion, 96,
                reason, auto=True)
            converted.add(row_id)

    if len(values) >= 8:
        numbers = np.array([number for _, number in values], dtype=float)
        median = float(np.median(numbers))
        if median > 0:
            for row_id, number in values:
                if number > median * 10 and number - median > 1000:
                    add(column, row_id, "format", "Statistical outlier",
                        rows[row_id], None, None,
                        f"{number:g} is more than 10x the column median ({median:g}). "
                        "Check for extra zeros; it was not changed automatically.",
                        action="none", severity="warning")

    for row_id, number in values:
        if non_negative and number < 0:
            add(column, row_id, "format", "Negative value not expected", rows[row_id], None, None,
                "Negative values are unusual for this kind of column. "
                "It was not changed automatically.", action="none", severity="warning")
        if row_id not in converted and re.fullmatch(r"0\d+", rows[row_id].strip() or "x"):
            suggestion = str(int(number))
            add(column, row_id, "format", "Extra leading zeros", rows[row_id], suggestion, 86,
                "Leading zeros are not meaningful for a number; verify this is not an ID.",
                auto=False)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


# --------------------------------------------------------------------------- #
# Duplicate rows / structural checks
# --------------------------------------------------------------------------- #

def _duplicate_row_issues(df: pd.DataFrame, add) -> None:
    if len(df) < 2:
        return
    keys: dict[tuple, list[int]] = defaultdict(list)
    for row_id in range(len(df)):
        row_key = tuple(as_text(value) for value in df.iloc[row_id].tolist())
        if all(part == "" for part in row_key):
            continue
        keys[row_key].append(row_id)
    for row_ids in keys.values():
        if len(row_ids) < 2:
            continue
        first = row_ids[0]
        for row_id in row_ids[1:]:
            add("", row_id, "duplicate", "Duplicate row", "",
                None, None,
                f"This row is identical to Excel row {first + 2}. It was not removed "
                "automatically because deletion needs your approval.",
                action="remove_row", auto=False, severity="warning")


def _empty_row_issues(df: pd.DataFrame, blank_rows: set[int], add) -> None:
    for row_id in blank_rows:
        add("", row_id, "structure", "Empty row", "", None, None,
            "Every cell in this row is empty; it adds no data.",
            action="remove_row", auto=True)


def _header_issues(raw_headers: list[str], columns: list[str], add) -> None:
    if raw_headers:
        cleaned = [normalize_spaces(str(name)) for name in raw_headers]
        counts = Counter(name for name in cleaned if name and not name.lower().startswith("unnamed"))
        for name, count in counts.items():
            if count > 1:
                add(name, None, "structure", "Duplicate column name", name, None, None,
                    f"{count} columns share this name; DataFix separated them while reading.",
                    action="none", severity="warning")
        for index, name in enumerate(cleaned, start=1):
            if not name or name.lower().startswith("unnamed") or name.lower() == "nan":
                add(columns[index - 1] if index <= len(columns) else "", None,
                    "structure", "Missing column header", name or "(empty)", None, None,
                    "The column has no header. Add a clear name in the source file.",
                    action="none", severity="warning")


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #

def analyze(df: pd.DataFrame, raw_headers: list[str] | None = None) -> dict:
    issues: list[dict] = []
    truncated = [False]

    def add(column: str, row_id: int | None, category: str, error: str,
            original: str, suggestion: str | None = None,
            confidence: int | None = None, reason: str = "", auto: bool = False,
            severity: str = "error", action: str = "fix") -> None:
        if len(issues) >= MAX_ISSUES:
            truncated[0] = True
            return
        issues.append(make_issue(column, row_id, category, error, original,
                                 suggestion, confidence, severity, reason, action, auto))

    columns = [str(column) for column in df.columns]
    row_count = len(df)
    if row_count == 0:
        return {"issues": [], "kinds": {}, "labels": {}, "profile": [], "truncated": False}

    blank_row_ids: set[int] = set()
    for row_id in range(row_count):
        if all(as_text(value) == "" for value in df.iloc[row_id].tolist()):
            blank_row_ids.add(row_id)

    _duplicate_row_issues(df, add)
    _empty_row_issues(df, blank_row_ids, add)
    _header_issues(raw_headers or [], columns, add)

    kinds: dict[str, str] = {}
    labels: dict[str, str] = {}
    profile: list[dict] = []

    for column_index, column in enumerate(columns):
        series = df[column]
        raw_by_row: dict[int, str] = {}
        for row_id in range(row_count):
            raw_by_row[row_id] = as_text(series.iloc[row_id])

        blank_ids = [row_id for row_id, text in raw_by_row.items() if text == ""]
        placeholder_ids = [row_id for row_id, text in raw_by_row.items()
                           if text and text.strip().lower() in PLACEHOLDERS]
        missing_ids = set(blank_ids) | set(placeholder_ids)

        for row_id in blank_ids:
            if row_id in blank_row_ids:
                continue
            add(column, row_id, "missing", "Empty cell", "", None, None,
                "No value was entered. DataFix never invents missing personal information.",
                action="none", severity="warning")

        for row_id in placeholder_ids:
            add(column, row_id, "missing", "Placeholder value", raw_by_row[row_id], None, None,
                f"'{raw_by_row[row_id]}' is used as a placeholder for an empty value.",
                action="none", severity="warning")

        values = {row_id: text for row_id, text in raw_by_row.items()
                  if row_id not in missing_ids}
        counts = Counter(values.values())

        if not values:
            kind, label = "text", "Text"
            add(column, None, "structure", "Empty column", "", None, None,
                "This column has no data at all.", action="none", severity="warning")
        else:
            kind, label = detect_kind(column, list(values.values()))
            tokens = _name_tokens(column)
            dob_hint = bool(tokens & {"dob", "birth", "birthday"})

            if kind == "email":
                _analyze_email(column, values, add)
            elif kind == "phone":
                _analyze_phone(column, values, add)
            elif kind == "date":
                _analyze_date(column, values, add, dob_hint)
            elif kind == "id":
                _analyze_id(column, values, add)
            elif kind == "numeric":
                _analyze_numeric(column, series, values, add)
            else:
                _analyze_text(column, kind, values, counts, add)

        kinds[column] = kind
        labels[column] = label

        filled = sum(1 for text in raw_by_row.values() if text != "")
        unique = len({text for text in raw_by_row.values() if text != ""})
        dominant_values = Counter(text for text in raw_by_row.values() if text != "")
        profile.append({
            "name": column,
            "index": column_index,
            "kind": kind,
            "label": label,
            "filled": round(100 * filled / max(1, row_count)),
            "unique": unique,
            "dominant": (dominant_values.most_common(1)[0][0][:80]
                         if dominant_values else ""),
            "issues": 0,
        })

    # One issue per cell/category/original/suggestion, deterministic order.
    deduped: dict[tuple, dict] = {}
    for issue in issues:
        key = (issue["row_id"], issue["column"], issue["category"],
               issue["original"], issue["suggestion"])
        deduped.setdefault(key, issue)

    column_order = {name: index for index, name in enumerate(columns)}
    final = sorted(
        deduped.values(),
        key=lambda item: (
            item["row_id"] if item["row_id"] is not None else -1,
            column_order.get(item["column"], -1),
            item["error"],
        ),
    )

    issue_counts = Counter(issue["column"] for issue in final)
    for entry in profile:
        entry["issues"] = issue_counts.get(entry["name"], 0)

    return {
        "issues": final,
        "kinds": kinds,
        "labels": labels,
        "profile": profile,
        "truncated": truncated[0],
    }


# --------------------------------------------------------------------------- #
# Value coercion / exports
# --------------------------------------------------------------------------- #

def coerce_value(kind: str, value: Any) -> Any:
    """Convert a suggestion to the value stored in the working data."""
    if value is None:
        return None
    if kind == "numeric":
        number = parse_number(str(value))
        if number is None:
            return str(value)
        return int(number) if float(number).is_integer() else number
    if kind in ("id", "phone"):
        return str(value)
    return str(value)


def _excel_value(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if kind == "date":
        parsed = parse_date_value(as_text(value))
        if parsed:
            return parsed["stamp"]
        return as_text(value)
    if kind == "numeric" and not isinstance(value, (int, float, np.integer, np.floating)):
        number = parse_number(as_text(value))
        if number is not None:
            return int(number) if number.is_integer() else number
        return as_text(value)
    if kind in ("id", "phone", "email"):
        return as_text(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _safe_sheet_title(title: str) -> str:
    cleaned = re.sub(r"[\\/*?:\[\]]", "", title or "").strip()
    return (cleaned or "Cleaned data")[:31]


def build_cleaned_workbook(df: pd.DataFrame, deleted_rows: set[int], changes: dict,
                           kinds: dict[str, str], sheet_title: str = "Clean data") -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = _safe_sheet_title(sheet_title)

    columns = [str(column) for column in df.columns]
    sheet.append(columns)
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF", size=11)
        cell.fill = PatternFill("solid", fgColor="6D5BFF")
        cell.alignment = Alignment(horizontal="left", vertical="center")

    excel_row = 2
    for row_id in range(len(df)):
        if row_id in deleted_rows:
            continue
        row_values = []
        for column in columns:
            value = df.iloc[row_id][column]
            row_values.append(_excel_value(value, kinds.get(column, "text")))
        sheet.append(row_values)

        for column_index, column in enumerate(columns, start=1):
            change = changes.get((row_id, column))
            if not change:
                continue
            cell = sheet.cell(row=excel_row, column=column_index)
            if change["status"] == "manual":
                cell.fill = PatternFill("solid", fgColor="FFF3CD")
            else:
                cell.fill = PatternFill("solid", fgColor="D4EDDA")
        excel_row += 1

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for index, column in enumerate(columns, start=1):
        width = min(42, max(12, len(str(column)) + 4, 16))
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.row_dimensions[1].height = 22

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def build_report_workbook(issues: list[dict], stats: dict, meta: dict,
                          profile: list[dict]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"

    title_font = Font(bold=True, size=16, color="1F2937")
    label_font = Font(bold=True, size=10, color="475569")
    header_fill = PatternFill("solid", fgColor="6D5BFF")
    header_font = Font(bold=True, color="FFFFFF", size=10)
    thin = Side(style="thin", color="E2E8F0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    summary["A1"] = "DataFix correction report"
    summary["A1"].font = title_font
    summary["A2"] = f"File: {meta.get('file_name', '')}"
    summary["A3"] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    summary["A2"].font = label_font
    summary["A3"].font = label_font
    summary.column_dimensions["A"].width = 34
    summary.column_dimensions["B"].width = 24
    summary.column_dimensions["C"].width = 60

    summary_rows = [
        ("Total rows (active)", stats.get("rows")),
        ("Columns", stats.get("columns")),
        ("Cells analysed", stats.get("cells")),
        ("Errors detected", stats.get("errors_detected")),
        ("Correct records", stats.get("correct_records")),
        ("Incorrect records", stats.get("incorrect_records")),
        ("Duplicate rows", stats.get("duplicates")),
        ("Missing values", stats.get("missing")),
        ("Auto-corrected", stats.get("auto_corrected")),
        ("Approved corrections", stats.get("approved")),
        ("Manual edits", stats.get("manual")),
        ("Needs approval", stats.get("needs_review")),
        ("Ignored", stats.get("ignored")),
        ("Data health", f"{stats.get('health', 0)}%"),
    ]
    start = 5
    summary.cell(row=start, column=1, value="Metric").font = header_font
    summary.cell(row=start, column=2, value="Value").font = header_font
    summary.cell(row=start, column=1).fill = header_fill
    summary.cell(row=start, column=2).fill = header_fill
    for offset, (label, value) in enumerate(summary_rows, start=1):
        summary.cell(row=start + offset, column=1, value=label).font = label_font
        summary.cell(row=start + offset, column=2, value=value)
        summary.cell(row=start + offset, column=1).border = border
        summary.cell(row=start + offset, column=2).border = border

    notes_row = start + len(summary_rows) + 3
    summary.cell(row=notes_row, column=1, value="Notes").font = Font(bold=True, size=11)
    notes = [
        "Corrected cells are highlighted in the cleaned workbook (green = automatic/approved, amber = manual edit).",
        "Missing personal information is never invented; those cells are listed for manual entry.",
        "Duplicate rows are only removed after explicit approval.",
    ]
    for offset, note in enumerate(notes, start=1):
        summary.cell(row=notes_row + offset, column=1, value=f"• {note}")

    # Corrections sheet ------------------------------------------------------
    sheet = workbook.create_sheet("Corrections")
    headers = ["Excel row", "Column", "Original value", "Detected error",
               "Category", "Suggested correction", "Confidence", "Status", "Why"]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center")
    sheet.freeze_panes = "A2"

    status_fills = {
        "auto": "D4EDDA", "accepted": "D4EDDA", "manual": "FFF3CD",
        "rejected": "E5E7EB", "removed": "E5E7EB", "missing": "FEE2E2",
        "pending": "FEF3C7", "duplicate": "FEF3C7", "reverted": "E5E7EB",
    }
    sorted_issues = sorted(
        issues,
        key=lambda item: (item.get("row_id") if item.get("row_id") is not None else 10**9,
                          str(item.get("column", ""))),
    )
    for issue in sorted_issues:
        status = issue.get("status", "pending")
        sheet.append([
            issue.get("row") or "—",
            issue.get("column") or "Whole row",
            issue.get("original") or "",
            issue.get("error", ""),
            CATEGORY_LABELS.get(issue.get("category", ""), issue.get("category", "")),
            issue.get("suggestion") or "",
            (f"{issue['confidence']}%" if issue.get("confidence") is not None else "—"),
            STATUS_LABELS.get(status, status),
            issue.get("reason", ""),
        ])
        row = sheet.max_row
        colour = status_fills.get(status)
        if colour:
            sheet.cell(row=row, column=8).fill = PatternFill("solid", fgColor=colour)
        for column in range(1, len(headers) + 1):
            sheet.cell(row=row, column=column).border = border
            sheet.cell(row=row, column=column).alignment = Alignment(vertical="top", wrap_text=column == 9)

    widths = [10, 18, 26, 24, 16, 26, 11, 15, 52]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.auto_filter.ref = sheet.dimensions

    # Column profile ---------------------------------------------------------
    profile_sheet = workbook.create_sheet("Columns")
    profile_headers = ["Column", "Detected type", "Filled %", "Unique values",
                       "Most common value", "Issues"]
    profile_sheet.append(profile_headers)
    for cell in profile_sheet[1]:
        cell.font = header_font
        cell.fill = header_fill
    for entry in profile:
        profile_sheet.append([
            entry["name"], entry["label"], entry["filled"], entry["unique"],
            entry["dominant"], entry["issues"],
        ])
    for index, width in enumerate([24, 18, 10, 14, 30, 10], start=1):
        profile_sheet.column_dimensions[get_column_letter(index)].width = width
    profile_sheet.freeze_panes = "A2"

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()

==============================================================================
FILE: requirements.txt
==============================================================================
flask>=3.0
pandas>=2.0
numpy>=1.26
openpyxl>=3.1
rapidfuzz>=3.0
xlrd>=2.0
odfpy>=1.4
xlwt>=1.3
reportlab>=4.0
python-docx>=1.1

==============================================================================
FILE: run.bat
==============================================================================
@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from python.org and try again.
  pause
  exit /b 1
)

python -c "import flask, pandas, openpyxl, rapidfuzz, xlrd, odf, xlwt, reportlab, docx" >nul 2>nul
if errorlevel 1 (
  echo Installing required packages...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Could not install the required packages.
    pause
    exit /b 1
  )
)

echo.
echo DataFix is starting at http://127.0.0.1:5000
echo Press Ctrl+C in this window to stop it.
echo.
start "" http://127.0.0.1:5000
python app.py

==============================================================================
FILE: static/app.js
==============================================================================
/* DataFix — front-end controller (vanilla JS, no build step). */
(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));

  const fmt = (value) => new Intl.NumberFormat().format(Number(value) || 0);

  const prettySize = (bytes) => {
    if (!bytes) return "0 KB";
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const MAX_MB = Number($('meta[name="datafix-max-mb"]')?.content || 50);
  const ALLOWED = [".xlsx", ".xls", ".xlsm", ".csv"];

  const S = {
    data: null,
    rows: null,
    page: 1,
    mode: "all",
    search: "",
    issueFilter: "needs",
    issueCat: "all",
    issueSearch: "",
    issueLimit: 40,
    file: null,
  };

  const ICONS = {
    rows: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><rect x="3.5" y="3.5" width="17" height="17" rx="3"/><path d="M3.5 9.5h17M3.5 14.5h17"/></svg>',
    columns: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><rect x="3.5" y="3.5" width="17" height="17" rx="3"/><path d="M9.5 3.5v17M14.5 3.5v17"/></svg>',
    correct: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8.5"/><path d="M8.2 12.3l2.6 2.6 5-5.2"/></svg>',
    incorrect: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.8l8.6 15H3.4L12 3.8z"/><path d="M12 10v4.2M12 16.8h.01"/></svg>',
    duplicate: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><rect x="8.5" y="8.5" width="11.5" height="11.5" rx="2.5"/><path d="M16 8.5V6.2A2.2 2.2 0 0 0 13.8 4H6.2A2.2 2.2 0 0 0 4 6.2v7.6a2.2 2.2 0 0 0 2.2 2.2h2.3"/></svg>',
    missing: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><circle cx="12" cy="12" r="8.5"/><path d="M8 12h8"/></svg>',
    auto: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M13 3L5.5 13.5H11L9.8 21 18 10.2h-5.6L13 3z"/></svg>',
    review: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4 4"/></svg>',
    download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><path d="M12 4v11M7.8 11l4.2 4.2L16.2 11"/><path d="M5 19h14"/></svg>',
    spark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"/></svg>',
  };

  const TITLES = {
    upload: ["Upload workbook", "Find mistakes, approve fixes, export clean data."],
    overview: ["Overview", "A quick read on the health of this workbook."],
    review: ["Review & compare", "Original values next to suggested corrections."],
    issues: ["Corrections", "Approve, reject or undo every detected fix."],
    downloads: ["Downloads", "Clean data, the correction report and your original file."],
    appearance: ["Theme settings", "Appearance mode, colour palettes and saved preferences."],
  };

  /* --------------------------------------------------------------------- */
  /* Boot                                                                  */
  /* --------------------------------------------------------------------- */

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    boot();
  });

  async function boot() {
    try {
      const response = await fetch("/api/state");
      const data = await response.json();
      if (response.ok && data && !data.empty) {
        applyData(data);
        showPanel("overview");
      }
    } catch (_error) {
      /* start on the upload screen */
    }
  }

  function bindEvents() {
    // Upload ------------------------------------------------------------
    $("#browse-btn").addEventListener("click", (event) => {
      event.stopPropagation();
      $("#file-input").click();
    });
    $("#dropzone").addEventListener("click", () => $("#file-input").click());
    $("#dropzone").addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        $("#file-input").click();
      }
    });
    $("#hero-cta").addEventListener("click", () => {
      $("#dropzone").scrollIntoView({ behavior: "smooth", block: "center" });
      $("#file-input").click();
    });
    $("#file-input").addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (file) setFile(file);
    });
    $("#clear-file").addEventListener("click", clearFile);
    $("#analyze-btn").addEventListener("click", analyzeFile);
    $("#back-btn").addEventListener("click", () => showPanel("overview"));
    $("#new-file-btn").addEventListener("click", () => {
      showPanel("upload");
      $("#back-btn").hidden = !S.data;
      $("#dropzone").scrollIntoView({ behavior: "smooth", block: "center" });
    });

    const dropzone = $("#dropzone");
    ["dragenter", "dragover"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragging");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragging");
      });
    });
    dropzone.addEventListener("drop", (event) => {
      const file = event.dataTransfer?.files?.[0];
      if (file) setFile(file);
    });

    // Navigation ---------------------------------------------------------
    $$(".nav-item").forEach((button) => {
      button.addEventListener("click", () => showPanel(button.dataset.panel));
    });
    $$("[data-jump]").forEach((button) => {
      button.addEventListener("click", () => showPanel(button.dataset.jump));
    });
    window.addEventListener("datafix:toast", (event) => {
      const detail = event.detail || {};
      toast(detail.message || "", detail.kind || "ok");
    });

    // Table --------------------------------------------------------------
    $("#data-table").addEventListener("dblclick", (event) => {
      const cell = event.target.closest("td[data-col]");
      if (cell) startEdit(cell);
    });
    $("#prev-page").addEventListener("click", () => goPage(S.page - 1));
    $("#next-page").addEventListener("click", () => goPage(S.page + 1));
    $("#table-mode").addEventListener("change", (event) => {
      S.mode = event.target.value;
      S.page = 1;
      goPage(1);
    });
    let tableTimer = null;
    $("#table-search").addEventListener("input", (event) => {
      clearTimeout(tableTimer);
      tableTimer = setTimeout(() => {
        S.search = event.target.value.trim();
        S.page = 1;
        goPage(1);
      }, 320);
    });

    // Issues -------------------------------------------------------------
    $("#issue-tabs").addEventListener("click", (event) => {
      const chip = event.target.closest("[data-filter]");
      if (!chip) return;
      S.issueFilter = chip.dataset.filter;
      S.issueLimit = 40;
      renderIssues();
    });
    $("#issue-category").addEventListener("change", (event) => {
      S.issueCat = event.target.value;
      S.issueLimit = 40;
      renderIssues();
    });
    let issueTimer = null;
    $("#issue-search").addEventListener("input", (event) => {
      clearTimeout(issueTimer);
      issueTimer = setTimeout(() => {
        S.issueSearch = event.target.value.trim().toLowerCase();
        S.issueLimit = 40;
        renderIssues();
      }, 260);
    });
    $("#issue-more").addEventListener("click", () => {
      S.issueLimit += 40;
      renderIssues();
    });
    $("#issue-list").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      runIssueAction(button.dataset.action, button.dataset.id, button.dataset.row);
    });
    $("#accept-all").addEventListener("click", acceptAll);
    ["#undo-btn", "#undo-btn-2"].forEach((id) => $(id).addEventListener("click", undo));
    ["#reset-btn", "#reset-btn-2"].forEach((id) => $(id).addEventListener("click", resetAll));
  }

  /* --------------------------------------------------------------------- */
  /* Upload                                                                */
  /* --------------------------------------------------------------------- */

  function setFile(file) {
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED.includes(extension)) {
      showUploadError("That file type is not supported. Choose .xlsx, .xls or .csv.");
      return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      showUploadError(`This file is larger than ${MAX_MB} MB.`);
      return;
    }
    if (file.size === 0) {
      showUploadError("That file is empty.");
      return;
    }
    S.file = file;
    hideUploadError();
    $("#file-badge").textContent = extension.replace(".", "").toUpperCase().slice(0, 4);
    $("#file-name").textContent = file.name;
    $("#file-size").textContent = prettySize(file.size);
    $("#file-chip").hidden = false;
    $("#analyze-btn").disabled = false;
    $("#upload-hint").textContent = "Ready when you are.";
  }

  function clearFile() {
    S.file = null;
    $("#file-input").value = "";
    $("#file-chip").hidden = true;
    $("#analyze-btn").disabled = true;
    $("#upload-hint").textContent = "Nothing leaves your computer.";
    hideUploadError();
  }

  function showUploadError(message) {
    const box = $("#upload-error");
    box.textContent = message;
    box.hidden = false;
  }

  function hideUploadError() {
    $("#upload-error").hidden = true;
  }

  const OVERLAY_STEPS = [
    ["Reading your workbook…", "Opening the first sheet and keeping the original untouched."],
    ["Detecting column types…", "Names, emails, phones, IDs, dates, categories and numbers."],
    ["Learning dominant patterns…", "Comparing every value with what is normal in its column."],
    ["Checking for duplicates…", "Rows, emails, phone numbers and identifiers."],
    ["Scoring corrections…", "High-confidence fixes are applied; the rest wait for you."],
  ];

  async function analyzeFile() {
    if (!S.file) return;
    const form = new FormData();
    form.append("file", S.file);
    showOverlay();
    $("#analyze-btn").disabled = true;
    try {
      const response = await fetch("/api/analyze", { method: "POST", body: form });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "Upload failed.");
      clearFile();
      S.mode = "all";
      S.search = "";
      S.page = 1;
      S.issueFilter = "needs";
      S.issueCat = "all";
      S.issueSearch = "";
      S.issueLimit = 40;
      $("#table-mode").value = "all";
      $("#table-search").value = "";
      $("#issue-search").value = "";
      applyData(data);
      showPanel("overview");
      toast(`Analyzed ${fmt(data.stats.rows)} rows — ${fmt(data.stats.errors_detected)} issues found.`, "ok");
    } catch (error) {
      showUploadError(error.message || "Something went wrong while analyzing the file.");
      toast(error.message || "Upload failed.", "bad");
    } finally {
      hideOverlay();
      $("#analyze-btn").disabled = !S.file;
    }
  }

  let overlayTimer = null;
  function showOverlay() {
    const overlay = $("#overlay");
    overlay.hidden = false;
    let step = 0;
    const paint = () => {
      const [title, sub] = OVERLAY_STEPS[step % OVERLAY_STEPS.length];
      $("#overlay-title").textContent = title;
      $("#overlay-sub").textContent = sub;
      step += 1;
    };
    paint();
    overlayTimer = setInterval(paint, 1400);
  }

  function hideOverlay() {
    clearInterval(overlayTimer);
    $("#overlay").hidden = true;
  }

  /* --------------------------------------------------------------------- */
  /* Views / state                                                         */
  /* --------------------------------------------------------------------- */

  function showPanel(name) {
    const appearance = name === "appearance";
    $("#view-upload").hidden = name !== "upload";
    $("#view-dashboard").hidden = name === "upload" || appearance;
    $("#view-appearance").hidden = !appearance;
    $$(".panel").forEach((panel) => {
      panel.classList.toggle("active", panel.dataset.panel === name);
    });
    $$(".nav-item").forEach((button) => {
      button.classList.toggle("active", button.dataset.panel === name);
    });
    const [title, sub] = TITLES[name] || TITLES.upload;
    $("#page-title").textContent = title;
    $("#page-sub").textContent = sub;
    const hasData = Boolean(S.data);
    $("#undo-btn").hidden = !hasData || name === "upload" || appearance;
    $("#reset-btn").hidden = !hasData || name === "upload" || appearance;
    $("#quick-download").hidden = !hasData || name === "upload" || appearance;
    $("#back-btn").hidden = !hasData || name !== "upload";
    if (name !== "upload") window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function applyData(data) {
    S.data = data;
    S.rows = data.rows;
    S.page = data.rows.page;
    document.body.classList.add("has-data");
    $$(".nav-item").forEach((button) => { button.disabled = false; });

    $("#side-file").textContent = data.meta.file_name;
    $("#side-meta").textContent =
      `${fmt(data.stats.rows)} rows × ${fmt(data.stats.columns)} cols · ${prettySize(data.meta.size)}`;

    const onUpload = !$("#view-upload").hidden;
    $("#undo-btn").hidden = onUpload;
    $("#reset-btn").hidden = onUpload;
    $("#quick-download").hidden = onUpload;
    $("#undo-btn").disabled = !data.can_undo;
    $("#undo-btn-2").disabled = !data.can_undo;
    $("#undo-btn").title = data.can_undo ? `Undo: ${data.last_action}` : "Nothing to undo yet";

    renderWelcome();
    renderStats();
    renderCategories();
    renderProfile();
    renderTable();
    renderIssues();
    renderDownloads();
  }

  /* --------------------------------------------------------------------- */
  /* Dashboard rendering                                                   */
  /* --------------------------------------------------------------------- */

  function renderWelcome() {
    const { meta, stats } = S.data;
    $("#welcome-title").textContent = `${meta.file_name} is ready`;
    const pieces = [
      `${fmt(stats.rows)} rows and ${fmt(stats.columns)} columns checked.`,
      stats.needs_review
        ? `${fmt(stats.needs_review)} corrections need your approval.`
        : "No corrections are waiting for approval.",
    ];
    if (meta.truncated) pieces.push("The issue list was capped for very large files.");
    $("#welcome-sub").textContent = pieces.join(" ");

    const ring = $("#health-ring");
    ring.style.setProperty("--p", stats.health);
    ring.classList.toggle("low", stats.health < 60);
    ring.classList.toggle("mid", stats.health >= 60 && stats.health < 85);
    $("#health-value").textContent = `${stats.health}%`;
    $("#health-note").textContent =
      `${fmt(stats.correct_records)} of ${fmt(stats.rows)} records have no open issues.`;

    $("#welcome-review").textContent = stats.needs_review
      ? "See what needs review"
      : "View all corrections";
  }

  function renderStats() {
    const stats = S.data.stats;
    const cards = [
      { label: "Rows", value: fmt(stats.rows), sub: `${fmt(stats.deleted_rows)} removed`, icon: "rows", tone: "violet" },
      { label: "Columns", value: fmt(stats.columns), sub: `${fmt(stats.cells)} cells analysed`, icon: "columns", tone: "cyan" },
      { label: "Correct records", value: fmt(stats.correct_records), sub: "no open issues", icon: "correct", tone: "green" },
      { label: "Incorrect records", value: fmt(stats.incorrect_records), sub: "need attention", icon: "incorrect", tone: "rose" },
      { label: "Duplicate rows", value: fmt(stats.duplicates), sub: "flagged, not deleted", icon: "duplicate", tone: "amber" },
      { label: "Missing values", value: fmt(stats.missing), sub: "never invented", icon: "missing", tone: "rose" },
      { label: "Auto-corrected", value: fmt(stats.auto_corrected), sub: `${fmt(stats.approved)} approved · ${fmt(stats.manual)} manual`, icon: "auto", tone: "green" },
      { label: "Needs approval", value: fmt(stats.needs_review), sub: "waiting for you", icon: "review", tone: "amber" },
    ];
    $("#stats").innerHTML = cards.map((card) => `
      <div class="stat tone-${card.tone}">
        <span class="stat-icon">${ICONS[card.icon]}</span>
        <div class="stat-body">
          <span class="stat-label">${card.label}</span>
          <strong class="stat-value">${card.value}</strong>
          <small>${card.sub}</small>
        </div>
      </div>`).join("");
  }

  function renderCategories() {
    const categories = S.data.categories;
    const total = categories.reduce((sum, entry) => sum + entry.count, 0);
    $("#cat-total").textContent = `${fmt(total)} open`;
    $("#cat-note").textContent = total
      ? "Issues still waiting for a decision."
      : "Everything detected has been handled.";
    if (!categories.length) {
      $("#cat-bars").innerHTML = `
        <div class="empty-panel">
          <span class="empty-icon">${ICONS.correct}</span>
          <strong>No open issues</strong>
          <p>Every detected problem has been corrected, approved or ignored.</p>
        </div>`;
      return;
    }
    const max = Math.max(...categories.map((entry) => entry.count));
    $("#cat-bars").innerHTML = categories.map((entry) => `
      <div class="cat-row">
        <span class="cat-label">${esc(entry.label)}</span>
        <span class="cat-track"><span class="cat-fill" style="--w:${Math.max(6, (entry.count / max) * 100)}%"></span></span>
        <span class="cat-count">${fmt(entry.count)}</span>
      </div>`).join("");
  }

  function renderProfile() {
    const profile = S.data.profile;
    const rows = profile.map((entry) => `
      <tr>
        <td class="col-name">${esc(entry.name)}</td>
        <td><span class="kind-chip">${esc(entry.label)}</span></td>
        <td>
          <span class="fill-cell"><span class="fill-track"><i style="--w:${entry.filled}%"></i></span>${entry.filled}%</span>
        </td>
        <td class="num">${fmt(entry.unique)}</td>
        <td class="dominant" title="${esc(entry.dominant)}">${esc(entry.dominant) || '<i class="empty">—</i>'}</td>
        <td class="num">${entry.issues ? `<span class="mini-count">${fmt(entry.issues)}</span>` : '<span class="muted">0</span>'}</td>
      </tr>`).join("");
    $("#profile-table").innerHTML = `
      <thead><tr><th>Column</th><th title="Detected type">Type</th><th>Filled</th><th>Unique</th><th>Most common</th><th>Issues</th></tr></thead>
      <tbody>${rows}</tbody>`;
  }

  /* --------------------------------------------------------------------- */
  /* Review table                                                          */
  /* --------------------------------------------------------------------- */

  function renderTable() {
    const rows = S.rows;
    if (!rows) return;
    const columns = S.data.columns.map((column) => column.name);
    const head = `<thead><tr><th class="idx">Row</th>${
      columns.map((column) => `<th title="${esc(column)}">${esc(column)}</th>`).join("")
    }</tr></thead>`;

    if (!rows.items.length) {
      $("#data-table").innerHTML = head + `<tbody><tr><td class="table-empty" colspan="${columns.length + 1}">
        <span>${ICONS.review}</span><strong>No rows match this view</strong>
        <p>Try another filter or clear the search.</p></td></tr></tbody>`;
    } else {
      const body = rows.items.map((row) => {
        const index = `<td class="idx${row.open_issues ? " has-issues" : ""}" title="${row.open_issues ? `${row.open_issues} open issue(s)` : "No open issues"}">${row.excel_row}${
          row.open_issues ? '<span class="row-dot"></span>' : ""
        }</td>`;
        const cells = columns.map((column) => cellHtml(row, column)).join("");
        return `<tr>${index}${cells}</tr>`;
      }).join("");
      $("#data-table").innerHTML = head + `<tbody>${body}</tbody>`;
    }

    const start = rows.total ? (rows.page - 1) * rows.size + 1 : 0;
    const end = Math.min(rows.page * rows.size, rows.total);
    $("#table-note").textContent = rows.total
      ? `Showing ${fmt(start)}–${fmt(end)} of ${fmt(rows.total)} rows`
      : "No rows to show";
    $("#page-label").textContent = `Page ${rows.page} of ${rows.pages}`;
    $("#prev-page").disabled = rows.page <= 1;
    $("#next-page").disabled = rows.page >= rows.pages;
  }

  function cellHtml(row, column) {
    const cell = row.cells[column];
    const value = cell.value ?? "";
    const changed = Boolean(cell.status);
    const pending = !changed && cell.suggestion !== null && cell.suggestion !== undefined;
    const flagged = !changed && !pending && Boolean(cell.error) && value !== "";
    let inner = "";

    if (changed && cell.old !== null && String(cell.old) !== String(value)) {
      inner = `<span class="cell-old">${cell.old === "" ? '<i class="empty">empty</i>' : esc(cell.old)}</span>
               <span class="cell-new">${value === "" ? '<i class="empty">empty</i>' : esc(value)}</span>`;
    } else if (changed) {
      inner = `<span class="cell-new">${value === "" ? '<i class="empty">empty</i>' : esc(value)}</span>
               <span class="conv">${cell.status === "manual" ? "edited" : "converted"}</span>`;
    } else if (pending && String(cell.suggestion) !== String(value)) {
      inner = `<span class="cell-old">${value === "" ? '<i class="empty">empty</i>' : esc(value)}</span>
               <span class="cell-sugg">${esc(cell.suggestion)}</span>`;
    } else if (value === "") {
      inner = '<span class="empty-cell">empty</span>';
    } else if (flagged) {
      inner = `<span class="cell-flag">${esc(value)}</span>`;
    } else {
      inner = `<span class="cell-value">${esc(value)}</span>`;
    }

    const classes = ["editable"];
    if (changed) classes.push(cell.status === "manual" ? "is-manual" : "is-fixed");
    else if (pending) classes.push("is-pending");
    else if (flagged) classes.push("is-flagged");
    if (value === "") classes.push("is-empty");

    let tip = "Double-click to edit";
    if (pending) {
      tip = `${cell.error || "Suggestion"}: ${cell.suggestion}${
        cell.confidence ? ` (${cell.confidence}% confidence)` : ""
      } — double-click to edit`;
    } else if (changed) {
      tip = `${cell.status === "manual" ? "Manual edit" : "Corrected"} — double-click to edit`;
    } else if (flagged) {
      tip = `${cell.error}: needs attention — double-click to edit`;
    }

    return `<td class="${classes.join(" ")}" data-row="${row.row_id}" data-col="${esc(column)}" title="${esc(tip)}">${inner}</td>`;
  }

  function startEdit(cellNode) {
    if (cellNode.querySelector("input")) return;
    const rowId = Number(cellNode.dataset.row);
    const column = cellNode.dataset.col;
    const row = S.rows.items.find((item) => item.row_id === rowId);
    if (!row) return;
    const current = row.cells[column]?.value ?? "";

    const input = document.createElement("input");
    input.className = "cell-input";
    input.value = current;
    input.setAttribute("aria-label", `Edit ${column}`);
    cellNode.innerHTML = "";
    cellNode.appendChild(input);
    input.focus();
    input.select();

    let finished = false;
    const finish = async (save) => {
      if (finished) return;
      finished = true;
      const nextValue = input.value;
      if (!save || nextValue === current) {
        renderTable();
        return;
      }
      try {
        const data = await post("/api/edit", { row_id: rowId, column, value: nextValue });
        applyData(data);
        toast(`Updated ${column} in row ${row.excel_row}.`, "ok");
      } catch (error) {
        toast(error.message, "bad");
        renderTable();
      }
    };

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        finish(true);
      } else if (event.key === "Escape") {
        event.preventDefault();
        finish(false);
      }
    });
    input.addEventListener("blur", () => finish(true));
  }

  async function goPage(page) {
    if (!S.rows) return;
    if (page < 1 || page > S.rows.pages) return;
    S.page = page;
    try {
      const params = new URLSearchParams({ page: String(page), mode: S.mode, search: S.search });
      const response = await fetch(`/api/rows?${params}`);
      const data = await response.json();
      if (!response.ok || !data.rows) throw new Error(data.error || "Could not load rows.");
      S.rows = data.rows;
      S.page = data.rows.page;
      renderTable();
    } catch (error) {
      toast(error.message, "bad");
    }
  }

  /* --------------------------------------------------------------------- */
  /* Corrections                                                           */
  /* --------------------------------------------------------------------- */

  const STATUS_GROUPS = {
    needs: ["pending", "missing", "duplicate"],
    auto: ["auto"],
    approved: ["accepted", "manual"],
    ignored: ["rejected", "removed"],
    all: null,
  };

  const STATUS_CHIP = {
    auto: "ok", accepted: "ok", manual: "manual", pending: "warn",
    missing: "bad", duplicate: "warn", rejected: "muted", removed: "muted",
    reverted: "muted",
  };

  function renderIssues() {
    const issues = S.data.issues;
    const counts = {
      needs: 0, auto: 0, approved: 0, ignored: 0, all: issues.length,
    };
    issues.forEach((issue) => {
      if (STATUS_GROUPS.needs.includes(issue.status)) counts.needs += 1;
      else if (issue.status === "auto") counts.auto += 1;
      else if (issue.status === "accepted" || issue.status === "manual") counts.approved += 1;
      else if (issue.status === "rejected" || issue.status === "removed") counts.ignored += 1;
    });

    const tabs = [
      ["needs", "Needs review"], ["auto", "Auto-corrected"], ["approved", "Approved"],
      ["ignored", "Ignored"], ["all", "All"],
    ];
    $("#issue-tabs").innerHTML = tabs.map(([key, label]) => `
      <button class="filter-chip${S.issueFilter === key ? " active" : ""}" data-filter="${key}" type="button">
        ${label}<span>${fmt(counts[key])}</span>
      </button>`).join("");

    const categories = new Map();
    issues.forEach((issue) => categories.set(issue.category, issue.category_label || issue.category));
    const options = [`<option value="all">All categories</option>`]
      .concat(Array.from(categories.entries())
        .sort((a, b) => a[1].localeCompare(b[1]))
        .map(([key, label]) => `<option value="${esc(key)}">${esc(label)}</option>`));
    const select = $("#issue-category");
    select.innerHTML = options.join("");
    select.value = S.issueCat;

    let filtered = issues.slice();
    const group = STATUS_GROUPS[S.issueFilter];
    if (group) filtered = filtered.filter((issue) => group.includes(issue.status));
    if (S.issueCat !== "all") filtered = filtered.filter((issue) => issue.category === S.issueCat);
    if (S.issueSearch) {
      filtered = filtered.filter((issue) => {
        const haystack = [issue.original, issue.suggestion, issue.column, issue.error, issue.reason]
          .join(" ").toLowerCase();
        return haystack.includes(S.issueSearch);
      });
    }

    const pendingSuggestions = issues.filter(
      (issue) => issue.status === "pending" && issue.suggestion && issue.action === "fix"
    ).length;
    $("#accept-all").disabled = pendingSuggestions === 0;
    $("#accept-all").textContent = pendingSuggestions
      ? `Approve all ${fmt(pendingSuggestions)} suggestions`
      : "Nothing to approve";
    $("#issue-count").textContent =
      `${fmt(filtered.length)} correction${filtered.length === 1 ? "" : "s"}`;
    $("#issue-note").textContent =
      `${fmt(counts.needs)} open · ${fmt(counts.auto)} auto-applied · ${fmt(counts.ignored)} ignored`;

    const visible = filtered.slice(0, S.issueLimit);
    if (!visible.length) {
      $("#issue-list").innerHTML = `
        <div class="card empty-panel">
          <span class="empty-icon">${ICONS.spark}</span>
          <strong>Nothing in this view</strong>
          <p>Switch the filter or clear the search to see more corrections.</p>
        </div>`;
      $("#issue-more").hidden = true;
      return;
    }

    $("#issue-list").innerHTML = visible.map(issueCard).join("");
    $("#issue-more").hidden = filtered.length <= S.issueLimit;
    $("#issue-more").textContent = `Show ${Math.min(40, filtered.length - S.issueLimit)} more`;
  }

  function issueCard(issue) {
    const statusClass = STATUS_CHIP[issue.status] || "muted";
    const confidence = issue.confidence !== null && issue.confidence !== undefined
      ? `<span class="confidence"><span class="conf-track"><i style="--w:${issue.confidence}%"></i></span>${issue.confidence}%</span>`
      : "";
    const rowLabel = issue.row ? `Excel row ${issue.row}` : "All rows";
    const columnLabel = issue.column || "Whole row";
    const original = issue.original
      ? `<span class="issue-original">${esc(issue.original)}</span>`
      : `<span class="issue-original muted">${issue.action === "remove_row" ? "Duplicate record" : "(blank)"}</span>`;
    let target = "";
    if (issue.suggestion) {
      target = `<span class="issue-arrow">→</span><span class="issue-suggested">${esc(issue.suggestion)}</span>`;
    } else if (issue.action === "remove_row") {
      target = `<span class="issue-arrow">→</span><span class="issue-suggested">remove this row</span>`;
    } else if (issue.status === "missing") {
      target = `<span class="issue-arrow">→</span><span class="issue-suggested muted">manual value needed</span>`;
    } else {
      target = `<span class="issue-arrow">→</span><span class="issue-suggested muted">review manually</span>`;
    }

    const open = STATUS_GROUPS.needs.includes(issue.status);
    const actions = [];
    if (open && issue.suggestion && issue.action === "fix") {
      actions.push(`<button class="btn primary xs" data-action="accept" data-id="${issue.id}" type="button">Accept</button>`);
    }
    if (open && issue.action === "remove_row") {
      actions.push(`<button class="btn primary xs" data-action="remove_row" data-id="${issue.id}" data-row="${issue.row_id}" type="button">Remove row</button>`);
    }
    if (open) {
      actions.push(`<button class="btn ghost xs" data-action="reject" data-id="${issue.id}" type="button">Ignore</button>`);
    }
    if (["auto", "accepted", "manual"].includes(issue.status)) {
      actions.push(`<button class="btn ghost xs" data-action="revert" data-id="${issue.id}" type="button">Undo</button>`);
    }
    if (issue.status === "rejected") {
      actions.push(`<button class="btn ghost xs" data-action="revert" data-id="${issue.id}" type="button">Restore</button>`);
    }

    return `
      <article class="issue">
        <div class="issue-head">
          <span class="issue-row">${rowLabel}</span>
          <span class="issue-column">${esc(columnLabel)}</span>
          <span class="status-chip ${statusClass}">${esc(issue.status_label)}</span>
          ${confidence}
        </div>
        <div class="issue-main">${original}${target}</div>
        <p class="issue-why">${esc(issue.reason)}</p>
        <div class="issue-foot">
          <span class="cat-chip">${esc(issue.category_label)}</span>
          <div class="issue-actions">${actions.join("")}</div>
        </div>
      </article>`;
  }

  async function runIssueAction(action, id, rowId) {
    try {
      let data;
      if (action === "accept") data = await post("/api/accept", { ids: [id] });
      else if (action === "reject") data = await post("/api/reject", { ids: [id] });
      else if (action === "revert") data = await post("/api/revert", { id });
      else if (action === "remove_row") data = await post("/api/remove_row", { row_id: Number(rowId) });
      if (!data) return;
      applyData(data);
      const labels = {
        accept: "Correction approved.",
        reject: "Suggestion ignored.",
        revert: "Change undone.",
        remove_row: "Duplicate row removed.",
      };
      toast(labels[action] || "Updated.", "ok");
    } catch (error) {
      toast(error.message, "bad");
    }
  }

  async function acceptAll() {
    try {
      const data = await post("/api/accept_all", {});
      applyData(data);
      toast("All remaining suggestions were approved.", "ok");
    } catch (error) {
      toast(error.message, "bad");
    }
  }

  async function undo() {
    try {
      const data = await post("/api/undo", {});
      applyData(data);
      toast("Last action undone.", "info");
    } catch (error) {
      toast(error.message, "bad");
    }
  }

  async function resetAll() {
    if (!window.confirm("Discard every correction and return to the automatically cleaned version?")) return;
    try {
      const data = await post("/api/reset", {});
      applyData(data);
      toast("All changes were reset.", "info");
    } catch (error) {
      toast(error.message, "bad");
    }
  }

  /* --------------------------------------------------------------------- */
  /* Downloads                                                             */
  /* --------------------------------------------------------------------- */

  function renderDownloads() {
    const { meta, stats } = S.data;
    $("#original-name").textContent = `${meta.file_name} · ${prettySize(meta.size)}`;
    $("#original-size").textContent = meta.sheets.length > 1
      ? `Sheets: ${meta.sheets.join(", ")}`
      : `Sheet: ${meta.sheet}`;
    $("#session-sheet").textContent = meta.sheet;

    const rows = [
      ["File", meta.file_name],
      ["Sheet analyzed", meta.sheet],
      ["Rows (active)", fmt(stats.rows)],
      ["Columns", fmt(stats.columns)],
      ["Issues detected", fmt(stats.errors_detected)],
      ["Auto-corrected", fmt(stats.auto_corrected)],
      ["Approved", fmt(stats.approved)],
      ["Manual edits", fmt(stats.manual)],
      ["Remaining issues", fmt(stats.needs_review)],
      ["Data health", `${stats.health}%`],
    ];
    $("#session-summary").innerHTML = rows.map(([label, value]) => `
      <div class="summary-item"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("");
  }

  /* --------------------------------------------------------------------- */
  /* Fetch helpers / toast                                                 */
  /* --------------------------------------------------------------------- */

  async function post(path, payload) {
    const body = Object.assign({ page: S.page, mode: S.mode, search: S.search }, payload);
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "Request failed.");
    return data;
  }

  function toast(message, kind = "ok") {
    const box = $("#toasts");
    const node = document.createElement("div");
    node.className = `toast ${kind}`;
    node.innerHTML = `<span class="toast-dot"></span><span>${esc(message)}</span>`;
    box.appendChild(node);
    setTimeout(() => {
      node.classList.add("leaving");
      setTimeout(() => node.remove(), 320);
    }, 3600);
  }
})();

==============================================================================
FILE: static/styles.css
==============================================================================
/* --------------------------------------------------------------------------
   DataFix — theme
   Dark violet workspace, glassy cards, one accent gradient.
   -------------------------------------------------------------------------- */

:root {
  color-scheme: dark;
  --primary-color: #8b7bff;
  --secondary-color: #c084fc;
  --accent-color: #5eead4;
  --background-color: #0a0713;
  --card-background: #151027;
  --text-color: #edeafb;
  --border-color: rgba(150, 135, 255, 0.14);
  --button-color: #6857f5;
  --hover-color: #7c3aed;

  --bg: var(--background-color);
  --bg-soft: #0e0a1b;
  --panel: rgba(21, 16, 35, 0.82);
  --panel-solid: var(--card-background);
  --panel-2: #1a1430;
  --surface-1: #120e21;
  --surface-2: #171129;
  --surface-3: #1c1533;
  --input-bg: #171129;
  --line: var(--border-color);
  --line-strong: rgba(150, 135, 255, 0.3);
  --line-soft: color-mix(in srgb, var(--text-color) 3.5%, transparent);
  --text: var(--text-color);
  --text-dim: #c9c3e2;
  --muted: #a8a2c6;
  --dim: #8a84a6;
  --faint: #7c7698;
  --accent: var(--primary-color);
  --accent-2: var(--secondary-color);
  --accent-text: #c4b5fd;
  --cyan: var(--accent-color);
  --cyan-text: #5eead4;
  --green: #34d399;
  --amber: #fbbf24;
  --rose: #fb7185;
  --ok-text: #6ee7b7;
  --warn-text: #fcd34d;
  --bad-text: #fda4af;
  --on-accent: #fff;
  --button-bg: var(--button-color);
  --button-hover: var(--hover-color);
  --button-grad: linear-gradient(135deg, var(--button-color), var(--hover-color));
  --icon-grad: linear-gradient(135deg, var(--accent), var(--accent-2));
  --text-grad: linear-gradient(100deg, #a78bfa, #e879f9 55%, #5eead4);
  --bar-grad: linear-gradient(90deg, var(--button-color), var(--secondary-color));
  --bar-grad-2: linear-gradient(90deg, var(--accent), var(--cyan));
  --glow: rgba(139, 123, 255, 0.45);
  --glow-strong: rgba(139, 123, 255, 0.9);
  --gloss: rgba(255, 255, 255, 0.18);
  --gloss-strong: rgba(255, 255, 255, 0.22);
  --shadow-ink: rgba(0, 0, 0, 0.8);
  --shadow-ink-strong: rgba(0, 0, 0, 0.9);
  --shadow: 0 24px 60px -30px rgba(0, 0, 0, 0.8);
  --shadow-soft: 0 16px 40px -24px rgba(0, 0, 0, 0.65);
  --shadow-strong: 0 40px 90px -30px rgba(0, 0, 0, 0.9);
  --backdrop-grad:
    radial-gradient(1200px 700px at 85% -10%, rgba(109, 91, 255, 0.16), transparent 60%),
    radial-gradient(900px 600px at 0% 100%, rgba(56, 189, 248, 0.08), transparent 55%),
    var(--bg);
  --sidebar-grad: linear-gradient(180deg, rgba(14, 10, 26, 0.9), rgba(10, 7, 19, 0.92));
  --topbar-bg: rgba(10, 7, 19, 0.94);
  --hero-grad:
    radial-gradient(700px 380px at 88% 12%, rgba(139, 123, 255, 0.2), transparent 65%),
    radial-gradient(500px 300px at 10% 100%, rgba(94, 234, 212, 0.06), transparent 60%),
    linear-gradient(155deg, rgba(27, 21, 47, 0.94), rgba(16, 12, 29, 0.94));
  --welcome-grad:
    radial-gradient(600px 300px at 92% 0%, rgba(139, 123, 255, 0.18), transparent 65%),
    linear-gradient(150deg, rgba(26, 20, 46, 0.94), rgba(16, 12, 29, 0.94));
  --next-grad:
    radial-gradient(500px 220px at 100% 0%, rgba(94, 234, 212, 0.07), transparent 60%),
    linear-gradient(150deg, rgba(26, 20, 46, 0.92), rgba(16, 12, 29, 0.92));
  --stat-grad: linear-gradient(160deg, rgba(26, 20, 46, 0.9), rgba(17, 13, 30, 0.9));
  --issue-grad: linear-gradient(160deg, rgba(25, 19, 44, 0.88), rgba(17, 13, 30, 0.88));
  --download-accent-grad:
    radial-gradient(420px 240px at 20% 0%, rgba(139, 123, 255, 0.2), transparent 65%),
    linear-gradient(160deg, rgba(30, 23, 54, 0.95), rgba(17, 13, 30, 0.95));
  --mock-bg: rgba(12, 9, 22, 0.85);
  --toast-bg: #19132c;
  --overlay-bg: rgba(6, 4, 12, 0.72);
  --overlay-card: #161026;
  --track: color-mix(in srgb, var(--text-color) 6%, transparent);
  --ghost: color-mix(in srgb, var(--text-color) 3%, transparent);
  --hover-bg: color-mix(in srgb, var(--text-color) 4%, transparent);
  --ring-track: color-mix(in srgb, var(--text-color) 7%, transparent);
  --cell-fixed-bg: rgba(52, 211, 153, 0.07);
  --cell-manual-bg: rgba(251, 191, 36, 0.08);
  --cell-pending-bg: rgba(192, 132, 252, 0.08);
  --cell-bad-bg: rgba(251, 113, 133, 0.08);
  --radius: 18px;
  --radius-sm: 12px;
  --font: "Inter", "Segoe UI", system-ui, sans-serif;
  --display: "Manrope", "Inter", system-ui, sans-serif;
}

[data-mode="light"] {
  color-scheme: light;
  --bg-soft: color-mix(in srgb, var(--background-color) 84%, var(--card-background));
  --panel: var(--card-background);
  --panel-2: color-mix(in srgb, var(--card-background) 90%, var(--primary-color) 10%);
  --surface-1: color-mix(in srgb, var(--card-background) 94%, var(--primary-color) 6%);
  --surface-2: color-mix(in srgb, var(--primary-color) 5%, var(--card-background));
  --surface-3: var(--card-background);
  --input-bg: var(--card-background);
  --line-strong: color-mix(in srgb, var(--border-color) 55%, var(--primary-color) 45%);
  --line-soft: color-mix(in srgb, var(--text-color) 9%, transparent);
  --text-dim: color-mix(in srgb, var(--text-color) 78%, var(--background-color));
  --muted: color-mix(in srgb, var(--text-color) 64%, var(--background-color));
  --dim: color-mix(in srgb, var(--text-color) 52%, var(--background-color));
  --faint: color-mix(in srgb, var(--text-color) 46%, var(--background-color));
  --accent-text: color-mix(in srgb, var(--secondary-color) 42%, var(--text-color));
  --cyan-text: color-mix(in srgb, var(--accent-color) 55%, var(--text-color));
  --green: #059669;
  --amber: #d97706;
  --rose: #dc2626;
  --ok-text: #047857;
  --warn-text: #b45309;
  --bad-text: #b91c1c;
  --text-grad: linear-gradient(100deg, var(--primary-color), color-mix(in srgb, var(--primary-color) 55%, var(--secondary-color)), color-mix(in srgb, var(--primary-color) 68%, var(--accent-color)));
  --glow: color-mix(in srgb, var(--primary-color) 40%, transparent);
  --glow-strong: color-mix(in srgb, var(--primary-color) 72%, transparent);
  --shadow-ink: color-mix(in srgb, var(--text-color) 14%, transparent);
  --shadow-ink-strong: color-mix(in srgb, var(--text-color) 18%, transparent);
  --shadow: 0 24px 50px -32px color-mix(in srgb, var(--text-color) 35%, transparent);
  --shadow-soft: 0 16px 34px -26px color-mix(in srgb, var(--text-color) 30%, transparent);
  --shadow-strong: 0 38px 80px -36px color-mix(in srgb, var(--text-color) 35%, transparent);
  --backdrop-grad:
    radial-gradient(1100px 700px at 88% -12%, color-mix(in srgb, var(--primary-color) 10%, transparent), transparent 60%),
    radial-gradient(900px 600px at 0% 105%, color-mix(in srgb, var(--accent-color) 12%, transparent), transparent 55%),
    var(--bg);
  --sidebar-grad: linear-gradient(180deg, color-mix(in srgb, var(--card-background) 92%, var(--primary-color) 8%), color-mix(in srgb, var(--background-color) 94%, var(--primary-color) 6%));
  --topbar-bg: color-mix(in srgb, var(--background-color) 86%, transparent);
  --hero-grad:
    radial-gradient(700px 380px at 88% 12%, color-mix(in srgb, var(--primary-color) 12%, transparent), transparent 65%),
    radial-gradient(500px 300px at 10% 100%, color-mix(in srgb, var(--accent-color) 10%, transparent), transparent 60%),
    linear-gradient(155deg, var(--card-background), color-mix(in srgb, var(--background-color) 78%, var(--card-background) 22%));
  --welcome-grad:
    radial-gradient(600px 300px at 92% 0%, color-mix(in srgb, var(--primary-color) 10%, transparent), transparent 65%),
    linear-gradient(150deg, var(--card-background), color-mix(in srgb, var(--background-color) 72%, var(--card-background) 28%));
  --next-grad:
    radial-gradient(500px 220px at 100% 0%, color-mix(in srgb, var(--accent-color) 10%, transparent), transparent 60%),
    linear-gradient(150deg, var(--card-background), color-mix(in srgb, var(--background-color) 82%, var(--card-background) 18%));
  --stat-grad: linear-gradient(160deg, var(--card-background), color-mix(in srgb, var(--background-color) 80%, var(--card-background) 20%));
  --issue-grad: linear-gradient(150deg, var(--card-background), color-mix(in srgb, var(--background-color) 82%, var(--card-background) 18%));
  --download-accent-grad:
    radial-gradient(420px 240px at 20% 0%, color-mix(in srgb, var(--primary-color) 12%, transparent), transparent 65%),
    linear-gradient(155deg, var(--card-background), color-mix(in srgb, var(--primary-color) 7%, var(--card-background)));
  --mock-bg: color-mix(in srgb, var(--primary-color) 4%, var(--card-background));
  --toast-bg: var(--card-background);
  --overlay-bg: rgba(17, 24, 39, 0.34);
  --overlay-card: var(--card-background);
  --track: color-mix(in srgb, var(--text-color) 8%, transparent);
  --ghost: color-mix(in srgb, var(--text-color) 4%, transparent);
  --hover-bg: color-mix(in srgb, var(--text-color) 6%, transparent);
  --ring-track: color-mix(in srgb, var(--text-color) 9%, transparent);
  --cell-fixed-bg: color-mix(in srgb, var(--green) 10%, var(--card-background));
  --cell-manual-bg: color-mix(in srgb, var(--amber) 12%, var(--card-background));
  --cell-pending-bg: color-mix(in srgb, var(--primary-color) 8%, var(--card-background));
  --cell-bad-bg: color-mix(in srgb, var(--rose) 9%, var(--card-background));
}

* { box-sizing: border-box; }

[hidden] { display: none !important; }

html { scroll-behavior: smooth; }

body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 14.5px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}

h1, h2, h3 {
  font-family: var(--display);
  line-height: 1.18;
  margin: 0;
  letter-spacing: -0.015em;
}

p { margin: 0; }
a { color: inherit; }
button, input, select { font: inherit; color: inherit; }
svg { display: block; }

::selection { background: color-mix(in srgb, var(--accent) 40%, transparent); }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: color-mix(in srgb, var(--accent) 22%, transparent);
  border-radius: 99px;
  border: 2px solid var(--bg);
}
::-webkit-scrollbar-thumb:hover { background: color-mix(in srgb, var(--accent) 40%, transparent); }

/* Backdrop ---------------------------------------------------------------- */

.backdrop {
  position: fixed;
  inset: 0;
  z-index: -1;
  overflow: hidden;
  background: var(--backdrop-grad);
}

.orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(90px);
  opacity: 0.55;
  animation: drift 22s ease-in-out infinite alternate;
}

.orb-a {
  width: 520px; height: 520px;
  top: -180px; left: 42%;
  background: color-mix(in srgb, var(--accent) 28%, transparent);
}

.orb-b {
  width: 420px; height: 420px;
  bottom: -160px; right: -80px;
  background: color-mix(in srgb, var(--cyan) 12%, transparent);
  animation-delay: -8s;
}

.grid-lines {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(color-mix(in srgb, var(--text-color) 2.5%, transparent) 1px, transparent 1px),
    linear-gradient(90deg, color-mix(in srgb, var(--text-color) 2.5%, transparent) 1px, transparent 1px);
  background-size: 56px 56px;
  mask-image: radial-gradient(ellipse at 50% 0%, #000 30%, transparent 75%);
}

@keyframes drift {
  from { transform: translate3d(0, 0, 0) scale(1); }
  to { transform: translate3d(-40px, 40px, 0) scale(1.08); }
}

/* Shell ------------------------------------------------------------------- */

.app {
  display: grid;
  grid-template-columns: 258px minmax(0, 1fr);
  min-height: 100vh;
}

.sidebar {
  position: sticky;
  top: 0;
  height: 100vh;
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 24px 18px;
  background: var(--sidebar-grad);
  border-right: 1px solid var(--line);
  backdrop-filter: blur(14px);
}

.brand {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 4px 6px;
  text-decoration: none;
}

.brand-mark {
  width: 38px;
  height: 38px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  color: var(--on-accent);
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  box-shadow: 0 12px 30px -12px var(--glow-strong);
}

.brand-mark svg { width: 20px; height: 20px; }

.brand-text { display: flex; flex-direction: column; line-height: 1.15; }
.brand-text strong { font-family: var(--display); font-size: 17px; letter-spacing: -0.01em; }
.brand-text small { color: var(--muted); font-size: 11px; }

.nav { display: flex; flex-direction: column; gap: 4px; }

.nav-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 11px;
  width: 100%;
  padding: 10px 12px;
  border: 1px solid transparent;
  border-radius: 11px;
  background: transparent;
  color: var(--muted);
  font-size: 13.5px;
  font-weight: 500;
  text-align: left;
  cursor: pointer;
  transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
}

.nav-item svg { width: 18px; height: 18px; flex: none; opacity: 0.85; }

.nav-item:hover:not(:disabled) { background: color-mix(in srgb, var(--text-color) 4%, transparent); color: var(--text); }

.nav-item.active {
  color: var(--text);
  background: linear-gradient(90deg, color-mix(in srgb, var(--accent) 18%, transparent), color-mix(in srgb, var(--accent) 4%, transparent));
  border-color: color-mix(in srgb, var(--accent) 22%, transparent);
}

.nav-item.active::before {
  content: "";
  position: absolute;
  left: -18px;
  top: 9px;
  bottom: 9px;
  width: 3px;
  border-radius: 0 3px 3px 0;
  background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.nav-item:disabled { opacity: 0.42; cursor: not-allowed; }

.side-card {
  padding: 13px 14px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: linear-gradient(160deg, color-mix(in srgb, var(--accent) 8%, transparent), color-mix(in srgb, var(--text-color) 1.5%, transparent));
}

.side-label {
  display: block;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 5px;
}

.side-file {
  font-weight: 600;
  font-size: 13px;
  overflow-wrap: anywhere;
}

.side-meta { color: var(--muted); font-size: 11.5px; margin-top: 2px; }

.side-tip {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 4px;
  color: var(--muted);
  font-size: 11.5px;
}

.side-tip-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--green) 14%, transparent);
  flex: none;
}

.sidebar .btn.block { margin-top: auto; }

.side-foot {
  text-align: center;
  color: var(--faint);
  font-size: 10.5px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* Main -------------------------------------------------------------------- */

.main {
  min-width: 0;
  padding: 22px 30px 72px;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 8px 0 16px;
  margin-bottom: 6px;
  background: linear-gradient(180deg, var(--topbar-bg) 55%, transparent);
  backdrop-filter: blur(10px);
}

.topbar-copy h1 { font-size: 22px; }
.topbar-copy p { color: var(--muted); font-size: 12.5px; margin-top: 2px; }

.topbar-actions { display: flex; align-items: center; gap: 8px; }

/* Buttons ----------------------------------------------------------------- */

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 18px;
  border-radius: 11px;
  border: 1px solid transparent;
  font-size: 13px;
  font-weight: 600;
  text-decoration: none;
  cursor: pointer;
  white-space: nowrap;
  transition: transform 0.15s ease, box-shadow 0.2s ease, background 0.2s ease,
              border-color 0.2s ease, color 0.2s ease, opacity 0.2s ease;
}

.btn:active { transform: translateY(1px); }

.btn.primary {
  color: var(--on-accent);
  background: var(--button-grad);
  box-shadow: 0 14px 34px -16px var(--glow-strong), inset 0 1px 0 var(--gloss);
}

.btn.primary:hover:not(:disabled) {
  box-shadow: 0 18px 40px -14px var(--glow-strong), inset 0 1px 0 var(--gloss-strong);
  transform: translateY(-1px);
}

.btn.ghost {
  color: var(--text-dim);
  background: color-mix(in srgb, var(--text-color) 3%, transparent);
  border-color: var(--line);
}

.btn.ghost:hover:not(:disabled) {
  color: var(--text);
  border-color: var(--line-strong);
  background: color-mix(in srgb, var(--accent) 10%, transparent);
}

.btn.danger {
  color: var(--bad-text);
  background: color-mix(in srgb, var(--rose) 10%, transparent);
  border-color: color-mix(in srgb, var(--rose) 28%, transparent);
}

.btn.danger:hover:not(:disabled) { background: color-mix(in srgb, var(--rose) 18%, transparent); }

.btn.sm { padding: 7px 13px; font-size: 12px; border-radius: 9px; }
.btn.xs { padding: 5px 11px; font-size: 11.5px; border-radius: 8px; }
.btn.block { width: 100%; }

.btn:disabled { opacity: 0.44; cursor: not-allowed; box-shadow: none; transform: none; }

.icon-btn {
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border-radius: 9px;
  border: 1px solid var(--line);
  background: color-mix(in srgb, var(--text-color) 3%, transparent);
  color: var(--text-dim);
  font-size: 17px;
  line-height: 1;
  cursor: pointer;
  transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
}

.icon-btn:hover:not(:disabled) { color: var(--text); border-color: var(--line-strong); background: color-mix(in srgb, var(--accent) 12%, transparent); }
.icon-btn:disabled { opacity: 0.35; cursor: not-allowed; }

.link-btn {
  padding: 0;
  border: 0;
  background: none;
  color: var(--accent-text);
  font-weight: 600;
  cursor: pointer;
  text-decoration: underline;
  text-decoration-color: color-mix(in srgb, var(--accent-2) 40%, transparent);
  text-underline-offset: 3px;
}

.link-btn:hover { color: var(--accent-text); }

:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

/* Cards / shared ---------------------------------------------------------- */

.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 22px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(8px);
}

.card-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 16px;
}

.card-head h3 { font-size: 15.5px; }
.card-head p { font-size: 12.5px; margin-top: 3px; }

.muted { color: var(--muted); }

.pill {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 99px;
  border: 1px solid var(--line);
  background: color-mix(in srgb, var(--text-color) 3%, transparent);
  color: var(--text-dim);
  font-size: 11px;
  font-weight: 500;
  white-space: nowrap;
}

.eyebrow {
  display: inline-block;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--accent-text);
  margin-bottom: 10px;
}

.hint { color: var(--muted); font-size: 12px; }

/* Upload view ------------------------------------------------------------- */

.view { animation: rise 0.4s ease both; }

@keyframes rise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.hero {
  display: grid;
  grid-template-columns: 1.08fr 0.92fr;
  gap: 26px;
  align-items: center;
  padding: 38px 38px 34px;
  margin-bottom: 20px;
  background: var(--hero-grad);
  overflow: hidden;
}

.hero-copy h2 { font-size: clamp(30px, 3.4vw, 44px); letter-spacing: -0.03em; }

.grad {
  background: var(--text-grad);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.hero-copy > p {
  color: var(--text-dim);
  font-size: 14px;
  max-width: 46ch;
  margin-top: 14px;
}

.hero-chips { display: flex; flex-wrap: wrap; gap: 7px; margin: 20px 0 24px; }

.chip {
  padding: 5px 11px;
  border-radius: 99px;
  border: 1px solid var(--line);
  background: color-mix(in srgb, var(--text-color) 3%, transparent);
  color: var(--text-dim);
  font-size: 11.5px;
}

.hero-art { position: relative; display: flex; justify-content: center; }

.mock-table {
  width: 100%;
  max-width: 380px;
  border: 1px solid var(--line-strong);
  border-radius: 14px;
  background: var(--mock-bg);
  overflow: hidden;
  box-shadow: 0 30px 70px -30px var(--shadow-ink-strong);
  transform: rotate(-1.4deg);
}

.mock-head, .mock-row {
  display: grid;
  grid-template-columns: 1.1fr 1.4fr 0.8fr;
  gap: 8px;
  padding: 9px 14px;
  font-size: 11px;
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
}

.mock-head {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 9.5px;
  border-bottom: 1px solid var(--line);
  background: color-mix(in srgb, var(--text-color) 2.5%, transparent);
}

.mock-row { border-bottom: 1px solid color-mix(in srgb, var(--text-color) 4%, transparent); }
.mock-row:last-child { border-bottom: 0; }
.mock-row .bad { color: var(--bad-text); text-decoration: line-through; text-decoration-color: color-mix(in srgb, var(--rose) 50%, transparent); }
.mock-row .good { color: var(--ok-text); }
.mock-row .warn { color: var(--warn-text); }

.mock-badge {
  position: absolute;
  bottom: -12px;
  right: 4px;
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 7px 13px;
  border-radius: 99px;
  background: var(--panel-solid);
  border: 1px solid var(--line-strong);
  font-size: 11.5px;
  font-weight: 600;
  box-shadow: 0 18px 40px -18px var(--shadow-ink-strong);
}

.pulse {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--accent-2);
  animation: pulse 2s ease-out infinite;
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent-2) 55%, transparent); }
  70% { box-shadow: 0 0 0 9px transparent; }
  100% { box-shadow: 0 0 0 0 transparent; }
}

.upload-grid {
  display: grid;
  grid-template-columns: 1.35fr 0.65fr;
  gap: 20px;
  align-items: start;
}

.upload-card { padding: 24px; }

.dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 30px 20px;
  border: 1.5px dashed var(--line-strong);
  border-radius: 15px;
  background:
    radial-gradient(400px 160px at 50% 0%, color-mix(in srgb, var(--accent) 8%, transparent), transparent 70%),
    color-mix(in srgb, var(--text-color) 1.2%, transparent);
  cursor: pointer;
  text-align: center;
  transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease;
}

.dropzone:hover, .dropzone:focus-visible, .dropzone.dragging {
  border-color: var(--accent);
  background:
    radial-gradient(400px 180px at 50% 0%, color-mix(in srgb, var(--accent) 18%, transparent), transparent 70%),
    color-mix(in srgb, var(--accent) 5%, transparent);
}

.dropzone.dragging { transform: scale(1.008); }

.dz-icon {
  width: 46px;
  height: 46px;
  display: grid;
  place-items: center;
  border-radius: 13px;
  background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 24%, transparent), color-mix(in srgb, var(--cyan) 10%, transparent));
  border: 1px solid var(--line-strong);
  color: var(--text);
  margin-bottom: 8px;
}

.dz-icon svg { width: 22px; height: 22px; }

.dz-title { font-family: var(--display); font-weight: 700; font-size: 15px; }
.dz-sub { color: var(--muted); font-size: 12.5px; }
.dz-note { color: var(--dim); font-size: 11px; margin-top: 8px; }

.file-chip {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 11px 13px;
  margin-top: 14px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--accent) 8%, transparent);
  animation: rise 0.25s ease both;
}

.file-badge {
  flex: none;
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 24%, var(--panel-solid)), color-mix(in srgb, var(--accent) 42%, var(--panel-solid)));
  font-size: 9.5px;
  font-weight: 800;
  letter-spacing: 0.06em;
  color: var(--accent-text);
}

.file-info { display: flex; flex-direction: column; min-width: 0; flex: 1; }
.file-info strong { font-size: 13px; overflow-wrap: anywhere; }
.file-info small { color: var(--muted); font-size: 11.5px; }

.upload-error {
  margin-top: 12px;
  padding: 10px 13px;
  border-radius: 10px;
  border: 1px solid color-mix(in srgb, var(--rose) 35%, transparent);
  background: color-mix(in srgb, var(--rose) 10%, transparent);
  color: var(--bad-text);
  font-size: 12.5px;
}

.upload-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-top: 18px;
}

@keyframes slide {
  0% { transform: translateX(-110%); }
  100% { transform: translateX(310%); }
}

#back-btn { margin-top: 14px; }

.checks-card h3 { font-size: 15.5px; margin-bottom: 14px; }

.check-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.check-list li { display: flex; gap: 11px; }
.check-list strong { display: block; font-size: 12.8px; font-weight: 600; }
.check-list small { color: var(--muted); font-size: 11.8px; line-height: 1.45; display: block; }

.dot {
  flex: none;
  width: 9px;
  height: 9px;
  margin-top: 5px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 12%, transparent);
}

.dot.violet { background: var(--accent); box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 13%, transparent); }
.dot.cyan { background: var(--cyan); box-shadow: 0 0 0 4px color-mix(in srgb, var(--cyan) 11%, transparent); }
.dot.green { background: var(--green); box-shadow: 0 0 0 4px color-mix(in srgb, var(--green) 11%, transparent); }
.dot.amber { background: var(--amber); box-shadow: 0 0 0 4px color-mix(in srgb, var(--amber) 11%, transparent); }
.dot.rose { background: var(--rose); box-shadow: 0 0 0 4px color-mix(in srgb, var(--rose) 11%, transparent); }

.feature-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
  margin-top: 20px;
}

.feature {
  display: flex;
  gap: 12px;
  padding: 16px 18px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--text-color) 1.5%, transparent);
}

.feature-icon {
  font-family: var(--display);
  font-weight: 800;
  font-size: 12px;
  color: var(--accent-text);
  background: color-mix(in srgb, var(--accent) 12%, transparent);
  border-radius: 8px;
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  flex: none;
}

.feature strong { display: block; font-size: 12.8px; }
.feature p { color: var(--muted); font-size: 11.8px; margin-top: 2px; }

/* Dashboard --------------------------------------------------------------- */

.welcome {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 26px;
  padding: 26px 28px;
  margin-bottom: 18px;
  background: var(--welcome-grad);
}

.welcome-copy h2 { font-size: clamp(22px, 2.4vw, 30px); }
.welcome-copy > p { color: var(--text-dim); font-size: 13px; margin-top: 7px; max-width: 70ch; }
.welcome-actions { display: flex; gap: 9px; margin-top: 16px; flex-wrap: wrap; }

.health-wrap {
  flex: none;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  width: 160px;
}

.health-ring {
  --p: 0;
  --ring: var(--green);
  width: 118px;
  height: 118px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: conic-gradient(var(--ring) calc(var(--p) * 1%), color-mix(in srgb, var(--text-color) 7%, transparent) 0);
  position: relative;
  transition: --p 0.6s ease;
  filter: drop-shadow(0 8px 30px color-mix(in srgb, var(--green) 18%, transparent));
}

.health-ring.mid { --ring: var(--amber); filter: drop-shadow(0 8px 30px color-mix(in srgb, var(--amber) 14%, transparent)); }
.health-ring.low { --ring: var(--rose); filter: drop-shadow(0 8px 30px color-mix(in srgb, var(--rose) 14%, transparent)); }

.health-inner {
  width: 92px;
  height: 92px;
  border-radius: 50%;
  background: var(--panel-solid);
  border: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1px;
}

.health-inner strong { font-family: var(--display); font-size: 24px; letter-spacing: -0.03em; }
.health-inner span { color: var(--muted); font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.1em; }
.health-wrap > small { color: var(--muted); font-size: 10.5px; text-align: center; line-height: 1.35; }

.stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 20px;
}

.stat {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 16px 17px;
  border: 1px solid var(--line);
  border-radius: 15px;
  background: var(--stat-grad);
  box-shadow: var(--shadow);
  transition: border-color 0.2s ease, transform 0.2s ease;
}

.stat:hover { border-color: var(--line-strong); transform: translateY(-2px); }

.stat-icon {
  flex: none;
  width: 34px;
  height: 34px;
  border-radius: 10px;
  display: grid;
  place-items: center;
  background: color-mix(in srgb, var(--accent) 13%, transparent);
  color: var(--accent-text);
}

.stat-icon svg { width: 17px; height: 17px; }

.stat.tone-green .stat-icon { background: color-mix(in srgb, var(--green) 12%, transparent); color: var(--ok-text); }
.stat.tone-rose .stat-icon { background: color-mix(in srgb, var(--rose) 12%, transparent); color: var(--bad-text); }
.stat.tone-amber .stat-icon { background: color-mix(in srgb, var(--amber) 12%, transparent); color: var(--warn-text); }
.stat.tone-cyan .stat-icon { background: color-mix(in srgb, var(--cyan) 10%, transparent); color: var(--cyan-text); }

.stat-body { min-width: 0; display: flex; flex-direction: column; }
.stat-label { color: var(--muted); font-size: 10.8px; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; }
.stat-value { font-family: var(--display); font-size: 24px; letter-spacing: -0.03em; line-height: 1.2; margin-top: 1px; }
.stat-body small { color: var(--dim); font-size: 10.8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.panel { display: none; }
.panel.active { display: block; animation: rise 0.32s ease both; }

.grid-2 {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 18px;
}

.grid-2 .card { display: flex; flex-direction: column; min-height: 340px; }

.cat-bars { display: flex; flex-direction: column; gap: 12px; padding-top: 2px; }

.cat-row {
  display: grid;
  grid-template-columns: 128px minmax(0, 1fr) 40px;
  gap: 12px;
  align-items: center;
}

.cat-label { color: var(--text-dim); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.cat-track {
  height: 8px;
  border-radius: 99px;
  background: color-mix(in srgb, var(--text-color) 5.5%, transparent);
  overflow: hidden;
}

.cat-fill {
  display: block;
  width: var(--w, 0%);
  height: 100%;
  border-radius: inherit;
  background: var(--bar-grad);
  box-shadow: 0 0 14px var(--glow);
  transition: width 0.5s ease;
}

.cat-count { text-align: right; font-size: 12px; font-weight: 600; color: var(--text-dim); font-variant-numeric: tabular-nums; }

.table-scroll {
  overflow: auto;
  border-radius: 10px;
}

.table-scroll.compact { max-height: 300px; flex: 1; border: 1px solid color-mix(in srgb, var(--text-color) 4%, transparent); }

.mini-table {
  width: 100%;
  min-width: 540px;
  border-collapse: collapse;
  font-size: 12px;
  table-layout: fixed;
}

.mini-table th:nth-child(1) { width: 24%; }
.mini-table th:nth-child(2) { width: 16%; }
.mini-table th:nth-child(3) { width: 14%; }
.mini-table th:nth-child(4) { width: 12%; }
.mini-table th:nth-child(5) { width: 22%; }
.mini-table th:nth-child(6) { width: 12%; text-align: right; }

.mini-table th {
  position: sticky;
  top: 0;
  z-index: 1;
  padding: 8px 10px;
  text-align: left;
  font-size: 9.8px;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--muted);
  background: var(--surface-2);
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.mini-table td {
  padding: 8px 10px;
  border-bottom: 1px solid color-mix(in srgb, var(--text-color) 3.5%, transparent);
  color: var(--text-dim);
  white-space: nowrap;
}

.mini-table tr:last-child td { border-bottom: 0; }
.mini-table .col-name { color: var(--text); font-weight: 600; max-width: 150px; overflow: hidden; text-overflow: ellipsis; }
.mini-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.mini-table .dominant { max-width: 150px; overflow: hidden; text-overflow: ellipsis; color: var(--muted); }

.kind-chip {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 99px;
  border: 1px solid color-mix(in srgb, var(--accent) 22%, transparent);
  background: color-mix(in srgb, var(--accent) 8%, transparent);
  color: var(--accent-text);
  font-size: 10px;
  font-weight: 500;
  white-space: nowrap;
}

.fill-cell { display: inline-flex; align-items: center; gap: 7px; }
.fill-track {
  display: inline-block;
  width: 44px;
  height: 5px;
  border-radius: 99px;
  background: color-mix(in srgb, var(--text-color) 7%, transparent);
  overflow: hidden;
}
.fill-track i {
  display: block;
  height: 100%;
  width: var(--w, 0%);
  border-radius: inherit;
  background: linear-gradient(90deg, var(--accent), var(--cyan));
}

.mini-count {
  display: inline-block;
  min-width: 22px;
  padding: 1px 6px;
  border-radius: 6px;
  background: color-mix(in srgb, var(--amber) 13%, transparent);
  color: var(--warn-text);
  font-size: 10.5px;
  font-weight: 600;
  text-align: center;
}

.empty { color: var(--faint); font-style: normal; }

.next-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  margin-top: 18px;
  background: var(--next-grad);
}

.next-copy h3 { font-size: 17px; margin-bottom: 4px; }
.next-copy p { color: var(--muted); font-size: 12.5px; max-width: 62ch; }
.next-actions { display: flex; gap: 9px; flex-wrap: wrap; flex: none; }

/* Review table ------------------------------------------------------------ */

.toolbar-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 13px 16px;
  margin-bottom: 14px;
}

.toolbar-card.wrap { flex-wrap: wrap; }

.legend { display: flex; flex-wrap: wrap; gap: 14px; color: var(--muted); font-size: 11.5px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }

.sw { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.sw.fixed { background: color-mix(in srgb, var(--green) 55%, transparent); }
.sw.manual { background: color-mix(in srgb, var(--amber) 65%, transparent); }
.sw.pending { background: color-mix(in srgb, var(--accent-2) 60%, transparent); }
.sw.flagged { background: color-mix(in srgb, var(--rose) 60%, transparent); }

.review-tools { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }

.search {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 12px;
  height: 36px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: color-mix(in srgb, var(--text-color) 2.5%, transparent);
  color: var(--muted);
  transition: border-color 0.2s ease, background 0.2s ease;
}

.search:focus-within { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 7%, transparent); }
.search svg { width: 15px; height: 15px; flex: none; }

.search input {
  border: 0;
  background: none;
  outline: none;
  width: 190px;
  color: var(--text);
  font-size: 12.5px;
}

.search input::placeholder { color: var(--faint); }

select {
  height: 36px;
  padding: 0 30px 0 12px;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: var(--input-bg)
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23948eb0' stroke-width='2.4' stroke-linecap='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E")
    no-repeat right 11px center;
  color: var(--text-dim);
  font-size: 12.5px;
  appearance: none;
  cursor: pointer;
  outline: none;
}

select:focus-visible { border-color: var(--accent); }

.table-card { padding: 0; overflow: hidden; }

.data-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 12.5px;
}

.data-table th {
  position: sticky;
  top: 0;
  z-index: 2;
  padding: 11px 12px;
  text-align: left;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  background: var(--surface-2);
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
  max-width: 230px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.data-table th.idx {
  left: 0;
  z-index: 3;
  width: 62px;
  text-align: right;
}

.data-table td {
  padding: 8px 12px;
  border-bottom: 1px solid color-mix(in srgb, var(--text-color) 3.5%, transparent);
  vertical-align: top;
  max-width: 260px;
  color: var(--text-dim);
}

.data-table tbody tr:hover td { background: color-mix(in srgb, var(--accent) 4.5%, transparent); }

.data-table td.idx {
  position: sticky;
  left: 0;
  z-index: 1;
  background: var(--surface-1);
  color: var(--dim);
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-size: 11px;
  white-space: nowrap;
}

.data-table td.idx.has-issues { color: var(--text-dim); }

.row-dot {
  display: inline-block;
  width: 5px;
  height: 5px;
  margin-left: 6px;
  border-radius: 50%;
  background: var(--amber);
  vertical-align: 2px;
}

.data-table td.editable { cursor: text; transition: background 0.15s ease; }

.data-table td.is-fixed { background: color-mix(in srgb, var(--green) 7%, transparent); }
.data-table td.is-manual { background: color-mix(in srgb, var(--amber) 8%, transparent); }
.data-table td.is-pending { background: color-mix(in srgb, var(--accent-2) 8%, transparent); }
.data-table td.is-empty { background: color-mix(in srgb, var(--rose) 7%, transparent); }
.data-table td.is-flagged { background: color-mix(in srgb, var(--rose) 8%, transparent); }

.cell-value, .cell-old, .cell-new, .cell-sugg { display: block; overflow-wrap: anywhere; }

.cell-old {
  color: var(--muted);
  text-decoration: line-through;
  text-decoration-color: color-mix(in srgb, var(--rose) 50%, transparent);
  font-size: 11.3px;
}

.cell-new { color: var(--ok-text); }
td.is-manual .cell-new { color: var(--warn-text); }

.cell-sugg {
  color: var(--accent-text);
  font-size: 11.5px;
  border-bottom: 1px dashed color-mix(in srgb, var(--accent-2) 40%, transparent);
  width: max-content;
  max-width: 100%;
}

.cell-flag {
  color: var(--bad-text);
  border-bottom: 1px dashed color-mix(in srgb, var(--rose) 45%, transparent);
  width: max-content;
  max-width: 100%;
  overflow-wrap: anywhere;
}

.conv {
  display: inline-block;
  margin-top: 1px;
  font-size: 9.5px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--dim);
}

.empty-cell {
  color: var(--bad-text);
  font-size: 11px;
  font-style: italic;
  opacity: 0.75;
}

.cell-input {
  width: 100%;
  min-width: 110px;
  padding: 3px 6px;
  border-radius: 6px;
  border: 1px solid var(--accent);
  background: var(--surface-3);
  color: var(--text);
  font-size: 12.5px;
  outline: none;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent);
}

.table-empty {
  padding: 48px 20px !important;
  text-align: center;
  color: var(--muted);
  background: transparent !important;
}

.table-empty span { display: block; width: 34px; margin: 0 auto 10px; opacity: 0.5; }
.table-empty span svg { width: 34px; height: 34px; }
.table-empty strong { display: block; color: var(--text-dim); font-size: 13px; }
.table-empty p { font-size: 12px; margin-top: 2px; }

.table-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 11px 16px;
  border-top: 1px solid var(--line);
  background: color-mix(in srgb, var(--text-color) 1.2%, transparent);
  font-size: 11.5px;
}

.pager { display: flex; align-items: center; gap: 9px; font-size: 11.5px; color: var(--text-dim); }

.hint { margin-top: 10px; padding: 0 4px; }

/* Corrections list -------------------------------------------------------- */

.chips { display: flex; flex-wrap: wrap; gap: 7px; }

.filter-chip {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 6px 12px;
  border-radius: 99px;
  border: 1px solid var(--line);
  background: color-mix(in srgb, var(--text-color) 2.5%, transparent);
  color: var(--muted);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.18s ease;
}

.filter-chip span {
  min-width: 18px;
  padding: 0 5px;
  border-radius: 99px;
  background: color-mix(in srgb, var(--text-color) 6%, transparent);
  font-size: 10.5px;
  text-align: center;
  color: var(--text-dim);
}

.filter-chip:hover { border-color: var(--line-strong); color: var(--text-dim); }

.filter-chip.active {
  background: color-mix(in srgb, var(--accent) 16%, transparent);
  border-color: color-mix(in srgb, var(--accent) 40%, transparent);
  color: var(--text);
}

.filter-chip.active span { background: color-mix(in srgb, var(--accent) 35%, transparent); color: var(--text); }

.bulk-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  flex-wrap: wrap;
  padding: 4px 6px 14px;
}

.bulk-copy { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.bulk-copy strong { font-size: 14.5px; font-family: var(--display); }
.bulk-copy .muted { font-size: 11.5px; }
.bulk-actions { display: flex; gap: 8px; flex-wrap: wrap; }

.issue-list { display: flex; flex-direction: column; gap: 10px; }

.issue {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--issue-grad);
  padding: 15px 17px;
  transition: border-color 0.2s ease, transform 0.15s ease;
}

.issue:hover { border-color: var(--line-strong); transform: translateX(2px); }

.issue-head {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 7px;
  margin-bottom: 9px;
}

.issue-row {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-dim);
  padding: 2px 8px;
  border-radius: 6px;
  background: color-mix(in srgb, var(--text-color) 5%, transparent);
  font-variant-numeric: tabular-nums;
}

.issue-column {
  font-size: 11.5px;
  color: var(--muted);
  overflow-wrap: anywhere;
}

.status-chip {
  display: inline-flex;
  align-items: center;
  padding: 2px 9px;
  border-radius: 99px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.02em;
  border: 1px solid transparent;
}

.status-chip.ok { color: var(--ok-text); background: color-mix(in srgb, var(--green) 10%, transparent); border-color: color-mix(in srgb, var(--green) 25%, transparent); }
.status-chip.warn { color: var(--warn-text); background: color-mix(in srgb, var(--amber) 10%, transparent); border-color: color-mix(in srgb, var(--amber) 25%, transparent); }
.status-chip.bad { color: var(--bad-text); background: color-mix(in srgb, var(--rose) 10%, transparent); border-color: color-mix(in srgb, var(--rose) 25%, transparent); }
.status-chip.manual { color: var(--accent-text); background: color-mix(in srgb, var(--accent-2) 10%, transparent); border-color: color-mix(in srgb, var(--accent-2) 28%, transparent); }
.status-chip.muted { color: var(--muted); background: color-mix(in srgb, var(--text-color) 5%, transparent); border-color: color-mix(in srgb, var(--text-color) 8%, transparent); }

.confidence {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
  font-size: 10.5px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}

.conf-track {
  width: 52px;
  height: 4px;
  border-radius: 99px;
  background: color-mix(in srgb, var(--text-color) 8%, transparent);
  overflow: hidden;
}

.conf-track i {
  display: block;
  height: 100%;
  width: var(--w, 0%);
  border-radius: inherit;
  background: linear-gradient(90deg, var(--accent), var(--cyan));
}

.issue-main {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px;
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  font-size: 12.5px;
  margin-bottom: 6px;
}

.issue-original {
  padding: 2px 7px;
  border-radius: 6px;
  background: color-mix(in srgb, var(--rose) 9%, transparent);
  color: var(--bad-text);
  overflow-wrap: anywhere;
}

.issue-original.muted { color: var(--muted); background: color-mix(in srgb, var(--text-color) 4%, transparent); }
.issue-arrow { color: var(--faint); }
.issue-suggested { padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--green) 9%, transparent); color: var(--ok-text); overflow-wrap: anywhere; }
.issue-suggested.muted { color: var(--muted); background: color-mix(in srgb, var(--text-color) 4%, transparent); }

.issue-why { color: var(--muted); font-size: 11.8px; line-height: 1.5; }

.issue-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 10px;
}

.cat-chip {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent-text);
  padding: 2px 8px;
  border-radius: 6px;
  border: 1px solid color-mix(in srgb, var(--accent) 22%, transparent);
  background: color-mix(in srgb, var(--accent) 7%, transparent);
}

.issue-actions { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }

.load-more { display: flex; justify-content: center; padding: 18px 0 4px; }

.empty-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 34px 20px;
  text-align: center;
  color: var(--muted);
}

.empty-icon { width: 32px; height: 32px; color: var(--accent-text); opacity: 0.7; margin-bottom: 4px; }
.empty-icon svg { width: 32px; height: 32px; }
.empty-panel strong { color: var(--text-dim); font-size: 13.5px; }
.empty-panel p { font-size: 12px; }

/* Downloads --------------------------------------------------------------- */

.download-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}

.download-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 24px;
}

.download-card h3 { font-size: 16.5px; margin-top: 6px; }
.download-card p { color: var(--muted); font-size: 12.5px; }

.download-card.accent {
  border-color: color-mix(in srgb, var(--accent) 40%, transparent);
  background: var(--download-accent-grad);
  box-shadow: 0 30px 70px -40px color-mix(in srgb, var(--accent) 80%, transparent);
}

.download-icon {
  width: 42px;
  height: 42px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  color: var(--on-accent);
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  box-shadow: 0 14px 30px -14px var(--glow-strong);
  margin-bottom: 4px;
}

.download-icon.soft {
  background: color-mix(in srgb, var(--accent) 12%, transparent);
  color: var(--accent-text);
  border: 1px solid var(--line);
  box-shadow: none;
}

.download-icon svg { width: 20px; height: 20px; }

.download-list {
  list-style: none;
  margin: 8px 0 18px;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 9px;
  font-size: 12px;
  color: var(--text-dim);
}

.download-list li { display: flex; align-items: center; gap: 9px; }

.download-card .btn { margin-top: auto; }

.session-card { margin-top: 18px; }

.session-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0 32px;
}

.summary-item {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 14px;
  padding: 9px 0;
  border-bottom: 1px dashed color-mix(in srgb, var(--text-color) 6%, transparent);
  font-size: 12.5px;
}

.summary-item span { color: var(--muted); }
.summary-item strong { color: var(--text-dim); font-weight: 600; text-align: right; overflow-wrap: anywhere; }

/* Toasts / overlay -------------------------------------------------------- */

.toasts {
  position: fixed;
  right: 22px;
  bottom: 22px;
  z-index: 60;
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-width: min(360px, calc(100vw - 44px));
}

.toast {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  padding: 12px 15px;
  border-radius: 12px;
  border: 1px solid var(--line-strong);
  background: var(--toast-bg);
  color: var(--text-dim);
  font-size: 12.5px;
  box-shadow: 0 24px 50px -20px var(--shadow-ink-strong);
  animation: toast-in 0.25s ease both;
}

.toast.leaving { animation: toast-out 0.3s ease both; }

.toast-dot {
  flex: none;
  width: 8px;
  height: 8px;
  margin-top: 5px;
  border-radius: 50%;
  background: var(--green);
}

.toast.bad .toast-dot { background: var(--rose); }
.toast.info .toast-dot { background: var(--accent-2); }

@keyframes toast-in {
  from { opacity: 0; transform: translateY(10px) scale(0.98); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes toast-out {
  to { opacity: 0; transform: translateY(6px) scale(0.98); }
}

.overlay {
  position: fixed;
  inset: 0;
  z-index: 80;
  display: grid;
  place-items: center;
  padding: 20px;
  background: var(--overlay-bg);
  backdrop-filter: blur(6px);
  animation: fade-in 0.2s ease both;
}

@keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }

.overlay-card {
  width: min(360px, 100%);
  padding: 30px 28px 26px;
  border-radius: 18px;
  border: 1px solid var(--line-strong);
  background: var(--overlay-card);
  text-align: center;
  box-shadow: 0 40px 100px -30px var(--shadow-ink-strong);
}

.overlay-card h3 { font-size: 16.5px; margin: 16px 0 5px; }
.overlay-card p { color: var(--muted); font-size: 12.5px; min-height: 36px; }

.spinner {
  width: 40px;
  height: 40px;
  margin: 0 auto;
  border-radius: 50%;
  border: 3px solid color-mix(in srgb, var(--accent) 18%, transparent);
  border-top-color: var(--accent);
  border-right-color: var(--accent-text);
  animation: spin 0.9s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.overlay-bar {
  height: 3px;
  margin-top: 18px;
  border-radius: 99px;
  background: color-mix(in srgb, var(--text-color) 6%, transparent);
  overflow: hidden;
}

.overlay-bar span {
  display: block;
  width: 36%;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--accent), var(--cyan));
  animation: slide 1.3s ease-in-out infinite;
}

/* Responsive -------------------------------------------------------------- */

@media (max-width: 1240px) {
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 1080px) {
  .hero { grid-template-columns: 1fr; padding: 30px; }
  .hero-art { display: none; }
  .grid-2 { grid-template-columns: 1fr; }
  .grid-2 .card { min-height: 0; }
  .download-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .upload-grid { grid-template-columns: 1fr; }
  .session-grid { grid-template-columns: 1fr; }
}

@media (max-width: 900px) {
  .app { display: block; }

  .sidebar {
    position: sticky;
    top: 0;
    z-index: 30;
    height: auto;
    flex-direction: row;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    overflow-x: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }

  .brand { flex: none; padding: 0; }
  .brand-text small { display: none; }
  .brand-mark { width: 32px; height: 32px; border-radius: 10px; }
  .brand-mark svg { width: 17px; height: 17px; }

  .nav { flex-direction: row; gap: 4px; flex: 1; }

  .nav-item {
    width: auto;
    padding: 7px 10px;
    font-size: 12px;
    white-space: nowrap;
  }

  .nav-item svg { width: 15px; height: 15px; }
  .nav-item.active::before { display: none; }
  .side-card, .side-tip, .side-foot, #new-file-btn { display: none; }

  .main { padding: 16px 16px 60px; }
  .topbar { padding-top: 4px; }
}

@media (max-width: 720px) {
  .topbar { flex-wrap: wrap; }
  .topbar-actions { width: 100%; }
  .topbar-actions .btn { flex: 1; }
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .welcome { flex-direction: column; align-items: flex-start; padding: 22px; }
  .health-wrap { flex-direction: row; width: auto; gap: 14px; }
  .health-ring { width: 84px; height: 84px; }
  .health-inner { width: 64px; height: 64px; }
  .health-inner strong { font-size: 18px; }
  .health-inner span { font-size: 8px; }
  .health-wrap > small { text-align: left; max-width: 180px; }
  .next-card { flex-direction: column; align-items: flex-start; }
  .download-grid { grid-template-columns: 1fr; }
  .feature-row { grid-template-columns: 1fr; }
  .toolbar-card { flex-direction: column; align-items: stretch; }
  .review-tools { width: 100%; }
  .search { flex: 1; }
  .search input { width: 100%; }
  .review-tools select { flex: 1; }
  .legend { order: 2; }
  .cat-row { grid-template-columns: 104px minmax(0, 1fr) 34px; }
  .bulk-actions { width: 100%; }
  .bulk-actions .btn { flex: 1; }
  .issue-actions { margin-left: 0; }
  .table-foot { flex-direction: column; align-items: flex-start; }
  .upload-actions { flex-direction: column-reverse; align-items: stretch; }
  .upload-actions .btn { width: 100%; }
  .upload-actions .hint { text-align: center; }
}

@media (max-width: 620px) {
  .brand-text { display: none; }
  .nav { justify-content: flex-end; }
  .nav-item { padding: 8px; }
  .nav-label { display: none; }
  .nav-item svg { width: 18px; height: 18px; }
}

@media (max-width: 480px) {
  .stats { grid-template-columns: 1fr; }
  .hero { padding: 22px; }
  .card { padding: 18px; }
  .topbar-copy h1 { font-size: 19px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* Theme settings ---------------------------------------------------------- */

.appearance-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  gap: 18px;
  align-items: start;
}

.appearance-card { min-width: 0; }
.appearance-card .card-head:first-child { margin-bottom: 14px; }

.mode-options {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.mode-option {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  padding: 13px;
  border: 1.5px solid var(--line);
  border-radius: 13px;
  background: var(--ghost);
  color: var(--text);
  text-align: left;
  cursor: pointer;
  transition: border-color 0.2s ease, background 0.2s ease, transform 0.15s ease,
              box-shadow 0.2s ease;
}

.mode-option:hover { border-color: var(--line-strong); background: var(--hover-bg); }
.mode-option.is-selected {
  border-color: var(--accent);
  background: color-mix(in srgb, var(--accent) 10%, transparent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 9%, transparent);
}

.mode-icon {
  flex: none;
  width: 36px;
  height: 36px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: color-mix(in srgb, var(--accent) 12%, transparent);
  color: var(--accent-text);
}
.mode-icon svg { width: 18px; height: 18px; }

.mode-copy { display: flex; flex-direction: column; gap: 1px; min-width: 0; padding-right: 10px; }
.mode-copy strong { font-size: 12.8px; }
.mode-copy small { color: var(--muted); font-size: 10.8px; line-height: 1.35; }

.mode-check {
  position: absolute;
  top: 9px;
  right: 9px;
  width: 17px;
  height: 17px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--accent);
  color: var(--on-accent);
  font-size: 10px;
  font-weight: 800;
  opacity: 0;
  transform: scale(0.6);
  transition: opacity 0.18s ease, transform 0.18s ease;
}
.mode-option.is-selected .mode-check { opacity: 1; transform: none; }

.settings-divider { height: 1px; background: var(--line); margin: 20px 0 18px; }
.settings-head { margin-bottom: 14px; }

.theme-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 11px;
}

.theme-card {
  position: relative;
  min-width: 0;
  padding: 8px;
  border: 1.5px solid var(--line);
  border-radius: 14px;
  background: var(--ghost);
  color: var(--text);
  text-align: left;
  cursor: pointer;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease,
              background 0.2s ease;
}
.theme-card:hover { border-color: var(--tp-primary); background: var(--hover-bg); transform: translateY(-2px); }
.theme-card.is-selected,
.theme-card[aria-checked="true"] {
  border-color: var(--tp-primary);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--tp-primary) 15%, transparent);
}
.theme-card:focus-visible { outline-color: var(--tp-primary); }

.theme-preview {
  position: relative;
  display: block;
  padding: 6px;
  border: 1px solid var(--tp-line);
  border-radius: 9px;
  background: var(--tp-bg);
  overflow: hidden;
}

.tp-window {
  display: block;
  border: 1px solid var(--tp-line);
  border-radius: 6px;
  background: var(--tp-card);
  overflow: hidden;
}

.tp-top {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 5px 7px;
  border-bottom: 1px solid var(--tp-line);
}
.tp-brand { width: 6px; height: 6px; border-radius: 2px; background: var(--tp-primary); }
.tp-nav { width: 18px; height: 3px; border-radius: 2px; background: color-mix(in srgb, var(--tp-text) 18%, transparent); }
.tp-nav.short { width: 10px; }

.tp-body { display: flex; min-height: 58px; }
.tp-rail {
  width: 20px;
  flex: none;
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 6px 4px;
  border-right: 1px solid var(--tp-line);
  background: color-mix(in srgb, var(--tp-primary) 7%, var(--tp-card));
}
.tp-rail i {
  display: block;
  width: 100%;
  height: 3px;
  border-radius: 2px;
  background: color-mix(in srgb, var(--tp-text) 18%, transparent);
}
.tp-rail i:first-child { background: var(--tp-primary); }

.tp-main { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 4px; padding: 6px; }
.tp-heading { display: block; width: 42%; height: 4px; border-radius: 2px; background: var(--tp-text); opacity: 0.7; }
.tp-tiles { display: flex; gap: 3px; }
.tp-tiles i {
  flex: 1;
  height: 9px;
  border: 1px solid var(--tp-line);
  border-radius: 2px;
  background: var(--tp-bg);
}
.tp-tiles i:first-child { background: color-mix(in srgb, var(--tp-primary) 20%, var(--tp-card)); border-color: transparent; }
.tp-chart { display: flex; align-items: flex-end; gap: 2px; height: 17px; }
.tp-chart i {
  flex: 1;
  height: var(--h, 40%);
  border-radius: 1px 1px 0 0;
  background: var(--tp-primary);
  opacity: 0.82;
}
.tp-chart i:nth-child(even) { background: var(--tp-secondary); opacity: 0.62; }
.tp-cta { display: block; width: 34%; height: 5px; border-radius: 2px; background: var(--tp-primary); }

.theme-check {
  position: absolute;
  top: 11px;
  right: 11px;
  width: 18px;
  height: 18px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--tp-primary);
  color: #fff;
  font-size: 10.5px;
  font-weight: 800;
  box-shadow: 0 2px 6px color-mix(in srgb, var(--tp-text) 22%, transparent);
  opacity: 0;
  transform: scale(0.6);
  transition: opacity 0.18s ease, transform 0.18s ease;
}
.theme-card.is-selected .theme-check,
.theme-card[aria-checked="true"] .theme-check { opacity: 1; transform: none; }

.theme-meta { display: flex; align-items: center; gap: 6px; padding: 7px 2px 1px; }
.theme-copy { display: flex; flex-direction: column; min-width: 0; flex: 1; }
.theme-copy strong { font-size: 12.4px; font-weight: 600; }
.theme-copy small {
  color: var(--muted);
  font-size: 10.2px;
  line-height: 1.35;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.theme-swatches { display: flex; align-items: center; gap: 3px; flex: none; }
.theme-swatches i {
  width: 9px;
  height: 9px;
  border: 1px solid transparent;
  border-radius: 50%;
  display: block;
}

.appearance-side { min-width: 0; }
.preview-card { position: sticky; top: 84px; }

.live-preview {
  padding: 13px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: linear-gradient(155deg, var(--panel-2), var(--panel-solid));
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--text-color) 5%, transparent);
}

.lp-head { display: flex; align-items: center; gap: 9px; margin-bottom: 10px; }
.lp-logo {
  flex: none;
  width: 27px;
  height: 27px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 2px;
  padding: 6px;
  border-radius: 8px;
  background: var(--icon-grad);
}
.lp-logo i { display: block; height: 2.5px; border-radius: 2px; background: var(--on-accent); }
.lp-logo i:nth-child(2) { width: 72%; opacity: 0.85; }
.lp-logo i:nth-child(3) { width: 48%; opacity: 0.7; }
.lp-title { display: flex; flex-direction: column; min-width: 0; }
.lp-title strong {
  font-family: var(--display);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.lp-title small { color: var(--muted); font-size: 9.5px; }
.lp-badge {
  margin-left: auto;
  flex: none;
  padding: 2px 7px;
  border: 1px solid color-mix(in srgb, var(--accent) 24%, transparent);
  border-radius: 99px;
  background: color-mix(in srgb, var(--accent) 10%, transparent);
  color: var(--accent-text);
  font-size: 8.5px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.lp-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
.lp-stat {
  padding: 7px 8px;
  border: 1px solid var(--line-soft);
  border-radius: 9px;
  background: color-mix(in srgb, var(--text-color) 3%, transparent);
}
.lp-stat small {
  display: block;
  color: var(--muted);
  font-size: 8.4px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.lp-stat strong { display: block; font-family: var(--display); font-size: 15px; line-height: 1.3; }

.lp-bars {
  display: flex;
  align-items: flex-end;
  gap: 4px;
  height: 44px;
  margin-top: 10px;
  padding: 0 2px;
  border-bottom: 1px solid var(--line);
}
.lp-bars i { flex: 1; height: var(--h, 40%); border-radius: 3px 3px 0 0; background: var(--bar-grad); opacity: 0.9; }
.lp-bars i:nth-child(3n) { background: var(--bar-grad-2); opacity: 0.78; }
.lp-legend { display: flex; gap: 10px; margin-top: 6px; color: var(--muted); font-size: 8.5px; }
.lp-legend span { display: inline-flex; align-items: center; gap: 4px; }
.lp-legend i { width: 6px; height: 6px; border-radius: 2px; background: var(--accent); }
.lp-legend span:last-child i { background: var(--cyan); }

.lp-actions { display: flex; gap: 6px; margin-top: 10px; }
.lp-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 5px 9px;
  border-radius: 6px;
  font-size: 9px;
  font-weight: 600;
  white-space: nowrap;
}
.lp-btn.primary { flex: 1.25; background: var(--button-grad); color: var(--on-accent); }
.lp-btn.ghost { flex: 0.75; border: 1px solid var(--line); color: var(--text-dim); }

.pref-line { display: flex; align-items: flex-start; gap: 8px; margin-top: 14px; }
.pref-status {
  flex: 1;
  min-height: 38px;
  display: flex;
  align-items: flex-start;
  gap: 7px;
  padding: 9px 11px;
  border: 1px solid var(--line-soft);
  border-radius: 9px;
  background: var(--ghost);
  color: var(--muted);
  font-size: 11.2px;
  line-height: 1.4;
}
.pref-status::before {
  content: "";
  flex: none;
  width: 7px;
  height: 7px;
  margin-top: 4px;
  border-radius: 50%;
  background: var(--muted);
}
.pref-status.is-preview::before,
.pref-status.is-warn::before { background: var(--amber); }
.pref-status.is-saved::before { background: var(--green); }
.pref-line .pill { margin-top: 2px; }

.settings-actions { display: grid; gap: 8px; margin-top: 12px; }
.actions-row { display: flex; gap: 8px; flex-wrap: wrap; }
.actions-row .btn { flex: 1 1 120px; padding-left: 12px; padding-right: 12px; }
.settings-actions .btn { padding-top: 9px; padding-bottom: 9px; }
.btn.danger-text { color: var(--bad-text); }
.btn.danger-text:hover:not(:disabled) {
  border-color: color-mix(in srgb, var(--rose) 38%, transparent);
  background: color-mix(in srgb, var(--rose) 9%, transparent);
}

.hint.tiny { font-size: 10.8px; line-height: 1.45; margin-top: 10px; }

/* Light appearance refinements ------------------------------------------- */

[data-mode="light"] .card { backdrop-filter: none; }
[data-mode="light"] .sidebar,
[data-mode="light"] .topbar { backdrop-filter: blur(12px); }
[data-mode="light"] .orb { opacity: 0.26; filter: blur(100px); }
[data-mode="light"] .orb-b { opacity: 0.18; }
[data-mode="light"] .grid-lines { opacity: 0.68; }
[data-mode="light"] .theme-preview { box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--tp-text) 3%, transparent); }
[data-mode="light"] .mock-badge { box-shadow: 0 16px 34px -20px var(--shadow-ink-strong); }
[data-mode="light"] .toast { box-shadow: 0 18px 44px -26px var(--shadow-ink-strong); }
[data-mode="light"] .overlay-card { box-shadow: 0 40px 80px -40px var(--shadow-ink-strong); }

/* Smooth palette hand-off */

html.is-theming,
html.is-theming * {
  transition: background-color 0.28s ease, border-color 0.28s ease, color 0.28s ease,
              box-shadow 0.28s ease !important;
}

@media (max-width: 1180px) {
  .appearance-grid { grid-template-columns: minmax(0, 1fr) 300px; }
}

@media (max-width: 1080px) {
  .appearance-grid { grid-template-columns: 1fr; }
  .preview-card { position: static; }
  .theme-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}

@media (max-width: 720px) {
  .mode-options { grid-template-columns: 1fr; }
  .theme-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .mode-copy small { font-size: 11px; }
}

@media (max-width: 440px) {
  .theme-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px; }
  .theme-card { padding: 6px; border-radius: 12px; }
  .theme-preview { padding: 4px; }
  .theme-meta { padding: 6px 1px 0; gap: 4px; }
  .theme-copy strong { font-size: 11.4px; }
  .theme-copy small { display: none; }
  .theme-swatches { gap: 2px; }
  .theme-swatches i { width: 7px; height: 7px; }
  .theme-swatches .theme-swatch-card { display: none; }
  .pref-line { flex-direction: column; align-items: stretch; }
  .pref-line .pill { align-self: flex-start; }
}

@media (prefers-reduced-motion: reduce) {
  html.is-theming,
  html.is-theming * { transition: none !important; }
}

==============================================================================
FILE: static/theme.js
==============================================================================
/* DataFix theme manager — palettes, appearance modes and preferences. */
(function () {
  "use strict";

  const THEMES = [
    {
      id: "ocean-blue",
      name: "Ocean Blue",
      description: "Professional blue & white",
      primary: "#1d4ed8",
      secondary: "#3b82f6",
      accent: "#06b6d4",
      bg: "#f2f7ff",
      card: "#ffffff",
      text: "#0f1f3a",
      border: "#d6e2f3",
      button: "#1d4ed8",
      hover: "#1e40af",
    },
    {
      id: "royal-purple",
      name: "Royal Purple",
      description: "Elegant purple & lavender",
      primary: "#6d28d9",
      secondary: "#a855f7",
      accent: "#c084fc",
      bg: "#f9f6ff",
      card: "#ffffff",
      text: "#211539",
      border: "#e6ddf6",
      button: "#6d28d9",
      hover: "#5b21b6",
    },
    {
      id: "emerald-green",
      name: "Emerald Green",
      description: "Fresh green & white",
      primary: "#047857",
      secondary: "#10b981",
      accent: "#14b8a6",
      bg: "#f2fbf6",
      card: "#ffffff",
      text: "#0c241b",
      border: "#d3eadf",
      button: "#047857",
      hover: "#065f46",
    },
    {
      id: "sunset-orange",
      name: "Sunset Orange",
      description: "Warm orange & cream",
      primary: "#c2410c",
      secondary: "#f97316",
      accent: "#f59e0b",
      bg: "#fff8ef",
      card: "#fffdf8",
      text: "#2b1807",
      border: "#f0dfc9",
      button: "#c2410c",
      hover: "#9a3412",
    },
    {
      id: "rose-pink",
      name: "Rose Pink",
      description: "Soft pink & white",
      primary: "#be185d",
      secondary: "#ec4899",
      accent: "#f472b6",
      bg: "#fff5f8",
      card: "#ffffff",
      text: "#2b101b",
      border: "#f4dae4",
      button: "#be185d",
      hover: "#9d174d",
    },
    {
      id: "midnight-navy",
      name: "Midnight Navy",
      description: "Deep navy & light blue",
      primary: "#1e3a8a",
      secondary: "#2563eb",
      accent: "#0ea5e9",
      bg: "#f1f5fb",
      card: "#ffffff",
      text: "#0b1730",
      border: "#d6dfec",
      button: "#1e3a8a",
      hover: "#172c6b",
    },
    {
      id: "teal-breeze",
      name: "Teal Breeze",
      description: "Teal & mint",
      primary: "#0f766e",
      secondary: "#14b8a6",
      accent: "#2dd4bf",
      bg: "#f1fbf9",
      card: "#ffffff",
      text: "#0a2522",
      border: "#d1e9e3",
      button: "#0f766e",
      hover: "#115e59",
    },
    {
      id: "crimson-red",
      name: "Crimson Red",
      description: "Red & soft white",
      primary: "#b91c1c",
      secondary: "#ef4444",
      accent: "#f87171",
      bg: "#fff5f5",
      card: "#ffffff",
      text: "#2b0f0f",
      border: "#f2d9d9",
      button: "#b91c1c",
      hover: "#991b1b",
    },
    {
      id: "golden-amber",
      name: "Golden Amber",
      description: "Golden yellow & cream",
      primary: "#b45309",
      secondary: "#d97706",
      accent: "#f59e0b",
      bg: "#fffaef",
      card: "#fffdf7",
      text: "#2b1d05",
      border: "#efe1c1",
      button: "#b45309",
      hover: "#92400e",
    },
    {
      id: "sky-blue",
      name: "Sky Blue",
      description: "Light blue & white",
      primary: "#0369a1",
      secondary: "#0ea5e9",
      accent: "#38bdf8",
      bg: "#f1f9ff",
      card: "#ffffff",
      text: "#082334",
      border: "#d3e6f4",
      button: "#0369a1",
      hover: "#075985",
    },
    {
      id: "lavender-dream",
      name: "Lavender Dream",
      description: "Lavender & pale purple",
      primary: "#7e22ce",
      secondary: "#a855f7",
      accent: "#d8b4fe",
      bg: "#fbf7ff",
      card: "#ffffff",
      text: "#22143f",
      border: "#e8ddf8",
      button: "#7e22ce",
      hover: "#6b21a8",
    },
    {
      id: "graphite-grey",
      name: "Graphite Grey",
      description: "Modern grey & white",
      primary: "#374151",
      secondary: "#6b7280",
      accent: "#9ca3af",
      bg: "#f4f5f7",
      card: "#ffffff",
      text: "#111827",
      border: "#dfe3e8",
      button: "#374151",
      hover: "#1f2937",
    },
  ];

  const THEME_BY_ID = Object.fromEntries(THEMES.map((theme) => [theme.id, theme]));
  const MODES = ["light", "dark", "system"];
  const DEFAULT_PREFS = { appearance: "system", theme: "royal-purple" };
  const STORAGE_KEY = "datafix.preferences.v1";
  const PALETTE_VARS = [
    ["--primary-color", "primary"],
    ["--secondary-color", "secondary"],
    ["--accent-color", "accent"],
    ["--background-color", "bg"],
    ["--card-background", "card"],
    ["--text-color", "text"],
    ["--border-color", "border"],
    ["--button-color", "button"],
    ["--hover-color", "hover"],
  ];

  const root = document.documentElement;
  const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  let saved = { ...DEFAULT_PREFS };
  let draft = { ...DEFAULT_PREFS };
  let backend = { ready: false, saved: false, prefs: null, synced: false };
  let saving = false;
  let mounted = false;
  let fadeTimer = 0;

  /* --------------------------------------------------------------------- */
  /* Boot (runs from <head>, before first paint)                           */
  /* --------------------------------------------------------------------- */

  const localPrefs = readLocal();
  const initialPrefs = normalisePrefs(window.DATAFIX_INITIAL_PREFS, null);
  if (localPrefs) {
    saved = localPrefs;
    draft = { ...localPrefs };
  } else if (initialPrefs) {
    saved = initialPrefs;
    draft = { ...initialPrefs };
    if (window.DATAFIX_INITIAL_PREFS && window.DATAFIX_INITIAL_PREFS.saved) writeLocal(saved);
  }
  applyPrefs(saved, false);

  if (media) {
    const onChange = () => {
      if (draft.appearance === "system") {
        applyPrefs(draft, true);
        render();
      }
    };
    if (media.addEventListener) media.addEventListener("change", onChange);
    else if (media.addListener) media.addListener(onChange);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }

  window.addEventListener("storage", (event) => {
    if (event.key !== STORAGE_KEY) return;
    const next = readLocal();
    if (!next) return;
    saved = { ...next };
    draft = { ...next };
    applyPrefs(saved, true);
    render();
  });

  /* --------------------------------------------------------------------- */
  /* Preferences                                                           */
  /* --------------------------------------------------------------------- */

  function isTheme(id) {
    return typeof id === "string" && Object.prototype.hasOwnProperty.call(THEME_BY_ID, id);
  }

  function normalisePrefs(raw, fallback) {
    if (!raw || typeof raw !== "object") return fallback ? { ...fallback } : null;
    const appearance = MODES.includes(raw.appearance)
      ? raw.appearance
      : fallback
        ? fallback.appearance
        : DEFAULT_PREFS.appearance;
    const theme = isTheme(raw.theme)
      ? raw.theme
      : fallback
        ? fallback.theme
        : DEFAULT_PREFS.theme;
    return { appearance, theme };
  }

  function readLocal() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || !MODES.includes(parsed.appearance) || !isTheme(parsed.theme)) return null;
      return { appearance: parsed.appearance, theme: parsed.theme };
    } catch (error) {
      return null;
    }
  }

  function writeLocal(prefs) {
    try {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ appearance: prefs.appearance, theme: prefs.theme })
      );
    } catch (error) {
      /* storage can be unavailable; the backend profile still works */
    }
  }

  function samePrefs(a, b) {
    return Boolean(a && b && a.appearance === b.appearance && a.theme === b.theme);
  }

  function resolvedMode(appearance) {
    if (appearance === "system") return media && media.matches ? "dark" : "light";
    return appearance === "dark" ? "dark" : "light";
  }

  function themeFor(id) {
    return THEME_BY_ID[id] || THEME_BY_ID[DEFAULT_PREFS.theme];
  }

  /* --------------------------------------------------------------------- */
  /* Applying                                                              */
  /* --------------------------------------------------------------------- */

  function applyPrefs(prefs, animate) {
    const theme = themeFor(prefs.theme);
    const mode = resolvedMode(prefs.appearance);

    if (animate && !root.classList.contains("is-theming")) {
      root.classList.add("is-theming");
    }
    if (animate) {
      window.clearTimeout(fadeTimer);
      fadeTimer = window.setTimeout(() => root.classList.remove("is-theming"), 380);
    }

    root.setAttribute("data-mode", mode);
    root.setAttribute("data-appearance", prefs.appearance);
    root.setAttribute("data-theme", theme.id);
    root.style.colorScheme = mode;

    if (mode === "light") {
      PALETTE_VARS.forEach(([variable, key]) => root.style.setProperty(variable, theme[key]));
    } else {
      PALETTE_VARS.forEach(([variable]) => root.style.removeProperty(variable));
    }

    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", mode === "light" ? theme.bg : "#0a0713");
  }

  function modeLabel(appearance) {
    const resolved = resolvedMode(appearance);
    if (appearance === "system") {
      return `System · ${resolved === "dark" ? "Dark" : "Light"}`;
    }
    return resolved === "dark" ? "Dark" : "Light";
  }

  /* --------------------------------------------------------------------- */
  /* Settings UI                                                           */
  /* --------------------------------------------------------------------- */

  function mount() {
    if (mounted) return;
    const grid = document.getElementById("theme-grid");
    if (!grid) return;
    mounted = true;

    grid.innerHTML = THEMES.map(themeCard).join("");
    grid.addEventListener("click", (event) => {
      const card = event.target.closest(".theme-card");
      if (card) selectTheme(card.dataset.theme);
    });
    grid.addEventListener("keydown", (event) => {
      const cards = Array.from(grid.querySelectorAll(".theme-card"));
      const index = cards.indexOf(document.activeElement);
      if (index < 0) return;
      let next = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % cards.length;
      else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + cards.length) % cards.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = cards.length - 1;
      if (next === null) return;
      event.preventDefault();
      cards[next].focus();
      selectTheme(cards[next].dataset.theme);
    });

    const modeBox = document.getElementById("mode-options");
    if (modeBox) {
      modeBox.addEventListener("click", (event) => {
        const option = event.target.closest(".mode-option");
        if (option) selectMode(option.dataset.mode);
      });
      modeBox.addEventListener("keydown", (event) => {
        const options = Array.from(modeBox.querySelectorAll(".mode-option"));
        const index = options.indexOf(document.activeElement);
        if (index < 0) return;
        let next = null;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % options.length;
        else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + options.length) % options.length;
        if (next === null) return;
        event.preventDefault();
        options[next].focus();
        selectMode(options[next].dataset.mode);
      });
    }

    const applyButton = document.getElementById("apply-theme");
    const saveButton = document.getElementById("save-prefs");
    const resetButton = document.getElementById("reset-theme");
    if (applyButton) applyButton.addEventListener("click", applyDraft);
    if (saveButton) saveButton.addEventListener("click", savePreferences);
    if (resetButton) resetButton.addEventListener("click", resetPreferences);

    render();
    loadServerPreferences();
  }

  function themeCard(theme) {
    const style = [
      `--tp-bg:${theme.bg}`,
      `--tp-card:${theme.card}`,
      `--tp-text:${theme.text}`,
      `--tp-line:${theme.border}`,
      `--tp-primary:${theme.primary}`,
      `--tp-secondary:${theme.secondary}`,
      `--tp-accent:${theme.accent}`,
    ].join(";");
    return `
      <button class="theme-card" type="button" role="radio" aria-checked="false"
              data-theme="${theme.id}" title="${theme.name} — ${theme.description}"
              aria-label="${theme.name} theme, ${theme.description}" style="${style}">
        <span class="theme-preview">
          <span class="tp-window">
            <span class="tp-top"><i class="tp-brand"></i><i class="tp-nav"></i><i class="tp-nav short"></i></span>
            <span class="tp-body">
              <span class="tp-rail"><i></i><i></i><i></i><i></i></span>
              <span class="tp-main">
                <span class="tp-heading"></span>
                <span class="tp-tiles"><i></i><i></i><i></i></span>
                <span class="tp-chart">
                  <i style="--h:34%"></i><i style="--h:58%"></i><i style="--h:44%"></i>
                  <i style="--h:78%"></i><i style="--h:62%"></i>
                </span>
                <span class="tp-cta"></span>
              </span>
            </span>
          </span>
          <span class="theme-check" aria-hidden="true">✓</span>
        </span>
        <span class="theme-meta">
          <span class="theme-copy"><strong>${theme.name}</strong><small>${theme.description}</small></span>
          <span class="theme-swatches" aria-hidden="true">
            <i style="background:${theme.primary}"></i>
            <i style="background:${theme.secondary}"></i>
            <i style="background:${theme.accent}"></i>
            <i class="theme-swatch-card" style="background:${theme.card};border-color:${theme.border}"></i>
          </span>
        </span>
      </button>`;
  }

  function selectMode(mode) {
    if (!MODES.includes(mode) || saving) return;
    draft = { ...draft, appearance: mode };
    applyPrefs(draft, true);
    render();
  }

  function selectTheme(id) {
    if (!isTheme(id) || saving) return;
    let appearance = draft.appearance;
    if (resolvedMode(appearance) !== "light") appearance = "light";
    draft = { appearance, theme: id };
    applyPrefs(draft, true);
    render();
  }

  function applyDraft() {
    if (samePrefs(draft, saved)) return;
    saved = { ...draft };
    writeLocal(saved);
    backend.synced = Boolean(
      backend.saved && backend.prefs && samePrefs(saved, backend.prefs)
    );
    applyPrefs(saved, false);
    render();
    notify("Theme applied on this device.", "ok");
  }

  async function savePreferences() {
    if (saving) return;
    saving = true;
    const target = { ...draft };
    const button = document.getElementById("save-prefs");
    if (button) button.textContent = "Saving…";
    render();

    try {
      const response = await fetch("/api/user/preferences", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(target),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `Could not save preferences (${response.status}).`);
      const serverPrefs = normalisePrefs(data, target);
      if (!serverPrefs) throw new Error("The server returned an unexpected preference payload.");

      saved = target;
      writeLocal(saved);
      backend = {
        ready: true,
        saved: true,
        prefs: { ...serverPrefs },
        synced: samePrefs(saved, serverPrefs) && samePrefs(draft, saved),
      };
      if (samePrefs(draft, target)) applyPrefs(saved, false);
      notify("Preferences saved to your DataFix profile.", "ok");
    } catch (error) {
      saved = target;
      writeLocal(saved);
      backend.synced = false;
      if (samePrefs(draft, target)) applyPrefs(saved, false);
      notify(error.message || "Preferences could not be saved to your profile.", "bad");
    } finally {
      saving = false;
      if (button) button.textContent = "Save preferences";
      render();
    }
  }

  function resetPreferences() {
    if (saving) return;
    draft = { ...DEFAULT_PREFS };
    saved = { ...DEFAULT_PREFS };
    writeLocal(saved);
    backend.synced = Boolean(
      backend.saved && backend.prefs && samePrefs(saved, backend.prefs)
    );
    applyPrefs(saved, true);
    render();
    notify("Reset to the default appearance.", "ok");
  }

  async function loadServerPreferences() {
    try {
      const response = await fetch("/api/user/preferences", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("Preferences unavailable.");
      const data = await response.json();
      const serverPrefs = normalisePrefs(data, null);
      if (!serverPrefs) throw new Error("Invalid preferences.");

      backend = {
        ready: true,
        saved: Boolean(data.saved),
        prefs: { ...serverPrefs },
        synced: false,
      };

      if (!readLocal() && data.saved) {
        saved = { ...serverPrefs };
        draft = { ...serverPrefs };
        writeLocal(saved);
        applyPrefs(saved, false);
      }
      backend.synced = Boolean(
        backend.saved && samePrefs(saved, backend.prefs) && samePrefs(draft, saved)
      );
    } catch (error) {
      backend = { ready: true, saved: false, prefs: null, synced: false };
    }
    render();
  }

  /* --------------------------------------------------------------------- */
  /* Rendering                                                             */
  /* --------------------------------------------------------------------- */

  function render() {
    if (!mounted) return;
    const theme = themeFor(draft.theme);
    const resolved = resolvedMode(draft.appearance);
    const modeText = modeLabel(draft.appearance);
    const dirty = !samePrefs(draft, saved);

    document.querySelectorAll(".mode-option").forEach((option) => {
      const active = option.dataset.mode === draft.appearance;
      option.classList.toggle("is-selected", active);
      option.setAttribute("aria-checked", active ? "true" : "false");
    });
    document.querySelectorAll(".theme-card").forEach((card) => {
      const active = card.dataset.theme === draft.theme;
      card.classList.toggle("is-selected", active);
      card.setAttribute("aria-checked", active ? "true" : "false");
    });

    setText("mode-pill", modeText);
    setText("theme-pill", theme.name);
    setText("preview-theme-name", theme.name);
    setText("preview-mode-label", modeText);
    setText("preview-pill", resolved === "dark" ? "Dark" : "Light");
    setText(
      "theme-hint",
      resolved === "dark"
        ? "Dark mode keeps its midnight surfaces. Pick a palette to preview it in Light mode."
        : "Selecting a palette previews it instantly across the whole app."
    );

    const status = document.getElementById("pref-status");
    const syncPill = document.getElementById("sync-pill");
    if (status) {
      status.classList.remove("is-preview", "is-saved", "is-warn");
      if (dirty) {
        status.classList.add("is-preview");
        status.textContent = "Previewing unsaved changes — apply or save to keep them.";
        if (syncPill) syncPill.textContent = "Preview";
      } else if (backend.synced) {
        status.classList.add("is-saved");
        status.textContent = "Saved. Your theme loads automatically next time.";
        if (syncPill) syncPill.textContent = "Synced";
      } else {
        status.classList.add("is-warn");
        status.textContent = backend.ready
          ? "Applied on this device. Save to sync your local profile."
          : "Loading your saved profile…";
        if (syncPill) syncPill.textContent = "Local";
      }
    }

    const applyButton = document.getElementById("apply-theme");
    const saveButton = document.getElementById("save-prefs");
    if (applyButton) applyButton.disabled = saving || !dirty;
    if (saveButton) saveButton.disabled = saving || (!dirty && backend.synced);
  }

  function setText(id, text) {
    const node = document.getElementById(id);
    if (node) node.textContent = text;
  }

  function notify(message, kind) {
    window.dispatchEvent(new CustomEvent("datafix:toast", { detail: { message, kind: kind || "ok" } }));
  }

  /* --------------------------------------------------------------------- */
  /* Public surface (used by the settings panel and tests)                 */
  /* --------------------------------------------------------------------- */

  window.DataFixTheme = {
    themes: THEMES.map((theme) => ({ ...theme })),
    defaults: { ...DEFAULT_PREFS },
    getPreferences: () => ({ ...saved }),
    getDraft: () => ({ ...draft }),
    selectMode,
    selectTheme,
    apply: applyDraft,
    save: savePreferences,
    reset: resetPreferences,
    isDirty: () => !samePrefs(draft, saved),
  };
})();

==============================================================================
FILE: static/workspace.css
==============================================================================
/* DataFix Workspace — layout on top of the existing theme tokens. */

*, *::before, *::after { box-sizing: border-box; }

html, body { height: 100%; }
body.ws-body {
  margin: 0;
  overflow: hidden;
  font-family: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text);
  background: var(--backdrop-grad);
  color-scheme: dark;
}
[data-mode="light"] body.ws-body,
body.ws-body[data-mode="light"] { color-scheme: light; }
[data-mode="dark"] body.ws-body { color-scheme: dark; }

button, input, select, textarea { font: inherit; color: inherit; }
/* Author display rules (.ws-filter-panel etc.) would otherwise defeat the hidden attribute. */
.ws-app [hidden] { display: none !important; }
button { cursor: pointer; }
button:disabled, [data-disabled] { cursor: not-allowed; opacity: .55; }
a { color: inherit; text-decoration: none; }

.ws-app {
  display: grid;
  grid-template-columns: 274px minmax(0, 1fr);
  height: 100vh;
  min-height: 0;
}

/* ---------------------------------------------------------------- sidebar */

.ws-sidebar {
  display: flex;
  flex-direction: column;
  min-height: 0;
  border-right: 1px solid var(--line);
  background: var(--sidebar-grad);
  backdrop-filter: blur(18px);
  padding: 14px 10px 10px;
  transition: width .18s ease, transform .18s ease;
}
.ws-app.sidebar-collapsed { grid-template-columns: 0 minmax(0, 1fr); }
.ws-app.sidebar-collapsed .ws-sidebar { transform: translateX(-100%); overflow: hidden; }

.ws-brand-row { display: flex; align-items: center; gap: 8px; padding: 2px 6px 12px; }
.ws-brand { display: flex; align-items: center; gap: 10px; min-width: 0; flex: 1; }
.ws-brand-mark {
  width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center;
  background: var(--button-grad); color: #fff; box-shadow: 0 8px 20px -10px var(--glow-strong);
}
.ws-brand-mark svg { width: 21px; height: 21px; }
.ws-brand-text { display: flex; flex-direction: column; min-width: 0; }
.ws-brand-text strong { font-family: "Manrope", sans-serif; font-size: 16px; letter-spacing: .2px; }
.ws-brand-text small { color: var(--dim); font-size: 10.5px; letter-spacing: .5px; text-transform: uppercase; }

.ws-icon-btn {
  width: 32px; height: 32px; border-radius: 9px; border: 1px solid transparent;
  background: transparent; color: var(--muted); display: inline-grid; place-items: center;
  transition: background .15s ease, color .15s ease, border-color .15s ease;
}
.ws-icon-btn svg { width: 16px; height: 16px; }
.ws-icon-btn:hover { background: var(--hover-bg); color: var(--text); border-color: var(--line); }
.ws-icon-btn:disabled { opacity: .35; }

.ws-new-file {
  display: flex; align-items: center; justify-content: center; gap: 8px;
  width: 100%; padding: 11px 12px; margin-bottom: 12px;
  border-radius: 11px; border: 1px solid color-mix(in srgb, var(--accent) 35%, var(--line));
  background: var(--button-grad); color: #fff; font-weight: 600; font-size: 13.5px;
  box-shadow: 0 12px 26px -16px var(--glow);
}
.ws-new-file svg { width: 16px; height: 16px; }
.ws-new-file:hover { filter: brightness(1.07); }

.ws-file-search {
  display: flex; align-items: center; gap: 8px; padding: 8px 10px; margin-bottom: 8px;
  border-radius: 10px; border: 1px solid var(--line); background: var(--input-bg);
}
.ws-file-search svg { width: 15px; height: 15px; color: var(--dim); flex: none; }
.ws-file-search input { flex: 1; min-width: 0; border: 0; outline: 0; background: transparent; font-size: 12.5px; color: var(--text); }
.ws-file-search input::placeholder { color: var(--dim); }

.ws-nav { display: flex; flex-direction: column; gap: 2px; }
.ws-nav-item {
  display: flex; align-items: center; gap: 10px; width: 100%; padding: 8px 10px;
  border: 1px solid transparent; border-radius: 9px; background: transparent;
  color: var(--muted); font-size: 12.8px; text-align: left; transition: background .12s ease;
}
.ws-nav-item svg { width: 16px; height: 16px; flex: none; opacity: .85; }
.ws-nav-item span { flex: 1; }
.ws-nav-item em { font-style: normal; font-size: 11px; color: var(--dim); }
.ws-nav-item:hover { background: var(--hover-bg); color: var(--text); }
.ws-nav-item.is-active { background: color-mix(in srgb, var(--accent) 13%, transparent); color: var(--text); border-color: color-mix(in srgb, var(--accent) 26%, transparent); }
.ws-nav-item.is-active svg { color: var(--accent); opacity: 1; }

.ws-files-head {
  display: flex; align-items: center; justify-content: space-between;
  margin: 14px 4px 6px; font-size: 11px; letter-spacing: .6px; text-transform: uppercase;
  color: var(--dim); font-weight: 600;
}
.ws-files-head .ws-icon-btn { width: 26px; height: 26px; }
.ws-files-head .ws-icon-btn svg { width: 13px; height: 13px; }

.ws-file-list { flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; padding-right: 2px; }
.ws-file-item {
  display: flex; align-items: center; gap: 4px; padding: 6px 6px 6px 9px;
  border-radius: 9px; border: 1px solid transparent;
}
.ws-file-item:hover { background: var(--hover-bg); }
.ws-file-item.is-active { background: color-mix(in srgb, var(--accent) 12%, transparent); border-color: color-mix(in srgb, var(--accent) 24%, transparent); }
.ws-file-main { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; background: none; border: 0; padding: 2px 0; text-align: left; color: var(--text); }
.ws-file-main strong { font-size: 12.6px; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ws-file-main small { font-size: 10.5px; color: var(--dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ws-file-actions { display: flex; gap: 1px; opacity: 0; transition: opacity .12s ease; }
.ws-file-item:hover .ws-file-actions, .ws-file-item.is-active .ws-file-actions { opacity: 1; }
.ws-file-actions .ws-icon-btn { width: 24px; height: 24px; }
.ws-file-actions .ws-icon-btn svg { width: 12px; height: 12px; }
.ws-file-actions .is-on { color: var(--amber); }
.ws-file-empty { padding: 16px 10px; color: var(--dim); font-size: 12px; text-align: center; line-height: 1.5; }

.ws-sidebar-footer { border-top: 1px solid var(--line); padding-top: 8px; margin-top: 8px; display: flex; flex-direction: column; gap: 2px; }
.ws-storage-mini { font-size: 10px; color: var(--dim); padding: 8px 10px 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.ws-sidebar-scrim { position: fixed; inset: 0; background: rgba(4, 3, 10, .55); z-index: 40; }

/* ------------------------------------------------------------------- main */

.ws-main { display: flex; flex-direction: column; min-width: 0; min-height: 0; background: var(--bg-soft); }

.ws-topbar {
  display: flex; align-items: center; gap: 12px; padding: 10px 16px;
  border-bottom: 1px solid var(--line); background: var(--topbar-bg); backdrop-filter: blur(14px);
}
.ws-mobile-menu { display: none; }
.ws-title-block { min-width: 150px; max-width: 300px; }
.ws-title-row { display: flex; align-items: center; gap: 6px; }
.ws-title-block h1 { margin: 0; font-size: 16px; font-family: "Manrope", sans-serif; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ws-star-btn { background: none; border: 0; color: var(--dim); padding: 2px; display: inline-grid; place-items: center; }
.ws-star-btn svg { width: 15px; height: 15px; }
.ws-star-btn.is-on { color: var(--amber); }
.ws-star-btn.is-on svg { fill: currentColor; }
.ws-meta { margin: 2px 0 0; font-size: 11px; color: var(--dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.ws-global-search { flex: 1; display: flex; align-items: center; gap: 8px; max-width: 520px; margin: 0 auto; padding: 8px 12px; border-radius: 10px; border: 1px solid var(--line); background: var(--input-bg); }
.ws-global-search:focus-within { border-color: var(--accent); }
.ws-global-search svg { width: 15px; height: 15px; color: var(--dim); flex: none; }
.ws-global-search input { flex: 1; min-width: 0; background: transparent; border: 0; outline: 0; font-size: 12.5px; color: var(--text); }
.ws-global-search input::placeholder { color: var(--dim); }
.ws-global-search kbd { font-size: 10px; color: var(--dim); border: 1px solid var(--line); border-radius: 5px; padding: 1px 6px; background: var(--surface-2); }

.ws-top-actions { display: flex; align-items: center; gap: 4px; }
.ws-save-chip {
  font-size: 11px; padding: 5px 9px; border-radius: 999px; border: 1px solid var(--line);
  color: var(--dim); background: var(--surface-2); white-space: nowrap; max-width: 160px;
  overflow: hidden; text-overflow: ellipsis;
}
.ws-save-chip[data-state="dirty"] { color: var(--warn-text); border-color: color-mix(in srgb, var(--amber) 40%, var(--line)); }
.ws-save-chip[data-state="saving"] { color: var(--accent-text); }
.ws-save-chip[data-state="saved"] { color: var(--ok-text); border-color: color-mix(in srgb, var(--green) 35%, var(--line)); }
.ws-save-chip[data-state="error"] { color: var(--bad-text); border-color: color-mix(in srgb, var(--rose) 50%, var(--line)); cursor: pointer; }

.ws-primary-btn {
  display: inline-flex; align-items: center; gap: 7px; padding: 8px 14px; border-radius: 9px;
  border: 0; background: var(--button-grad); color: #fff; font-size: 12.5px; font-weight: 600;
  box-shadow: 0 10px 22px -16px var(--glow);
}
.ws-primary-btn svg { width: 15px; height: 15px; }
.ws-primary-btn:hover { filter: brightness(1.07); }
.ws-primary-btn.danger { background: linear-gradient(135deg, #dc2626, #b91c1c); }

.ws-small-btn {
  padding: 6px 11px; border-radius: 8px; border: 1px solid var(--line);
  background: var(--surface-2); color: var(--text); font-size: 12px;
}
.ws-small-btn:hover { border-color: var(--accent); }
.ws-small-btn.ghost { background: transparent; color: var(--muted); }

.ws-file-menu {
  position: fixed; z-index: 90; min-width: 234px; padding: 6px;
  border-radius: 12px; border: 1px solid var(--line); background: var(--panel);
  box-shadow: 0 22px 48px -14px rgba(0, 0, 0, .5);
  display: flex; flex-direction: column;
}
.ws-file-menu-title {
  padding: 7px 10px 8px; margin-bottom: 4px; border-bottom: 1px solid var(--line);
  color: var(--dim); font-size: 10.5px; letter-spacing: .5px; text-transform: uppercase;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ws-file-menu button {
  width: 100%; padding: 8px 10px; border: 0; border-radius: 8px; background: transparent;
  color: var(--text); font-size: 12.6px; text-align: left;
}
.ws-file-menu button:hover,
.ws-file-menu button:focus-visible { background: var(--hover-bg); outline: none; }
.ws-file-menu button.danger { color: #dc2626; }
.ws-file-menu button.danger:hover { background: rgba(220, 38, 38, .12); }

/* ----------------------------------------------------------------- ribbon */

.ws-ribbon { border-bottom: 1px solid var(--line); background: var(--panel); }
.ws-ribbon-tabs { display: flex; gap: 2px; padding: 0 10px; border-bottom: 1px solid var(--line); overflow-x: auto; scrollbar-width: none; }
.ws-ribbon-tab {
  border: 0; background: transparent; color: var(--dim); font-size: 10.5px; font-weight: 600;
  letter-spacing: .7px; padding: 8px 10px 7px; border-bottom: 2px solid transparent; white-space: nowrap;
}
.ws-ribbon-tab:hover { color: var(--text); }
.ws-ribbon-tab.is-active { color: var(--accent-text); border-bottom-color: var(--accent); }
.ws-ribbon-body { min-height: 52px; padding: 6px 10px; overflow-x: auto; }
.ws-ribbon-panel { display: flex; align-items: stretch; gap: 8px; }
.ws-ribbon-panel[hidden] { display: none; }
.ws-group { display: flex; align-items: center; gap: 4px; padding-right: 8px; border-right: 1px solid var(--line); }
.ws-group:last-child { border-right: 0; }
.ws-rbtn {
  display: inline-flex; align-items: center; gap: 5px; padding: 6px 9px; border-radius: 7px;
  border: 1px solid transparent; background: transparent; color: var(--text-dim);
  font-size: 12px; white-space: nowrap;
}
.ws-rbtn:hover { background: var(--hover-bg); border-color: var(--line); color: var(--text); }
.ws-rbtn small { font-size: 9.5px; color: var(--dim); text-transform: uppercase; letter-spacing: .4px; }
.ws-rbtn[data-disabled] { color: var(--dim); }
.ws-rbtn[data-disabled]:hover { background: var(--hover-bg); border-color: var(--line); }
.ws-rbtn.ws-color { gap: 5px; cursor: pointer; }
.ws-rbtn.ws-color input[type="color"] { width: 20px; height: 20px; padding: 0; border: 1px solid var(--line); border-radius: 5px; background: transparent; }
.ws-k { font-size: 9.5px; color: var(--dim); }
.ws-select { padding: 6px 8px; border-radius: 7px; border: 1px solid var(--line); background: var(--input-bg); font-size: 12px; }
.ws-ribbon-note { font-size: 10.5px; color: var(--dim); max-width: 290px; line-height: 1.35; }

/* -------------------------------------------------------------- formula bar */

.ws-formula-bar { display: flex; align-items: center; gap: 6px; padding: 6px 12px; border-bottom: 1px solid var(--line); background: var(--surface-1); }
.ws-name-box { width: 90px; padding: 5px 8px; border-radius: 6px; border: 1px solid var(--line); background: var(--input-bg); font-size: 12px; font-variant-numeric: tabular-nums; }
.ws-fx { font-size: 12px; font-style: italic; color: var(--dim); padding: 0 4px; }
.ws-formula-input { flex: 1; padding: 5px 9px; border-radius: 6px; border: 1px solid var(--line); background: var(--input-bg); font-size: 12.5px; font-family: "Consolas", "SFMono-Regular", monospace; min-width: 0; }
.ws-name-box:focus, .ws-formula-input:focus { outline: none; border-color: var(--accent); }
.ws-formula-bar .ws-icon-btn { width: 28px; height: 28px; }

/* ------------------------------------------------------------- filter panel */

.ws-filter-panel { border-bottom: 1px solid var(--line); background: var(--surface-2); padding: 8px 12px; display: flex; flex-direction: column; gap: 7px; }
.ws-filter-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-size: 12.5px; }
.ws-filter-note { flex: 1; font-size: 10.5px; color: var(--dim); }
.ws-filter-actions { display: flex; gap: 5px; }
.ws-filter-rows { display: flex; flex-direction: column; gap: 5px; max-height: 128px; overflow-y: auto; }
.ws-filter-row { display: flex; align-items: center; gap: 6px; }
.ws-filter-row select, .ws-filter-row input { padding: 5px 7px; border-radius: 6px; border: 1px solid var(--line); background: var(--input-bg); font-size: 11.5px; }
.ws-filter-row select.ws-filter-col { min-width: 150px; max-width: 240px; }
.ws-filter-row select.ws-filter-op { min-width: 128px; }
.ws-filter-row input { flex: 1; min-width: 100px; }
.ws-filter-row .ws-icon-btn { width: 25px; height: 25px; }
.ws-filter-row .ws-icon-btn svg { width: 12px; height: 12px; }
.ws-filter-foot { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-size: 11.5px; color: var(--muted); }
.ws-inline-field { display: inline-flex; align-items: center; gap: 5px; }
.ws-inline-field.grow { flex: 1; }
.ws-inline-field.grow input { flex: 1; }
.ws-inline-field input:not([type="checkbox"]), .ws-inline-field select { padding: 5px 7px; border-radius: 6px; border: 1px solid var(--line); background: var(--input-bg); font-size: 11.5px; }
.ws-inline-note { font-size: 10.5px; color: var(--dim); }

/* -------------------------------------------------------------------- grid */

.ws-sheet-area { position: relative; flex: 1; min-height: 0; display: flex; flex-direction: column; }
.ws-grid-shell { position: relative; flex: 1; min-height: 0; overflow: hidden; background: var(--surface-1); }
.ws-grid-viewport {
  position: absolute; top: 30px; left: 58px; right: 0; bottom: 0;
  overflow: auto; outline: none; background: var(--surface-1);
}
.ws-grid-viewport:focus-visible { box-shadow: inset 0 0 0 1px var(--accent); }
.ws-grid-canvas { position: relative; min-width: 100%; min-height: 100%; }

.ws-grid-corner {
  position: absolute; top: 0; left: 0; width: 58px; height: 30px; z-index: 9;
  background: var(--surface-3); border-right: 1px solid var(--line-strong); border-bottom: 1px solid var(--line-strong);
  cursor: pointer;
}
.ws-grid-colheads {
  position: absolute; top: 0; left: 58px; right: 0; height: 30px; z-index: 8; overflow: hidden;
  background: var(--surface-3); border-bottom: 1px solid var(--line-strong);
}
.ws-grid-colheads-inner { position: relative; height: 30px; will-change: transform; }
.ws-colhead {
  position: absolute; top: 0; height: 30px; display: flex; align-items: center; justify-content: center;
  border-right: 1px solid var(--line); font-size: 10.5px; color: var(--dim); font-weight: 600;
  user-select: none;
}
.ws-colhead.is-selected { background: color-mix(in srgb, var(--accent) 18%, transparent); color: var(--accent-text); }
.ws-col-resize { position: absolute; top: 0; right: -3px; width: 6px; height: 100%; cursor: col-resize; z-index: 2; }
.ws-col-resize:hover { background: color-mix(in srgb, var(--accent) 45%, transparent); }
.ws-grid-rowheads {
  position: absolute; top: 30px; left: 0; width: 58px; bottom: 0; z-index: 8; overflow: hidden;
  background: var(--surface-3); border-right: 1px solid var(--line-strong);
}
.ws-grid-rowheads-inner { position: relative; will-change: transform; }
.ws-rowhead {
  position: absolute; left: 0; width: 58px; display: flex; align-items: center; justify-content: center;
  border-bottom: 1px solid var(--line); font-size: 10.5px; color: var(--dim);
  font-variant-numeric: tabular-nums; user-select: none;
}
.ws-rowhead.is-selected { background: color-mix(in srgb, var(--accent) 18%, transparent); color: var(--accent-text); }
.ws-row-resize { position: absolute; bottom: -3px; left: 0; width: 100%; height: 6px; cursor: row-resize; z-index: 2; }
.ws-row-resize:hover { background: color-mix(in srgb, var(--accent) 45%, transparent); }
.ws-grid-cells { position: absolute; top: 0; left: 0; }
.ws-grid-shell.no-gridlines .ws-cell { border-right-color: transparent; border-bottom-color: transparent; }

.ws-cell {
  position: absolute; display: flex; align-items: center; padding: 0 5px; overflow: hidden;
  white-space: nowrap; border-right: 1px solid var(--line-soft); border-bottom: 1px solid var(--line-soft);
  font-size: 12px; color: var(--text); cursor: cell; user-select: none;
  background: var(--surface-1);
}
.ws-cell.is-alt { background: var(--surface-2); }
.ws-cell.is-header { background: color-mix(in srgb, var(--accent) 6%, var(--surface-2)); font-weight: 600; }
.ws-cell.is-selected { background: color-mix(in srgb, var(--accent) 16%, var(--surface-1)); }
.ws-cell.is-active { box-shadow: inset 0 0 0 2px var(--accent); z-index: 3; }
.ws-cell.is-match { background: color-mix(in srgb, var(--amber) 18%, var(--surface-1)); }
.ws-cell.is-formula { color: var(--accent-text); }
.ws-cell.align-center { justify-content: center; }
.ws-cell.align-right { justify-content: flex-end; }
.ws-cell .ws-cell-text { overflow: hidden; text-overflow: ellipsis; white-space: pre; }
.ws-cell.is-wrap .ws-cell-text { white-space: pre-wrap; overflow-wrap: anywhere; }
.ws-cell.bold .ws-cell-text { font-weight: 700; }
.ws-cell.italic .ws-cell-text { font-style: italic; }
.ws-cell.underline .ws-cell-text { text-decoration: underline; }
.ws-cell.strike .ws-cell-text { text-decoration: line-through; }
.ws-cell.err { color: var(--bad-text); }
.ws-cell-editor { position: absolute; z-index: 20; }
.ws-cell-editor input {
  width: 100%; height: 100%; padding: 0 4px; border: 2px solid var(--accent);
  background: var(--surface-1); color: var(--text); font-size: 12px; outline: none;
  box-shadow: 0 10px 24px -14px var(--glow-strong);
}

.ws-empty {
  position: absolute; inset: 0; display: grid; place-items: center; z-index: 8;
  background: var(--backdrop-grad); overflow: auto; padding: 24px;
}
.ws-empty[hidden] { display: none; }
.ws-empty-card {
  max-width: 620px; width: 100%; border: 1px solid var(--line); border-radius: 18px;
  padding: 30px; background: var(--welcome-grad); box-shadow: var(--shadow);
}
.ws-empty-mark { width: 52px; height: 52px; border-radius: 14px; display: grid; place-items: center; background: var(--button-grad); color: #fff; margin-bottom: 16px; }
.ws-empty-mark svg { width: 28px; height: 28px; }
.ws-empty-card h2 { margin: 0 0 8px; font-size: 24px; font-family: "Manrope", sans-serif; }
.ws-empty-card p { margin: 0 0 18px; color: var(--muted); font-size: 13.5px; line-height: 1.6; }
.ws-empty-actions { display: flex; gap: 8px; flex-wrap: wrap; }
.ws-drop-hint { margin-top: 16px; padding: 12px; border: 1px dashed var(--line-strong); border-radius: 10px; color: var(--dim); font-size: 12px; text-align: center; }
.ws-recent-mini { margin-top: 18px; display: flex; flex-direction: column; gap: 4px; }
.ws-recent-mini button { display: flex; justify-content: space-between; gap: 10px; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line); background: var(--surface-2); font-size: 12px; text-align: left; }
.ws-recent-mini button small { color: var(--dim); }
.ws-drop-overlay { position: absolute; inset: 12px; border: 2px dashed var(--accent); border-radius: 16px; display: grid; place-items: center; background: color-mix(in srgb, var(--accent) 12%, rgba(0,0,0,.55)); z-index: 30; font-size: 16px; font-weight: 600; backdrop-filter: blur(4px); }

/* --------------------------------------------------------------- statusbar */

.ws-statusbar {
  display: flex; align-items: center; gap: 10px; padding: 0 12px; height: 38px;
  border-top: 1px solid var(--line); background: var(--surface-1); font-size: 11.5px; color: var(--muted);
}
.ws-sheet-tabs { display: flex; gap: 3px; overflow-x: auto; max-width: 55%; scrollbar-width: thin; }
.ws-sheet-tab {
  border: 1px solid transparent; border-bottom: 0; background: transparent; color: var(--muted);
  padding: 7px 12px; border-radius: 8px 8px 0 0; font-size: 11.5px; white-space: nowrap;
}
.ws-sheet-tab:hover { background: var(--hover-bg); }
.ws-sheet-tab.is-active { background: var(--surface-3); color: var(--text); border-color: var(--line); font-weight: 600; }
.ws-sheet-add { border: 1px solid var(--line); background: transparent; color: var(--muted); width: 24px; height: 24px; border-radius: 6px; line-height: 1; }
.ws-status-spacer { flex: 1; }
.ws-statusbar span { white-space: nowrap; }
#ws-status-storage { max-width: 240px; overflow: hidden; text-overflow: ellipsis; color: var(--dim); }

/* ------------------------------------------------------------ analysis panel */

.ws-analysis-panel {
  border-top: 1px solid var(--line); background: var(--surface-2); max-height: 260px;
  display: flex; flex-direction: column;
}
.ws-analysis-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 8px 12px; border-bottom: 1px solid var(--line); }
.ws-analysis-head strong { font-size: 12.5px; }
.ws-analysis-head span { display: block; color: var(--dim); font-size: 10.5px; margin-top: 1px; }
.ws-analysis-head > div:last-child { display: flex; gap: 6px; }
.ws-analysis-list { overflow-y: auto; padding: 6px 10px 10px; display: flex; flex-direction: column; gap: 4px; }
.ws-issue-row {
  display: flex; align-items: center; gap: 8px; padding: 7px 9px; border-radius: 8px;
  border: 1px solid var(--line); background: var(--surface-1); font-size: 11.5px;
}
.ws-issue-row .ws-issue-ref { font-weight: 600; color: var(--accent-text); min-width: 58px; }
.ws-issue-row .ws-issue-body { flex: 1; min-width: 0; }
.ws-issue-row .ws-issue-body strong { display: block; font-weight: 600; }
.ws-issue-row .ws-issue-body small { color: var(--dim); display: block; overflow: hidden; text-overflow: ellipsis; }
.ws-issue-row .ws-issue-actions { display: flex; gap: 4px; }
.ws-issue-row .ws-small-btn { padding: 4px 8px; font-size: 10.5px; }
.ws-issue-empty { padding: 14px; color: var(--dim); font-size: 12px; text-align: center; }
.ws-severity { width: 7px; height: 7px; border-radius: 50%; background: var(--amber); flex: none; }
.ws-severity.error { background: var(--rose); }
.ws-severity.warning { background: var(--amber); }
.ws-severity.info { background: var(--accent); }

/* ------------------------------------------------------------------ modals */

.ws-modal {
  position: fixed; inset: 0; z-index: 80; display: grid; place-items: center;
  background: var(--overlay-bg); backdrop-filter: blur(6px); padding: 20px;
}
.ws-modal[hidden] { display: none; }
.ws-modal-card {
  width: min(460px, 100%); max-height: calc(100vh - 48px); overflow-y: auto;
  background: var(--overlay-card); border: 1px solid var(--line-strong);
  border-radius: 16px; padding: 22px; box-shadow: var(--shadow-strong);
}
.ws-modal-card.wide { width: min(760px, 100%); }
.ws-modal-card h2 { margin: 0 0 6px; font-size: 18px; font-family: "Manrope", sans-serif; }
.ws-modal-card > p { margin: 0 0 14px; color: var(--muted); font-size: 12.5px; line-height: 1.55; }
.ws-subhead { margin: 18px 0 6px; font-size: 13px; }
.ws-field { display: flex; flex-direction: column; gap: 5px; font-size: 11.5px; color: var(--muted); margin-bottom: 12px; }
.ws-field input, .ws-field select { padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line); background: var(--input-bg); font-size: 12.5px; }
.ws-field input:focus, .ws-field select:focus { outline: none; border-color: var(--accent); }
.ws-path-row { display: flex; gap: 6px; }
.ws-path-row input { flex: 1; min-width: 0; }
.ws-modal-actions { display: flex; justify-content: flex-end; align-items: center; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
.ws-check-row { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11.5px; color: var(--muted); margin-bottom: 12px; }
.ws-check-row label { display: inline-flex; gap: 5px; align-items: center; }

.ws-progress { height: 8px; border-radius: 999px; background: var(--track); overflow: hidden; margin-bottom: 12px; }
.ws-progress-bar { height: 100%; width: 0; background: var(--bar-grad); transition: width .15s linear; }
.ws-upload-list { display: flex; flex-direction: column; gap: 6px; max-height: 240px; overflow-y: auto; font-size: 12px; }
.ws-upload-row { display: flex; justify-content: space-between; gap: 8px; padding: 7px 9px; border: 1px solid var(--line); border-radius: 7px; background: var(--surface-2); }
.ws-upload-row small { color: var(--dim); }
.ws-upload-row.is-ok small { color: var(--ok-text); }
.ws-upload-row.is-bad small { color: var(--bad-text); }

.ws-export-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.ws-format-note { padding: 10px 12px; border-radius: 8px; border: 1px solid var(--line); background: var(--surface-2); color: var(--muted); font-size: 11.5px; line-height: 1.5; margin: 0 0 10px; }
.ws-export-history { max-height: 190px; overflow-y: auto; display: flex; flex-direction: column; gap: 4px; }
.ws-export-row { display: flex; align-items: center; gap: 8px; padding: 7px 9px; border: 1px solid var(--line); border-radius: 7px; font-size: 11.5px; }
.ws-export-row span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ws-export-row small { color: var(--dim); }

.ws-version-actions { display: flex; gap: 6px; margin-bottom: 12px; }
.ws-version-actions input { flex: 1; padding: 7px 9px; border-radius: 7px; border: 1px solid var(--line); background: var(--input-bg); font-size: 12px; }
.ws-version-list, .ws-batch-list { display: flex; flex-direction: column; gap: 5px; max-height: 220px; overflow-y: auto; }
.ws-version-row, .ws-batch-row { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface-2); font-size: 11.5px; }
.ws-version-row div, .ws-batch-row div { flex: 1; min-width: 0; }
.ws-version-row strong { display: block; font-size: 12px; }
.ws-version-row small, .ws-batch-row small { color: var(--dim); }
.ws-batch-row.is-undone { opacity: .55; }
.ws-find-status { flex: 1; font-size: 11.5px; color: var(--dim); }

.ws-mode-row { display: flex; gap: 6px; margin-bottom: 14px; }
.ws-mode-row button { flex: 1; padding: 8px; border-radius: 8px; border: 1px solid var(--line); background: var(--surface-2); font-size: 12px; }
.ws-mode-row button.is-active { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 14%, transparent); }
.ws-theme-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; max-height: 320px; overflow-y: auto; }
.ws-theme-card { padding: 8px; border-radius: 10px; border: 1px solid var(--line); background: var(--surface-2); text-align: left; font-size: 11px; }
.ws-theme-card.is-active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
.ws-theme-swatches { display: flex; gap: 4px; margin-bottom: 6px; }
.ws-theme-swatches span { width: 18px; height: 18px; border-radius: 5px; border: 1px solid rgba(255,255,255,.15); }
.ws-theme-card strong { display: block; font-size: 11.5px; }

.ws-search-results { display: flex; flex-direction: column; gap: 4px; max-height: 380px; overflow-y: auto; }
.ws-search-row { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface-2); font-size: 11.5px; text-align: left; }
.ws-search-row:hover { border-color: var(--accent); }
.ws-search-row .ws-search-ref { font-weight: 600; color: var(--accent-text); min-width: 44px; }
.ws-search-row div { flex: 1; min-width: 0; }
.ws-search-row strong { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ws-search-row small { color: var(--dim); }
.ws-search-group { font-size: 10.5px; text-transform: uppercase; letter-spacing: .6px; color: var(--dim); margin: 10px 0 3px; }
.ws-storage-info { display: flex; flex-direction: column; gap: 4px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface-2); font-size: 11.5px; margin-bottom: 14px; word-break: break-all; }
.ws-storage-info strong { color: var(--text); }
.ws-storage-info span { color: var(--dim); }

.ws-toasts { position: fixed; right: 16px; bottom: 54px; z-index: 120; display: flex; flex-direction: column; gap: 8px; max-width: 380px; }
.ws-toast {
  padding: 10px 14px; border-radius: 10px; border: 1px solid var(--line-strong);
  background: var(--toast-bg); color: var(--text); font-size: 12.5px; box-shadow: var(--shadow);
  animation: ws-toast-in .18s ease;
}
.ws-toast.ok { border-color: color-mix(in srgb, var(--green) 45%, var(--line)); }
.ws-toast.error { border-color: color-mix(in srgb, var(--rose) 55%, var(--line)); }
@keyframes ws-toast-in { from { transform: translateY(6px); opacity: 0; } to { transform: none; opacity: 1; } }

.ws-upload-dock { position: fixed; left: 50%; bottom: 52px; transform: translateX(-50%); z-index: 100; padding: 10px 14px; border-radius: 10px; background: var(--toast-bg); border: 1px solid var(--line-strong); font-size: 12px; box-shadow: var(--shadow); }

/* --------------------------------------------------------------- responsive */

@media (max-width: 1100px) {
  .wx-title-block, .ws-title-block { max-width: 180px; }
  .ws-global-search { max-width: none; }
  .ws-statusbar #ws-status-storage { display: none; }
}

@media (max-width: 820px) {
  .ws-app { grid-template-columns: 1fr; }
  .ws-sidebar {
    position: fixed; top: 0; bottom: 0; left: 0; width: 274px; z-index: 50;
    transform: translateX(-100%);
  }
  .ws-app.sidebar-open .ws-sidebar { transform: none; }
  .ws-app.sidebar-collapsed { grid-template-columns: 1fr; }
  .ws-app.sidebar-collapsed .ws-sidebar { transform: translateX(-100%); }
  .ws-mobile-menu { display: inline-grid; }
  .ws-title-block { max-width: 140px; }
  .ws-global-search { margin: 0; }
  .ws-topbar { flex-wrap: wrap; }
  .ws-global-search { order: 4; flex-basis: 100%; max-width: none; }
  .ws-top-actions { margin-left: auto; }
  .ws-ribbon-body { padding-bottom: 8px; }
  .ws-analysis-panel { max-height: 40vh; }
  .ws-sheet-tabs { max-width: 70%; }
  .ws-export-grid { grid-template-columns: 1fr; }
  .ws-toasts { left: 12px; right: 12px; max-width: none; }
}

/* -------------------------------------------------------- section pages */

.ws-page-nav { display: flex; flex-direction: column; gap: 2px; padding-bottom: 8px; margin-bottom: 4px; border-bottom: 1px solid var(--line); }
.ws-page-item {
  display: flex; align-items: center; gap: 10px; width: 100%; padding: 8px 10px;
  border: 1px solid transparent; border-radius: 9px; background: transparent;
  color: var(--muted); font-size: 12.8px; text-align: left; transition: background .12s ease;
}
.ws-page-item svg { width: 16px; height: 16px; flex: none; opacity: .85; }
.ws-page-item span { flex: 1; }
.ws-page-item:hover { background: var(--hover-bg); color: var(--text); }
.ws-page-item.is-active { background: color-mix(in srgb, var(--accent) 13%, transparent); color: var(--text); border-color: color-mix(in srgb, var(--accent) 26%, transparent); }
.ws-page-item.is-active svg { color: var(--accent); opacity: 1; }

.ws-pages { flex: 1; min-height: 0; overflow-y: auto; padding: 26px clamp(16px, 4vw, 44px) 44px; }
.ws-pages-inner { max-width: 1180px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
.ws-main.is-page .ws-ribbon,
.ws-main.is-page .ws-formula-bar,
.ws-main.is-page .ws-filter-panel,
.ws-main.is-page .ws-sheet-area,
.ws-main.is-page .ws-analysis-panel,
.ws-main.is-page .ws-statusbar { display: none; }
.ws-main.is-page .ws-top-actions > * { display: none; }
.ws-main.is-page .ws-title-block { max-width: none; }

.ws-page-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.ws-page-head h2 { margin: 0; font-family: "Manrope", sans-serif; font-size: 24px; letter-spacing: -.2px; }
.ws-page-head p { margin: 5px 0 0; color: var(--dim); font-size: 12.5px; max-width: 760px; line-height: 1.55; }
.ws-page-actions { display: flex; gap: 8px; flex-wrap: wrap; }

.ws-hero {
  display: flex; align-items: center; justify-content: space-between; gap: 24px; flex-wrap: wrap;
  padding: 24px; border-radius: 18px; border: 1px solid var(--line);
  background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 14%, var(--panel)), var(--panel));
}
.ws-hero h2 { margin: 0 0 6px; font-family: "Manrope", sans-serif; font-size: 26px; letter-spacing: -.3px; }
.ws-hero p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.6; max-width: 560px; }
.ws-hero-actions { display: flex; gap: 8px; flex-wrap: wrap; }

.ws-stat-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.ws-stat-card {
  padding: 15px 16px; border-radius: 14px; border: 1px solid var(--line);
  background: var(--panel); display: flex; flex-direction: column; gap: 4px;
}
.ws-stat-card span { font-size: 10.6px; text-transform: uppercase; letter-spacing: .7px; color: var(--dim); font-weight: 600; }
.ws-stat-card strong { font-family: "Manrope", sans-serif; font-size: 27px; line-height: 1.15; }
.ws-stat-card small { color: var(--dim); font-size: 10.5px; }
.ws-stat-card.accent strong { color: var(--accent); }
.ws-stat-card.warn strong { color: var(--amber); }
.ws-stat-card.ok strong { color: var(--ok); }

.ws-card { border-radius: 14px; border: 1px solid var(--line); background: var(--panel); overflow: hidden; }
.ws-card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; padding: 13px 16px; border-bottom: 1px solid var(--line); }
.ws-card-head strong { font-size: 13px; }
.ws-card-head span { color: var(--dim); font-size: 11px; }
.ws-card-head > div > span { display: block; margin-top: 3px; }
.ws-card-body { padding: 14px 16px; }
.ws-card-body.flush { padding: 0; }

.ws-table-wrap { width: 100%; overflow-x: auto; }
.ws-table tbody tr[data-file-menu] { cursor: pointer; }
.ws-table tbody tr[data-file-menu]:hover { background: var(--hover-bg); }
.ws-table { width: 100%; border-collapse: collapse; font-size: 12.4px; }
.ws-table th {
  text-align: left; padding: 9px 12px; font-size: 10.4px; letter-spacing: .6px; text-transform: uppercase;
  color: var(--dim); font-weight: 600; border-bottom: 1px solid var(--line); background: var(--surface-2); white-space: nowrap;
}
.ws-table td { padding: 10px 12px; border-bottom: 1px solid var(--line); vertical-align: middle; }
.ws-table tr:last-child td { border-bottom: 0; }
.ws-table tbody tr:hover { background: var(--hover-bg); }
.ws-table .ws-name-cell { font-weight: 600; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ws-table .ws-num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ws-table .ws-actions-cell { white-space: nowrap; text-align: right; }
.ws-table .ws-actions-cell button { margin-left: 4px; }

.ws-pill { display: inline-flex; align-items: center; gap: 5px; padding: 3px 8px; border-radius: 999px; font-size: 10.5px; font-weight: 600; border: 1px solid var(--line); color: var(--muted); background: var(--surface-2); white-space: nowrap; }
.ws-pill.ok { color: var(--ok); border-color: color-mix(in srgb, var(--ok) 40%, var(--line)); background: color-mix(in srgb, var(--ok) 10%, transparent); }
.ws-pill.warn { color: var(--amber); border-color: color-mix(in srgb, var(--amber) 40%, var(--line)); background: color-mix(in srgb, var(--amber) 10%, transparent); }
.ws-pill.bad { color: var(--danger); border-color: color-mix(in srgb, var(--danger) 40%, var(--line)); background: color-mix(in srgb, var(--danger) 10%, transparent); }
.ws-pill.info { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 40%, var(--line)); background: color-mix(in srgb, var(--accent) 10%, transparent); }

.ws-home-search { max-width: 620px; margin: 0; }
.ws-muted { color: var(--dim); font-size: 12px; line-height: 1.6; margin: 6px 0 0; }
.ws-empty-box { padding: 34px 20px; text-align: center; color: var(--dim); font-size: 12.5px; line-height: 1.6; }
.ws-empty-box strong { display: block; color: var(--text); font-size: 14px; margin-bottom: 4px; }

.ws-dropzone {
  border: 2px dashed color-mix(in srgb, var(--accent) 45%, var(--line)); border-radius: 18px;
  padding: 42px 24px; text-align: center; background: color-mix(in srgb, var(--accent) 5%, var(--panel));
  display: flex; flex-direction: column; align-items: center; gap: 10px; cursor: pointer;
}
.ws-dropzone:hover { background: color-mix(in srgb, var(--accent) 9%, var(--panel)); }
.ws-dropzone svg { width: 38px; height: 38px; color: var(--accent); }
.ws-dropzone strong { font-size: 16px; font-family: "Manrope", sans-serif; }
.ws-dropzone p { margin: 0; color: var(--dim); font-size: 12px; max-width: 520px; line-height: 1.6; }
.ws-upload-info-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }

.ws-toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 10px 14px; border-bottom: 1px solid var(--line); background: var(--surface-2); }
.ws-toolbar .ws-spacer { flex: 1; }
.ws-toolbar select, .ws-toolbar input { padding: 6px 9px; border-radius: 8px; border: 1px solid var(--line); background: var(--input-bg); color: var(--text); font-size: 12px; }
.ws-filter-tabs { display: flex; gap: 4px; flex-wrap: wrap; }
.ws-filter-tabs button { padding: 5px 10px; border-radius: 999px; border: 1px solid var(--line); background: transparent; color: var(--muted); font-size: 11px; }
.ws-filter-tabs button.is-active { background: color-mix(in srgb, var(--accent) 15%, transparent); border-color: color-mix(in srgb, var(--accent) 35%, var(--line)); color: var(--text); }
.ws-filter-tabs button em { font-style: normal; color: var(--dim); margin-left: 4px; }

.ws-summary-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
.ws-summary-strip div { border: 1px solid var(--line); border-radius: 10px; padding: 9px 11px; background: var(--surface-1); }
.ws-summary-strip span { display: block; font-size: 10px; text-transform: uppercase; letter-spacing: .6px; color: var(--dim); }
.ws-summary-strip strong { font-size: 18px; font-family: "Manrope", sans-serif; }

.ws-cat-row { display: grid; grid-template-columns: minmax(0, 1fr) 90px 110px; gap: 10px; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--line); font-size: 12.5px; }
.ws-cat-row:last-child { border-bottom: 0; }
.ws-cat-row strong { font-weight: 600; }
.ws-cat-row .ws-num { text-align: right; font-variant-numeric: tabular-nums; }
.ws-cat-row button { justify-self: end; }

.ws-problem-table td { vertical-align: top; }
.ws-problem-table .ws-original, .ws-problem-table .ws-suggestion { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11.5px; max-width: 220px; overflow-wrap: anywhere; }
.ws-problem-table tr.is-applied { background: color-mix(in srgb, var(--ok) 7%, transparent); }
.ws-problem-table tr.is-rejected { opacity: .6; }
.ws-problem-table input[type="checkbox"] { accent-color: var(--accent); }
.ws-confidence { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; }
.ws-confidence i { display: inline-block; width: 34px; height: 5px; border-radius: 999px; background: var(--track); overflow: hidden; }
.ws-confidence i b { display: block; height: 100%; background: var(--accent); }
.ws-reason { color: var(--dim); font-size: 10.5px; display: block; margin-top: 3px; max-width: 240px; }

.ws-report-cats { display: flex; gap: 4px; flex-wrap: wrap; }
.ws-report-cats .ws-pill { font-weight: 500; }

@media (max-width: 1080px) {
  .ws-stat-grid, .ws-summary-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ws-upload-info-grid { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
  .ws-pages { padding: 18px 12px 32px; }
  .ws-hero { padding: 18px; }
  .ws-page-head h2 { font-size: 20px; }
  .ws-cat-row { grid-template-columns: minmax(0, 1fr) 70px auto; }
}

@media (max-width: 520px) {
  .ws-save-chip { display: none; }
  .ws-top-actions .ws-icon-btn { display: none; }
  .ws-empty-card { padding: 20px; }
  .ws-empty-card h2 { font-size: 20px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .001ms !important; transition-duration: .001ms !important; }
}

==============================================================================
FILE: static/workspace.js
==============================================================================
/* DataFix Workspace — permanent workbook library + virtualized spreadsheet client. */
(function () {
  "use strict";

  var API = "/api/workspace";
  var EXTENSIONS = [".xlsx", ".xlsm", ".xls", ".csv", ".ods"];
  var MODAL_IDS = [
    "ws-modal-new", "ws-modal-upload", "ws-modal-export", "ws-modal-versions",
    "ws-modal-find", "ws-modal-goto", "ws-modal-import", "ws-modal-storage",
    "ws-modal-appearance", "ws-modal-search", "ws-modal-confirm",
  ];
  var HEAD_W = 58, HEAD_H = 30;

  function $(id) { return document.getElementById(id); }
  function qs(selector, root) { return (root || document).querySelector(selector); }
  function qsa(selector, root) { return Array.prototype.slice.call((root || document).querySelectorAll(selector)); }
  function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (char) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char];
    });
  }
  function letter(col) {
    var name = "", value = Number(col) || 0;
    while (value > 0) { var rem = (value - 1) % 26; name = String.fromCharCode(65 + rem) + name; value = Math.floor((value - 1) / 26); }
    return name || "A";
  }
  function columnNumber(text) {
    var value = 0;
    String(text || "").toUpperCase().split("").forEach(function (char) {
      if (char >= "A" && char <= "Z") value = value * 26 + char.charCodeAt(0) - 64;
    });
    return value;
  }
  function debounce(fn, delay) {
    var timer = null;
    return function () {
      var args = arguments, context = this;
      clearTimeout(timer);
      timer = setTimeout(function () { fn.apply(context, args); }, delay);
    };
  }
  function when(iso) {
    if (!iso) return "";
    var date = new Date(iso);
    if (isNaN(date.getTime())) return String(iso);
    return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }
  function clock(date) {
    var stamp = date instanceof Date ? date : new Date();
    return stamp.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }
  function bytes(size) {
    var value = Number(size) || 0;
    if (value < 1024) return value + " B";
    if (value < 1024 * 1024) return (value / 1024).toFixed(1) + " KB";
    return (value / 1024 / 1024).toFixed(1) + " MB";
  }
  function safeColor(value) {
    var text = String(value || "");
    return /^#[0-9a-f]{3,8}$|^rgba?\([\d\s.,%]+\)$|^[a-z]+$/i.test(text) ? text : "";
  }
  function safeFont(value) {
    return String(value || "").replace(/[^\w \-]/g, "").slice(0, 60);
  }
  function plural(count, one, many) { return count === 1 ? one : (many || one + "s"); }

  var S = {
    storage: null, limits: {}, formats: {}, maxMb: 0,
    workbooks: [], archived: [], counts: { all: 0, starred: 0, archived: 0 },
    fileQuery: "", fileResults: null, scope: "all",
    wb: null, sheetId: null, sheets: [], sheetMeta: null,
    rows: [], columns: [], cells: new Map(), headers: {},
    totalRows: 0, rowCount: 0, colCount: 0,
    frozenRows: 0, frozenCols: 0,
    filters: [], filterDraft: [], sort: null, search: "",
    active: { r: 1, c: 1 }, sel: { r1: 1, c1: 1, r2: 1, c2: 1 }, anchor: { r: 1, c: 1 },
    highlight: null, showFormulas: false, zoom: 1, gridlines: true,
    colWidths: new Map(), rowHeights: new Map(),
    queue: [], queueLabel: "Edit", pending: new Map(), save: { error: "", state: "idle" },
    lastSaved: null, canUndo: false, canRedo: false,
    editing: false, dragging: false,
    analysis: null, analysisStale: false, issueIndex: -1,
    findState: null, findKey: "",
    gridSerial: 0, loadingRows: false, loadingCols: false,
    clipboard: null, versions: null,
    themeDraft: null, openToken: 0, searchSeq: 0,
    page: "home", dashboard: null, reports: null, filesTab: "active",
    reportWbId: null, reportAnalysis: null, reportCategory: "all",
    reportStatus: "open", reportSelected: {},
  };

  var flushing = null;
  var flushTimer = null;
  var retryTimer = null;
  var refreshTimer = null;
  var nameRequest = null;
  var confirmRequest = null;
  var confirmMode = "confirm";
  var activeXhr = null;
  var dragDepth = 0;
  var renderQueued = false;

  /* ------------------------------------------------------------------ API */

  async function api(path, options) {
    options = options || {};
    var init = { method: options.method || (options.body !== undefined ? "POST" : "GET"), headers: { Accept: "application/json" } };
    if (options.signal) init.signal = options.signal;
    if (options.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    var response;
    try {
      response = await fetch(API + path, init);
    } catch (error) {
      throw new Error("Cannot reach the DataFix backend. Start the Python server, then retry.");
    }
    var data = {};
    try { data = await response.json(); } catch (error) { data = {}; }
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || ("Request failed (" + response.status + ")."));
    }
    return data;
  }

  function toast(message, kind, ms) {
    var host = $("ws-toasts");
    if (!host) return;
    var node = document.createElement("div");
    node.className = "ws-toast " + (kind === "error" ? "error" : kind === "ok" ? "ok" : "");
    node.textContent = String(message || "");
    host.appendChild(node);
    setTimeout(function () {
      node.style.opacity = "0";
      setTimeout(function () { if (node.parentNode) node.parentNode.removeChild(node); }, 220);
    }, ms || 4200);
  }

  function reportError(error) {
    console.error(error);
    toast((error && error.message) || "That action could not be completed.", "error");
  }

  /* --------------------------------------------------------------- modals */

  function openModal(id) {
    var modal = $(id);
    if (!modal) return;
    modal.hidden = false;
    var focus = qs("input:not([type=hidden]), select, button", modal);
    if (focus) setTimeout(function () { try { focus.focus(); } catch (error) { /* ignore */ } }, 30);
  }

  function closeModal(id) {
    var modal = $(id);
    if (!modal || modal.hidden) return;
    if (id === "ws-modal-upload" && activeXhr) return;
    modal.hidden = true;
    if (id === "ws-modal-new" && nameRequest) {
      var resolveName = nameRequest;
      nameRequest = null;
      resolveName(null);
    }
    if (id === "ws-modal-confirm" && confirmRequest) {
      var resolveConfirm = confirmRequest;
      confirmRequest = null;
      resolveConfirm(confirmMode === "info" ? true : false);
    }
  }

  function anyModalOpen() { return MODAL_IDS.some(function (id) { var node = $(id); return node && !node.hidden; }); }
  function lastOpenModal() {
    for (var i = MODAL_IDS.length - 1; i >= 0; i -= 1) {
      var node = $(MODAL_IDS[i]);
      if (node && !node.hidden) return MODAL_IDS[i];
    }
    return "";
  }

  function askName(options) {
    var modal = $("ws-modal-new");
    $("ws-new-title").textContent = options.title || "Name";
    qs("p", modal).textContent = options.text || "";
    $("ws-new-name").value = options.value || "";
    $("ws-new-create").textContent = options.button || "Continue";
    openModal("ws-modal-new");
    setTimeout(function () {
      var input = $("ws-new-name");
      input.focus();
      input.select();
    }, 40);
    return new Promise(function (resolve) { nameRequest = resolve; });
  }

  function finishName(value) {
    var resolve = nameRequest;
    nameRequest = null;
    $("ws-modal-new").hidden = true;
    if (resolve) resolve(value);
  }

  function confirmDialog(options) {
    $("ws-confirm-title").textContent = options.title || "Are you sure?";
    $("ws-confirm-text").textContent = options.text || "";
    var ok = $("ws-confirm-ok");
    ok.textContent = options.button || "Continue";
    ok.classList.toggle("danger", options.danger !== false);
    $("ws-confirm-cancel").hidden = false;
    confirmMode = "confirm";
    openModal("ws-modal-confirm");
    return new Promise(function (resolve) { confirmRequest = resolve; });
  }

  function showInfo(title, text) {
    $("ws-confirm-title").textContent = title;
    $("ws-confirm-text").textContent = text;
    var ok = $("ws-confirm-ok");
    ok.textContent = "Got it";
    ok.classList.remove("danger");
    $("ws-confirm-cancel").hidden = true;
    confirmMode = "info";
    confirmRequest = null;
    openModal("ws-modal-confirm");
  }

  function settleConfirm(value) {
    var resolve = confirmRequest;
    confirmRequest = null;
    $("ws-modal-confirm").hidden = true;
    if (resolve) resolve(value);
  }

  /* ---------------------------------------------------------- save status */

  function updateSaveChip() {
    var chip = $("ws-save-status");
    if (!chip) return;
    S.save.pending = S.queue.length;
    if (!S.wb) {
      chip.dataset.state = "idle";
      chip.textContent = "Ready";
      chip.title = "Open or create a workbook to start editing.";
      return;
    }
    if (S.save.error) {
      chip.dataset.state = "error";
      chip.textContent = "Not saved — retry";
      chip.title = S.save.error + (S.queue.length ? " · " + S.queue.length + " change(s) waiting" : "") + " · Click to retry.";
      return;
    }
    if (flushing || S.queue.length) {
      chip.dataset.state = "saving";
      chip.textContent = "Saving…";
      chip.title = "Saving to the Python backend. " + S.queue.length + " change(s) queued.";
      return;
    }
    chip.dataset.state = "saved";
    chip.textContent = S.lastSaved ? "Saved " + clock(S.lastSaved) : "Saved";
    chip.title = "All changes are written to the backend database and backup files.";
  }

  function scheduleFlush(delay) {
    clearTimeout(flushTimer);
    flushTimer = setTimeout(function () { flushSave().catch(reportError); }, delay == null ? 450 : delay);
  }

  function scheduleRetry() {
    clearTimeout(retryTimer);
    retryTimer = setTimeout(function () {
      if (S.queue.length && !flushing) { S.save.error = ""; flushSave().catch(reportError); }
    }, 8000);
  }

  function queueOps(ops, label) {
    if (!S.wb || !ops || !ops.length) return;
    var sheet = S.sheetId;
    ops.forEach(function (op) { if (op.sheet == null) op.sheet = sheet; S.queue.push(op); });
    S.queueLabel = label || "Edit";
    S.save.error = "";
    updateSaveChip();
    scheduleFlush();
  }

  function rememberPending(ops) {
    ops.forEach(function (op) {
      if (op.op === "set_cell") {
        var key = op.row + ":" + op.col;
        if (op.formula) S.pending.set(key, { f: op.formula, v: null, n: null });
        else if (op.value === "" || op.value == null) S.pending.set(key, { f: null, v: null, n: null });
        else S.pending.set(key, { f: null, v: op.value, n: null });
      }
    });
  }

  function flushSave() {
    if (flushing) return flushing;
    clearTimeout(flushTimer);
    clearTimeout(retryTimer);
    if (!S.wb || !S.queue.length) { updateSaveChip(); return Promise.resolve(); }
    var workbookId = S.wb.id;
    flushing = (async function () {
      try {
        while (S.wb && S.wb.id === workbookId && S.queue.length) {
          var chunk = S.queue.splice(0, 180);
          try {
            var result = await api("/workbooks/" + encodeURIComponent(workbookId) + "/ops", {
              method: "POST", body: { ops: chunk, label: S.queueLabel || "Edit" },
            });
            adoptWorkbook(result.workbook);
            setUndo(result.can_undo, result.can_redo);
            S.lastSaved = new Date();
            S.save.error = "";
            markAnalysisStale();
          } catch (error) {
            S.queue = chunk.concat(S.queue);
            S.save.error = error.message || "Autosave failed.";
            scheduleRetry();
            return;
          }
          updateSaveChip();
        }
        if (!S.queue.length) {
          S.pending.clear();
          scheduleRefresh();
        }
      } finally {
        flushing = null;
        updateSaveChip();
      }
    })();
    flushing.then(function () { updateSaveChip(); });
    return flushing;
  }

  function scheduleRefresh() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(function () {
      if (!S.wb || S.queue.length || flushing) return;
      reloadGrid({ keepScroll: true, silent: true }).catch(function (error) {
        console.warn("Grid refresh failed", error);
      });
    }, 260);
  }

  function setUndo(canUndo, canRedo) {
    S.canUndo = Boolean(canUndo);
    S.canRedo = Boolean(canRedo);
    $("ws-undo").disabled = !S.canUndo;
    $("ws-redo").disabled = !S.canRedo;
  }

  /* -------------------------------------------------------------- library */

  async function loadState() {
    var data = await api("/state");
    S.storage = data.storage;
    S.limits = data.limits || {};
    S.formats = data.formats || {};
    S.workbooks = data.workbooks || [];
    S.archived = data.archived || [];
    S.counts = {
      all: S.workbooks.length,
      starred: S.workbooks.filter(function (wb) { return wb.starred; }).length,
      archived: S.archived.length,
    };
    renderSidebar();
    renderEmptyRecent();
    populateExportFormats();
    updateStorageLabels();
  }

  async function refreshLibrary() {
    try {
      var data = await api("/state");
      S.storage = data.storage;
      S.limits = data.limits || S.limits;
      S.workbooks = data.workbooks || [];
      S.archived = data.archived || [];
      S.counts = {
        all: S.workbooks.length,
        starred: S.workbooks.filter(function (wb) { return wb.starred; }).length,
        archived: S.archived.length,
      };
      renderSidebar();
      renderEmptyRecent();
    } catch (error) { /* background refresh */ }
  }

  function scopeFiles() {
    if (S.fileResults) return S.fileResults;
    if (S.scope === "starred") return S.workbooks.filter(function (wb) { return wb.starred; });
    if (S.scope === "archived") return S.archived;
    if (S.scope === "recent") return S.workbooks.slice(0, 8);
    if (S.scope === "opened") return S.workbooks.slice(0, 20);
    return S.workbooks;
  }

  var STAR = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m12 4 2.4 4.9 5.4.8-3.9 3.8.9 5.4-4.8-2.5-4.8 2.5.9-5.4L4.2 9.7l5.4-.8z"/></svg>';
  var BOX = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="5" width="16" height="4" rx="1.5"/><path d="M6 9v8.5A1.5 1.5 0 0 0 7.5 19h9a1.5 1.5 0 0 0 1.5-1.5V9M10 13h4"/></svg>';
  var RESTORE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 12a8 8 0 1 0 2.3-5.6M4 4v4h4"/><path d="M12 8v4l2.5 1.5"/></svg>';

  function renderSidebar() {
    var label = { all: "All files", recent: "Recent", starred: "Starred", opened: "Recently opened", archived: "Archived" }[S.scope] || "All files";
    $("ws-files-label").textContent = label;
    $("ws-count-all").textContent = S.counts.all;
    $("ws-count-starred").textContent = S.counts.starred;
    $("ws-count-archived").textContent = S.counts.archived;
    qsa(".ws-nav-item[data-scope]").forEach(function (button) {
      button.classList.toggle("is-active", button.dataset.scope === S.scope);
    });
    var files = scopeFiles();
    var host = $("ws-file-list");
    if (!files.length) {
      host.innerHTML = '<div class="ws-file-empty">' + (S.fileQuery ? "No files match that search." : S.scope === "archived" ? "Nothing is archived." : "No workbooks yet. Create one or upload a spreadsheet.") + "</div>";
      return;
    }
    host.innerHTML = files.map(function (wb) {
      var active = S.wb && S.wb.id === wb.id;
      var archived = Boolean(wb.archived);
      return '<div class="ws-file-item' + (active ? " is-active" : "") + '" role="listitem" data-id="' + esc(wb.id) + '">' +
        '<button class="ws-file-main" type="button" data-open="' + esc(wb.id) + '" title="' + esc(wb.name) + '">' +
        "<strong>" + esc(wb.name) + "</strong>" +
        "<small>" + esc(wb.sheet_count || 0) + " " + plural(wb.sheet_count || 0, "sheet") + " · " + esc(when(wb.updated_at)) + "</small>" +
        "</button>" +
        '<div class="ws-file-actions">' +
        '<button class="ws-icon-btn' + (wb.starred ? " is-on" : "") + '" type="button" data-star="' + esc(wb.id) + '" title="' + (wb.starred ? "Remove star" : "Star this file") + '" aria-label="Star">' + STAR + "</button>" +
        '<button class="ws-icon-btn" type="button" data-archive="' + esc(wb.id) + '" title="' + (archived ? "Restore from archive" : "Archive this file") + '" aria-label="Archive">' + (archived ? RESTORE : BOX) + "</button>" +
        "</div></div>";
    }).join("");
  }

  function renderEmptyRecent() {
    var host = $("ws-recent-mini");
    if (!host) return;
    if (!S.workbooks.length) { host.innerHTML = ""; return; }
    host.innerHTML = '<div class="ws-search-group">Continue where you left off</div>' + S.workbooks.slice(0, 5).map(function (wb) {
      return '<button type="button" data-open="' + esc(wb.id) + '"><span>' + esc(wb.name) + "</span><small>" + esc(when(wb.updated_at)) + "</small></button>";
    }).join("");
  }

  /* ------------------------------------------------------------ sections */

  var CATEGORY_NAMES = {
    case: "Capitalisation", spacing: "Extra spaces", spelling: "Spelling mistakes",
    duplicate: "Duplicate records", missing: "Missing values", type: "Data types",
    format: "Formatting", sequence: "IDs and sequences", special: "Special characters",
    structure: "Structure",
  };

  function findWorkbook(id) {
    return S.workbooks.filter(function (wb) { return wb.id === id; })[0] ||
      S.archived.filter(function (wb) { return wb.id === id; })[0] || null;
  }

  function fileMenuButton(wb) {
    return '<button class="ws-small-btn" type="button" data-file-menu="' + esc(wb.id) +
      '" aria-haspopup="menu" aria-expanded="false">Options</button>';
  }

  function closeFileMenu() {
    var menu = $("ws-file-menu");
    if (menu) menu.remove();
    qsa("[data-file-menu]").forEach(function (button) {
      button.setAttribute("aria-expanded", "false");
    });
  }

  function openFileMenu(anchor, id) {
    var wb = findWorkbook(id);
    if (!wb) return;
    closeFileMenu();
    var archived = !!wb.archived;
    var entries = [
      ["open", "Open in Excel editor"],
      ["analyse", "Analyse again"],
      ["report", "View analysis report"],
      ["download", "Download original file"],
      ["rename", "Rename workbook"],
      [archived ? "restore" : "archive", archived ? "Restore from archive" : "Archive (hide, keep data)"],
      ["delete", "Delete permanently", "danger"],
    ];
    var menu = document.createElement("div");
    menu.id = "ws-file-menu";
    menu.className = "ws-file-menu";
    menu.setAttribute("role", "menu");
    menu.setAttribute("aria-label", "File options for " + wb.name);
    menu.innerHTML = '<div class="ws-file-menu-title">' + esc(wb.name) + "</div>" +
      entries.map(function (entry) {
        return '<button type="button" role="menuitem" class="' + (entry[2] || "") +
          '" data-file-action="' + entry[0] + '" data-id="' + esc(wb.id) + '">' + entry[1] + "</button>";
      }).join("");
    $("ws-pages").appendChild(menu);
    var anchorEl = anchor.tagName === "TR" ? (anchor.querySelector("[data-file-menu]") || anchor) : anchor;
    var rect = anchorEl.getBoundingClientRect();
    var width = 234;
    var left = Math.min(Math.max(8, rect.right - width), window.innerWidth - width - 8);
    var height = menu.offsetHeight || 280;
    var top = rect.bottom + 6;
    if (top + height > window.innerHeight - 8) top = Math.max(8, rect.top - height - 6);
    menu.style.left = left + "px";
    menu.style.top = top + "px";
    anchorEl.setAttribute("aria-expanded", "true");
    var first = menu.querySelector("button");
    if (first) setTimeout(function () {
      try { first.focus({ preventScroll: true }); } catch (error) { /* ignore */ }
    }, 20);
  }

  function pageHeader(title, text) {
    $("ws-title").textContent = "DataFix Workspace";
    $("ws-meta").textContent = "Permanent workspace · local files";
    $("ws-star-btn").hidden = true;
    $("ws-title").title = "";
    document.title = title + " — DataFix Workspace";
  }

  function setPageHeader(host, title, text, actions) {
    pageHeader(title, text);
    return '<div class="ws-pages-inner"><div class="ws-page-head"><div><h2>' + esc(title) + "</h2><p>" + esc(text) + "</p></div>" +
      '<div class="ws-page-actions">' + (actions || "") + "</div></div>";
  }

  function analysisPill(wb) {
    if (!wb.analysis_at) return '<span class="ws-pill warn">Not analysed</span>';
    if (wb.analysis_stale) return '<span class="ws-pill warn">Needs re-analysis</span>';
    return '<span class="ws-pill ok">Analysed</span>';
  }

  function typeLabel(wb) {
    return (wb.extension || ".xlsx").replace(/^\./, "").toUpperCase() || "XLSX";
  }

  function setPageHeading(name) {
    var headings = {
      home: ["Workspace home", "Upload, analyse and edit spreadsheets. Every file is stored by the Python backend."],
      upload: ["Upload dataset", "Import XLSX, XLSM, XLS, CSV or ODS files. Originals stay untouched; each import is analysed automatically."],
      files: ["My files", "Every uploaded and created workbook, with analysis status and file actions."],
      analysis: ["Data analysis", "Review detected problems, accept or reject corrections and open the sheet."],
      reports: ["Reports", "Data-quality summaries and correction reports for analysed workbooks."],
    };
    var entry = headings[name] || headings.home;
    pageHeader(entry[0], entry[1]);
  }

  async function setPage(name, options) {
    options = options || {};
    if (name === "editor" && !S.wb) name = "home";
    if (S.page === "editor" && name !== "editor") {
      await flushSave();
      if (S.queue.length) { toast("Save the pending changes before leaving the editor.", "error"); return; }
    }
    S.page = name;
    var main = qs(".ws-main");
    if (main) main.classList.toggle("is-page", name !== "editor");
    $("ws-pages").hidden = name === "editor";
    qsa(".ws-page-item").forEach(function (button) {
      button.classList.toggle("is-active", button.dataset.page === name);
    });
    if (name === "editor") { updateHeader(); updateHash(); return; }
    setPageHeading(name);
    try {
      if (name === "home") await renderHome();
      else if (name === "upload") await renderUpload();
      else if (name === "files") await renderFilesPage();
      else if (name === "analysis") await renderAnalysisPage(options.wbId);
      else if (name === "reports") await renderReports();
    } catch (error) {
      $("ws-pages").innerHTML = '<div class="ws-pages-inner"><div class="ws-card"><div class="ws-empty-box"><strong>This page could not load.</strong>' +
        esc(error.message || "Unknown error") + "</div></div></div>";
      reportError(error);
    }
  }

  function statCard(label, value, hint, tone) {
    return '<div class="ws-stat-card ' + (tone || "") + '"><span>' + esc(label) + "</span><strong>" +
      esc(value) + "</strong><small>" + esc(hint) + "</small></div>";
  }

  function homeRecentRows(files) {
    if (!files.length) {
      return '<tr><td colspan="6"><div class="ws-empty-box"><strong>No files yet</strong>Upload a spreadsheet or create a blank workbook to get started.</div></td></tr>';
    }
    return files.map(function (wb) {
      return '<tr data-file-menu="' + esc(wb.id) + '"><td class="ws-name-cell" title="' + esc(wb.name) + '">' + esc(wb.name) + "</td>" +
        "<td>" + esc(typeLabel(wb)) + "</td>" +
        "<td>" + esc(when(wb.updated_at)) + "</td>" +
        '<td class="ws-num">' + esc(wb.record_count || 0) + "</td>" +
        "<td>" + analysisPill(wb) + "</td>" +
        '<td class="ws-actions-cell">' + fileMenuButton(wb) + "</td></tr>";
    }).join("");
  }

  async function renderHome() {
    var host = $("ws-pages");
    host.innerHTML = '<div class="ws-pages-inner"><div class="ws-empty-box">Loading your workspace…</div></div>';
    var data = await api("/dashboard");
    S.dashboard = data;
    var totals = data.totals || {};
    var recent = data.recent || [];
    var needs = data.unanalysed || [];
    var header = setPageHeader(host, "Workspace home", "Upload, analyse and edit spreadsheets. Every file is stored by the Python backend.",
      '<button class="ws-small-btn" type="button" data-action="new-file">New Excel file</button>');
    var html = header +
      '<div class="ws-hero"><div><h2>Welcome to DataFix</h2><p>Upload a workbook to keep the original untouched, read every worksheet, detect data-quality problems and correct them with review. Open the Excel editor whenever you are ready.</p></div>' +
      '<div class="ws-hero-actions"><button class="ws-primary-btn" type="button" data-page-action="pick-files">Upload Excel file</button>' +
      '<button class="ws-small-btn" type="button" data-page-action="go-files">My files</button></div></div>' +
      '<label class="ws-file-search ws-home-search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4 4"/></svg>' +
      '<input type="search" id="ws-home-search" placeholder="Search saved workbooks by name" autocomplete="off" aria-label="Search saved workbooks"></label>' +
      '<div id="ws-home-results"></div>' +
      '<div class="ws-stat-grid">' +
      statCard("Total files", totals.files || 0, (totals.archived || 0) + " archived", "accent") +
      statCard("Total records", totals.records || 0, (totals.sheets || 0) + " worksheets", "") +
      statCard("Errors detected", totals.issues || 0, "Across stored analyses", totals.issues ? "warn" : "") +
      statCard("Errors corrected", totals.corrected || 0, (totals.rejected || 0) + " rejected", "ok") +
      "</div>" +
      '<div class="ws-card"><div class="ws-card-head"><strong>Recent files</strong><span>' + recent.length + " shown</span></div>" +
      '<div class="ws-table-wrap"><table class="ws-table"><thead><tr><th>File</th><th>Type</th><th>Last modified</th><th class="ws-num">Records</th><th>Analysis</th><th></th></tr></thead><tbody>' +
      homeRecentRows(recent) + "</tbody></table></div></div>";
    if (needs.length) {
      html += '<div class="ws-card"><div class="ws-card-head"><strong>Waiting for analysis</strong><span>' + needs.length + " file(s)</span></div><div class=\"ws-card-body\">" +
        needs.map(function (wb) {
          return '<div class="ws-cat-row"><strong>' + esc(wb.name) + "</strong><span>" + esc(wb.record_count || 0) + ' records</span>' +
            '<button class="ws-small-btn" type="button" data-file-action="analyse" data-id="' + esc(wb.id) + '">Analyse now</button></div>';
        }).join("") + "</div></div>";
    }
    host.innerHTML = html + "</div>";
  }

  function homeSearchRows(files, query) {
    if (!files.length) return '<div class="ws-empty-box">No saved workbook matches "' + esc(query) + '".</div>';
    return '<div class="ws-card"><div class="ws-table-wrap"><table class="ws-table"><thead><tr><th>File</th><th>Type</th><th>Modified</th><th></th></tr></thead><tbody>' +
      files.map(function (wb) {
        return '<tr data-file-menu="' + esc(wb.id) + '"><td class="ws-name-cell">' + esc(wb.name) + "</td><td>" + esc(typeLabel(wb)) + "</td><td>" + esc(when(wb.updated_at)) + "</td>" +
          '<td class="ws-actions-cell">' + fileMenuButton(wb) + "</td></tr>";
      }).join("") + "</tbody></table></div></div>";
  }

  async function renderUpload() {
    var host = $("ws-pages");
    await refreshLibrary();
    var recent = S.workbooks.slice(0, 5);
    setPageHeading("upload");
    var html = '<div class="ws-pages-inner"><div class="ws-page-head"><div><h2>Upload dataset</h2>' +
      "<p>Import XLSX, XLSM, XLS, CSV or ODS files. Originals stay untouched; each import is analysed automatically.</p></div>" +
      '<div class="ws-page-actions"><button class="ws-small-btn" type="button" data-page-action="open-import">Import from path…</button></div></div>' +
      '<div class="ws-dropzone" id="ws-dropzone" role="button" tabindex="0" data-page-action="pick-files">' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 16V5m0 0L7.5 9.5M12 5l4.5 4.5"/><path d="M5 15v3.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V15"/></svg>' +
      "<strong>Drop spreadsheets here</strong><p>or click to browse. Upload up to 20 files at once, up to " + esc(S.limits.max_upload_mb || 0) + " MB each. You can cancel while the upload is running.</p>" +
      '<button class="ws-primary-btn" type="button" data-page-action="pick-files">Browse files</button></div>' +
      '<div class="ws-upload-info-grid">' +
      '<div class="ws-card"><div class="ws-card-body"><strong>Supported formats</strong><p class="ws-muted">XLSX, XLSM, XLS, CSV and ODS. Every worksheet is imported and stored separately by workbook ID.</p></div></div>' +
      '<div class="ws-card"><div class="ws-card-body"><strong>After the upload</strong><p class="ws-muted">DataFix saves the original, reads the sheets, runs the local cleaning rules and opens the analysis report. Nothing is corrected without review.</p></div></div>' +
      '<div class="ws-card"><div class="ws-card-body"><strong>Your files stay local</strong><p class="ws-muted">The backend writes to permanent storage at ' + esc((S.storage && S.storage.root) || "the configured folder") + ".</p></div></div>" +
      "</div>" +
      '<div class="ws-card"><div class="ws-card-head"><strong>Recent imports</strong><button class="ws-small-btn ghost" type="button" data-page-action="go-files">All files</button></div>' +
      '<div class="ws-table-wrap"><table class="ws-table"><thead><tr><th>File</th><th>Type</th><th>Uploaded</th><th class="ws-num">Records</th><th>Analysis</th><th></th></tr></thead><tbody>' +
      (recent.length ? recent.map(function (wb) {
        return '<tr data-file-menu="' + esc(wb.id) + '"><td class="ws-name-cell">' + esc(wb.name) + "</td><td>" + esc(typeLabel(wb)) + "</td><td>" + esc(when(wb.created_at)) + "</td>" +
          '<td class="ws-num">' + esc(wb.record_count || 0) + "</td><td>" + analysisPill(wb) + "</td>" +
          '<td class="ws-actions-cell">' + fileMenuButton(wb) + "</td></tr>";
      }).join("") : '<tr><td colspan="6"><div class="ws-empty-box">No uploads yet.</div></td></tr>') +
      "</tbody></table></div></div></div>";
    host.innerHTML = html;
  }

  async function renderFilesPage() {
    await refreshLibrary();
    var archived = S.filesTab === "archived";
    var files = archived ? S.archived : S.workbooks;
    setPageHeading("files");
    var host = $("ws-pages");
    var rows = files.map(function (wb) {
      return '<tr data-file-menu="' + esc(wb.id) + '"><td class="ws-name-cell" title="' + esc(wb.name) + '">' + esc(wb.name) + "</td>" +
        "<td>" + esc(typeLabel(wb)) + "</td>" +
        "<td>" + esc(when(wb.created_at)) + "</td>" +
        "<td>" + esc(when(wb.updated_at)) + "</td>" +
        '<td class="ws-num">' + esc(wb.record_count || 0) + "</td>" +
        '<td class="ws-num">' + esc(wb.sheet_count || 0) + "</td>" +
        "<td>" + analysisPill(wb) + "</td>" +
        '<td class="ws-actions-cell">' + fileMenuButton(wb) + "</td></tr>";
    }).join("");
    var html = '<div class="ws-pages-inner"><div class="ws-page-head"><div><h2>My files</h2>' +
      "<p>Every uploaded and created workbook, with analysis status. Use Options to open, rename, archive or permanently delete a file. Files are never deleted automatically.</p></div>" +
      '<div class="ws-page-actions"><div class="ws-filter-tabs">' +
      '<button type="button" class="' + (archived ? "" : "is-active") + '" data-page-action="files-tab" data-tab="active">Active<em>' + S.workbooks.length + "</em></button>" +
      '<button type="button" class="' + (archived ? "is-active" : "") + '" data-page-action="files-tab" data-tab="archived">Archived<em>' + S.archived.length + "</em></button>" +
      "</div><button class=\"ws-small-btn\" type=\"button\" data-action=\"new-file\">New file</button>" +
      '<button class="ws-primary-btn" type="button" data-page-action="pick-files">Upload</button></div></div>' +
      '<div class="ws-card"><div class="ws-table-wrap"><table class="ws-table"><thead><tr><th>Filename</th><th>Type</th><th>Uploaded</th><th>Last modified</th><th class="ws-num">Records</th><th class="ws-num">Sheets</th><th>Analysis</th><th></th></tr></thead><tbody>' +
      (rows || '<tr><td colspan="8"><div class="ws-empty-box"><strong>' + (archived ? "Nothing is archived" : "No files yet") + "</strong>" + (archived ? "Archived workbooks can be restored at any time." : "Upload a spreadsheet to create your first workbook.") + "</div></td></tr>") +
      "</tbody></table></div></div></div>";
    host.innerHTML = html;
  }

  function issueCategoryLabel(issue) {
    var text = (String(issue.column || "") + " " + String(issue.error || "")).toLowerCase();
    if (issue.category === "format") {
      if (/e-?mail/.test(text)) return "Email errors";
      if (/phone|mobile|contact/.test(text)) return "Phone errors";
      if (/date|dob|birth/.test(text)) return "Date errors";
      if (/age|mark|salary|amount|number|roll|usn/.test(text)) return "Number errors";
    }
    return CATEGORY_NAMES[issue.category] || issue.category || "Other issues";
  }

  function issueStatus(issue) {
    if (issue.status) return issue.status;
    if (issue.applied) return "applied";
    if (issue.rejected) return "rejected";
    return "open";
  }

  function groupFindings(issues) {
    var groups = {};
    issues.forEach(function (issue) {
      var label = issueCategoryLabel(issue);
      if (!groups[label]) groups[label] = [];
      groups[label].push(issue);
    });
    return Object.keys(groups).map(function (label) {
      return { label: label, issues: groups[label], count: groups[label].length };
    }).sort(function (a, b) { return b.count - a.count; });
  }

  function confidenceCell(issue) {
    if (issue.confidence == null) return '<span class="ws-pill">Not scored</span>';
    var value = clamp(Number(issue.confidence) || 0, 0, 100);
    return '<span class="ws-confidence"><i><b style="width:' + value + '%"></b></i>' + Math.round(value) + "%</span>";
  }

  function filteredFindings() {
    var issues = S.reportAnalysis ? (S.reportAnalysis.issues || []) : [];
    return issues.filter(function (issue) {
      var status = issueStatus(issue);
      if (S.reportStatus === "open" && status !== "open") return false;
      if (S.reportStatus !== "all" && S.reportStatus !== "open" && status !== S.reportStatus) return false;
      if (S.reportCategory !== "all" && issueCategoryLabel(issue) !== S.reportCategory) return false;
      return true;
    });
  }

  function findingRow(issue) {
    var status = issueStatus(issue);
    var row = issue.sheet_row || issue.row || "";
    var canEdit = row && issue.col;
    var selected = S.reportSelected[issue.id] ? " checked" : "";
    return '<tr class="' + (status === "applied" ? "is-applied" : status === "rejected" ? "is-rejected" : "") + '">' +
      '<td><input type="checkbox" data-issue-select="' + esc(issue.id) + '"' + selected + "></td>" +
      '<td class="ws-num">' + esc(row || "—") + "</td>" +
      "<td>" + esc(issue.column || "—") + "</td>" +
      "<td>" + esc(issue.error || issue.category || "Issue") + '<span class="ws-reason">' + esc(issue.reason || "") + "</span></td>" +
      '<td class="ws-original">' + esc(issue.original == null || issue.original === "" ? "(blank)" : issue.original) + "</td>" +
      '<td class="ws-suggestion">' + esc(issue.suggestion == null ? "—" : issue.suggestion) + "</td>" +
      "<td>" + confidenceCell(issue) + "</td>" +
      "<td>" + (status === "applied" ? '<span class="ws-pill ok">Applied</span>' : status === "rejected" ? '<span class="ws-pill">Rejected</span>' : '<span class="ws-pill warn">Open</span>') + "</td>" +
      '<td class="ws-actions-cell">' +
      (issue.suggestion != null && canEdit ? '<button class="ws-small-btn" type="button" data-finding-action="accept" data-id="' + esc(issue.id) + '">Accept</button>' : "") +
      (canEdit ? '<button class="ws-small-btn ghost" type="button" data-finding-action="edit" data-id="' + esc(issue.id) + '">Edit</button>' : "") +
      (canEdit ? '<button class="ws-small-btn ghost" type="button" data-finding-action="view" data-id="' + esc(issue.id) + '">Row</button>' : "") +
      (status === "open"
        ? '<button class="ws-small-btn ghost" type="button" data-finding-action="reject" data-id="' + esc(issue.id) + '">Reject</button>'
        : '<button class="ws-small-btn ghost" type="button" data-finding-action="reopen" data-id="' + esc(issue.id) + '">Reopen</button>') +
      "</td></tr>";
  }

  function analysisReportHtml(wb) {
    var analysis = S.reportAnalysis;
    var issues = analysis ? (analysis.issues || []) : [];
    var grouped = groupFindings(issues);
    var applied = issues.filter(function (issue) { return issueStatus(issue) === "applied"; }).length;
    var rejected = issues.filter(function (issue) { return issueStatus(issue) === "rejected"; }).length;
    var open = issues.length - applied - rejected;
    var suggested = issues.filter(function (issue) { return issue.suggestion != null; }).length;
    var missing = issues.filter(function (issue) { return issue.category === "missing"; }).length;
    var duplicates = issues.filter(function (issue) { return issue.category === "duplicate"; }).length;
    var filtered = filteredFindings();
    var shown = filtered.slice(0, 400);
    var html = "";

    if (!analysis) {
      return '<div class="ws-card"><div class="ws-empty-box"><strong>No analysis stored for this workbook.</strong>' +
        'Run the cleaning rules to detect spelling, spacing, missing values, duplicates, email, phone, date and ID problems.' +
        '<div class="ws-page-actions" style="justify-content:center;margin-top:12px">' +
        '<button class="ws-primary-btn" type="button" data-page-action="analyse-report">Run analysis</button>' +
        '<button class="ws-small-btn" type="button" data-page-action="open-sheet">Open Excel sheet</button></div></div></div>';
    }

    html += '<div class="ws-card"><div class="ws-card-head"><div><strong>File: ' + esc(wb.name) + "</strong>" +
      '<span>Sheet ' + esc(analysis.sheet_name || "Sheet") + " · analysed " + esc(when(analysis.created_at)) +
      (analysis.stale ? " · data changed since this analysis" : "") + "</span></div>" +
      '<div class="ws-page-actions"><button class="ws-small-btn" type="button" data-page-action="analyse-report">Analyse again</button>' +
      '<button class="ws-small-btn ghost" type="button" data-page-action="export-findings">Export findings CSV</button>' +
      '<button class="ws-primary-btn" type="button" data-page-action="open-sheet">Open Excel sheet</button></div></div>' +
      '<div class="ws-summary-strip">' +
      '<div><span>Records</span><strong>' + esc(wb.record_count || analysis.analyzed_rows || 0) + "</strong></div>" +
      '<div><span>Columns</span><strong>' + esc((analysis.columns || []).length) + "</strong></div>" +
      '<div><span>Worksheets</span><strong>' + esc(wb.sheet_count || 0) + "</strong></div>" +
      '<div><span>Problems</span><strong>' + issues.length + "</strong></div>" +
      "</div><div class=\"ws-summary-strip\">" +
      "<div><span>Missing values</span><strong>" + missing + "</strong></div>" +
      "<div><span>Duplicate records</span><strong>" + duplicates + "</strong></div>" +
      "<div><span>Suggested corrections</span><strong>" + suggested + "</strong></div>" +
      "<div><span>Corrected / rejected</span><strong>" + applied + " / " + rejected + "</strong></div>" +
      "</div></div>";

    html += '<div class="ws-card"><div class="ws-card-head"><strong>Problem categories</strong><span>All counts come from the stored analysis</span></div>' +
      (grouped.length ? grouped.map(function (group) {
        return '<div class="ws-cat-row"><strong>' + esc(group.label) + "</strong><span class=\"ws-num\">" + group.count + "</span>" +
          '<button class="ws-small-btn" type="button" data-page-action="category" data-category="' + esc(group.label) + '">View</button></div>';
      }).join("") : '<div class="ws-empty-box">No problems were detected in the analysed rows.</div>') + "</div>";

    html += '<div class="ws-card"><div class="ws-card-head"><strong>View problems and correct them</strong>' +
      '<span>' + filtered.length + " shown · " + open + " open</span></div>" +
      '<div class="ws-toolbar"><div class="ws-filter-tabs">' +
      ['open', 'all', 'applied', 'rejected'].map(function (status) {
        var counts = { open: open, all: issues.length, applied: applied, rejected: rejected };
        return '<button type="button" class="' + (S.reportStatus === status ? "is-active" : "") + '" data-page-action="status-tab" data-status="' + status + '">' +
          status.charAt(0).toUpperCase() + status.slice(1) + "<em>" + counts[status] + "</em></button>";
      }).join("") + "</div>" +
      '<span class="ws-pill info">' + esc(S.reportCategory === "all" ? "All categories" : S.reportCategory) + "</span>" +
      (S.reportCategory !== "all" ? '<button class="ws-small-btn ghost" type="button" data-page-action="all-categories">Clear category</button>' : "") +
      '<span class="ws-spacer"></span>' +
      '<button class="ws-small-btn" type="button" data-page-action="apply-selected">Accept selected</button>' +
      '<button class="ws-small-btn" type="button" data-page-action="reject-selected">Reject selected</button>' +
      '<button class="ws-small-btn" type="button" data-page-action="accept-high">Accept all high-confidence</button>' +
      '<button class="ws-small-btn ghost" type="button" data-page-action="undo-correction">Undo last correction</button>' +
      "</div>" +
      '<div class="ws-table-wrap"><table class="ws-table ws-problem-table"><thead><tr>' +
      '<th><input type="checkbox" id="ws-select-all" aria-label="Select all visible findings"></th><th>Row</th><th>Column</th><th>Problem</th><th>Original</th><th>Suggested</th><th>Confidence</th><th>Status</th><th></th></tr></thead><tbody>' +
      (shown.length ? shown.map(findingRow).join("") : '<tr><td colspan="9"><div class="ws-empty-box">No findings match this filter.</div></td></tr>') +
      "</tbody></table></div>" +
      (filtered.length > shown.length ? '<div class="ws-card-body"><span class="ws-inline-note">Showing the first ' + shown.length + " of " + filtered.length + " findings. Use the category filter to narrow the list.</span></div>" : "") +
      "</div>";
    return html;
  }

  async function renderAnalysisPage(wbId) {
    if (!S.workbooks.length && !S.archived.length) await refreshLibrary();
    var books = S.workbooks.concat(S.archived);
    if (!books.length) {
      setPageHeading("analysis");
      $("ws-pages").innerHTML = '<div class="ws-pages-inner"><div class="ws-card"><div class="ws-empty-box"><strong>No workbooks to analyse yet.</strong>' +
        '<div class="ws-page-actions" style="justify-content:center;margin-top:12px"><button class="ws-primary-btn" type="button" data-page-action="pick-files">Upload dataset</button>' +
        '<button class="ws-small-btn" type="button" data-action="new-file">New file</button></div></div></div></div>';
      return;
    }
    if (!wbId || !findWorkbook(wbId)) wbId = S.reportWbId || books[0].id;
    var wb = findWorkbook(wbId) || books[0];
    setPageHeading("analysis");
    var host = $("ws-pages");
    host.innerHTML = '<div class="ws-pages-inner"><div class="ws-empty-box">Loading the stored analysis…</div></div>';
    var data = await api("/workbooks/" + encodeURIComponent(wb.id) + "/analysis");
    if (S.reportWbId !== wb.id) {
      S.reportStatus = "open";
      S.reportCategory = "all";
    }
    S.reportWbId = wb.id;
    S.reportAnalysis = data.analysis;
    S.reportSelected = {};
    var selector = '<div class="ws-page-actions"><label class="ws-inline-field">Workbook <select id="ws-report-wb">' +
      books.map(function (item) {
        return '<option value="' + esc(item.id) + '"' + (item.id === wb.id ? " selected" : "") + ">" + esc(item.name) + "</option>";
      }).join("") + "</select></label></div>";
    var html = '<div class="ws-pages-inner"><div class="ws-page-head"><div><h2>Data analysis</h2>' +
      "<p>Review detected problems, accept or reject corrections and open the sheet. The original values stay visible next to every suggestion.</p></div>" +
      selector + "</div>" + analysisReportHtml(wb) + "</div>";
    host.innerHTML = html;
  }

  async function renderReports() {
    var data = await api("/reports");
    S.reports = data;
    var totals = data.totals || {};
    var items = data.reports || [];
    setPageHeading("reports");
    var rows = items.map(function (report) {
      var categories = Object.keys(report.categories || {}).sort(function (a, b) {
        return report.categories[b] - report.categories[a];
      }).slice(0, 3).map(function (key) {
        return '<span class="ws-pill">' + esc(CATEGORY_NAMES[key] || key) + " " + report.categories[key] + "</span>";
      }).join(" ");
      return '<tr><td class="ws-name-cell" title="' + esc(report.workbook_name) + '">' + esc(report.workbook_name) + "</td>" +
        "<td>" + esc(report.sheet_name) + "</td>" +
        '<td class="ws-num">' + esc(report.analyzed_rows) + "</td>" +
        '<td class="ws-num">' + esc(report.total) + "</td>" +
        '<td class="ws-num">' + esc(report.applied) + "</td>" +
        '<td class="ws-num">' + esc(report.rejected) + "</td>" +
        "<td>" + esc(when(report.created_at)) + "</td>" +
        "<td>" + (report.stale ? '<span class="ws-pill warn">Data changed</span>' : '<span class="ws-pill ok">Current</span>') + "</td>" +
        '<td><div class="ws-report-cats">' + (categories || '<span class="ws-pill ok">No issues</span>') + "</div></td>" +
        '<td class="ws-actions-cell"><button class="ws-small-btn" type="button" data-file-action="report" data-id="' + esc(report.workbook_id) + '">Open report</button>' +
        '<button class="ws-small-btn ghost" type="button" data-file-action="open" data-id="' + esc(report.workbook_id) + '">Open sheet</button></td></tr>';
    }).join("");
    var html = '<div class="ws-pages-inner"><div class="ws-page-head"><div><h2>Reports</h2>' +
      "<p>Data-quality summaries and correction reports for analysed workbooks. Counts are calculated from stored analysis data.</p></div>" +
      '<div class="ws-page-actions"><button class="ws-small-btn" type="button" data-page-action="refresh-reports">Refresh</button>' +
      '<button class="ws-primary-btn" type="button" data-page-action="go-analysis">Data analysis</button></div></div>' +
      '<div class="ws-stat-grid">' +
      statCard("Files analysed", totals.analysed || 0, (totals.files || 0) + " active files", "accent") +
      statCard("Problems detected", totals.issues || 0, "All stored findings", totals.issues ? "warn" : "") +
      statCard("Corrections applied", totals.corrected || 0, "Undoable in the editor", "ok") +
      statCard("Rejected findings", totals.rejected || 0, "Kept unchanged", "") +
      "</div>" +
      '<div class="ws-card"><div class="ws-card-head"><strong>Analysis reports</strong><span>' + items.length + " report(s)</span></div>" +
      '<div class="ws-table-wrap"><table class="ws-table"><thead><tr><th>Workbook</th><th>Sheet</th><th class="ws-num">Rows</th><th class="ws-num">Detected</th><th class="ws-num">Corrected</th><th class="ws-num">Rejected</th><th>Analysed</th><th>Status</th><th>Top categories</th><th></th></tr></thead><tbody>' +
      (rows || '<tr><td colspan="10"><div class="ws-empty-box"><strong>No reports yet</strong>Upload and analyse a workbook to create the first report.</div></td></tr>') +
      "</tbody></table></div></div></div>";
    $("ws-pages").innerHTML = html;
  }

  function selectedFindings() {
    return (S.reportAnalysis ? S.reportAnalysis.issues || [] : []).filter(function (issue) {
      return S.reportSelected[issue.id];
    });
  }

  async function applyFindings(issues) {
    if (!S.reportWbId || !S.reportAnalysis) return;
    var seen = {};
    var corrections = [];
    issues.forEach(function (issue) {
      var row = issue.sheet_row || issue.row;
      if (!row || !issue.col || issue.suggestion == null) return;
      var key = row + ":" + issue.col;
      if (seen[key]) return;
      seen[key] = true;
      corrections.push({ sheet_id: S.reportAnalysis.sheet_id, row: row, col: issue.col, value: issue.suggestion });
    });
    if (!corrections.length) { toast("Select at least one finding with a suggested correction.", "error"); return; }
    if (S.wb && S.wb.id === S.reportWbId) await flushSave();
    var result = await api("/workbooks/" + encodeURIComponent(S.reportWbId) + "/analysis/apply", {
      method: "POST", body: { sheet_id: S.reportAnalysis.sheet_id, corrections: corrections },
    });
    if (S.wb && S.wb.id === S.reportWbId) {
      adoptWorkbook(result.workbook);
      setUndo(result.can_undo, result.can_redo);
      S.lastSaved = new Date();
      await reloadGrid({ keepScroll: true });
    }
    S.reportSelected = {};
    await refreshLibrary();
    await renderAnalysisPage(S.reportWbId);
    toast(corrections.length + " " + plural(corrections.length, "correction") + " applied. Every correction is undoable.");
  }

  async function setFindingsStatus(issues, status) {
    if (!S.reportWbId || !issues.length) { toast("Select at least one finding.", "error"); return; }
    var ids = issues.map(function (issue) { return issue.id; }).filter(Boolean);
    await api("/workbooks/" + encodeURIComponent(S.reportWbId) + "/analysis/status", {
      method: "POST", body: { status: status, ids: ids },
    });
    S.reportSelected = {};
    await renderAnalysisPage(S.reportWbId);
    toast(issues.length + " " + plural(issues.length, "finding") + (status === "rejected" ? " rejected. No data was changed." : " reopened."));
  }

  async function analyseWorkbook(wbId, sheetId) {
    var wb = findWorkbook(wbId);
    if (!wb) return null;
    if (S.wb && S.wb.id === wbId) await flushSave();
    toast("Running the local cleaning rules…");
    var body = {};
    if (sheetId) body.sheet_id = sheetId;
    var data = await api("/workbooks/" + encodeURIComponent(wbId) + "/analysis", { method: "POST", body: body });
    await refreshLibrary();
    var count = (data.analysis.issues || []).length;
    toast(count ? count + " " + plural(count, "finding") + " detected in " + (data.analysis.sheet_name || "the sheet") + "." : "No problems detected in the analysed rows.");
    if (S.page === "analysis" || S.page === "reports" || S.page === "files") {
      await setPage(S.page === "analysis" ? "analysis" : S.page, { wbId: wbId });
    }
    return data.analysis;
  }

  async function renameWorkbookById(id) {
    var wb = findWorkbook(id);
    if (!wb) return;
    var name = await askName({
      title: "Rename workbook", text: "The workbook keeps its ID, versions, analysis and exports.",
      value: wb.name, button: "Rename",
    });
    if (name == null) return;
    if (!name.trim()) { toast("Enter a workbook name.", "error"); return; }
    await patchWorkbook(id, { name: name.trim() });
    toast("Workbook renamed.");
    await setPage(S.page, { wbId: S.reportWbId });
  }

  async function downloadWorkbookFile(id) {
    var wb = findWorkbook(id);
    if (!wb) return;
    if (wb.original_name) {
      triggerDownload(API + "/workbooks/" + encodeURIComponent(id) + "/original");
      toast("Downloading the untouched original file.");
      return;
    }
    var sheetIds = (wb.sheets || []).map(function (sheet) { return sheet.id; });
    var data = await api("/workbooks/" + encodeURIComponent(id) + "/export", {
      method: "POST", body: { format: "xlsx", sheet_ids: sheetIds },
    });
    triggerDownload(data.export.url);
    toast("Workbook exported as XLSX: " + data.export.filename);
  }

  async function downloadReportFindings(workbookId) {
    if (!workbookId) { toast("Choose a workbook first.", "error"); return; }
    var rows;
    if (S.reportAnalysis && S.reportWbId === workbookId) rows = S.reportAnalysis.issues || [];
    else {
      var data = await api("/workbooks/" + encodeURIComponent(workbookId) + "/analysis");
      rows = data.analysis ? data.analysis.issues || [] : [];
    }
    exportIssueCsv(rows, workbookId);
  }

  function exportIssueCsv(issues, workbookId) {
    var wb = findWorkbook(workbookId);
    var header = ["sheet", "row", "column", "category", "issue", "original", "suggested", "confidence", "severity", "status", "reason"];
    function csv(value) { return '"' + String(value == null ? "" : value).replace(/"/g, '""') + '"'; }
    var lines = [header.map(csv).join(",")];
    issues.forEach(function (issue) {
      lines.push([
        (S.reportAnalysis && S.reportWbId === workbookId) ? (S.reportAnalysis.sheet_name || "") : "",
        issue.sheet_row || issue.row || "", issue.column || "", issueCategoryLabel(issue),
        issue.error || "", issue.original || "", issue.suggestion == null ? "" : issue.suggestion,
        issue.confidence == null ? "" : issue.confidence, issue.severity || "", issueStatus(issue), issue.reason || "",
      ].map(csv).join(","));
    });
    var blob = new Blob(["\ufeff" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
    var anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = ((wb ? wb.name : "datafix").replace(/[^\w \-]/g, "_")) + "-findings.csv";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(function () { URL.revokeObjectURL(anchor.href); }, 4000);
    toast("Findings CSV downloaded (" + issues.length + " rows).");
  }

  async function editFinding(issue) {
    var value = await askName({
      title: "Edit correction",
      text: "Change the suggested value for " + (issue.column || "this cell") + " at row " + (issue.sheet_row || issue.row) + ". The original value is never changed until you apply.",
      value: issue.suggestion == null ? "" : String(issue.suggestion), button: "Apply correction",
    });
    if (value == null) return;
    var copy = Object.assign({}, issue, { suggestion: value });
    await applyFindings([copy]);
  }

  async function handleFileAction(action, id) {
    if (!id) return;
    if (action === "open") { await openWorkbook(id); return; }
    if (action === "report") { await setPage("analysis", { wbId: id }); return; }
    if (action === "analyse") { await analyseWorkbook(id); return; }
    if (action === "download") { await downloadWorkbookFile(id); return; }
    if (action === "rename") { await renameWorkbookById(id); return; }
    if (action === "archive" || action === "restore") {
      await toggleArchive(id);
      await refreshLibrary();
      await setPage("files");
      return;
    }
    if (action === "delete") { await deleteWorkbookById(id); return; }
  }

  function findingById(id) {
    return (S.reportAnalysis ? S.reportAnalysis.issues || [] : []).filter(function (issue) {
      return issue.id === id;
    })[0] || null;
  }

  async function handlePageAction(action, node) {
    if (!action) return;
    if (action === "pick-files") { $("ws-file-input").click(); return; }
    if (action === "open-import") { $("ws-import-path").value = ""; openModal("ws-modal-import"); return; }
    if (action === "go-home") { await setPage("home"); return; }
    if (action === "go-files") { await setPage("files"); return; }
    if (action === "go-analysis") { await setPage("analysis"); return; }
    if (action === "go-reports") { await setPage("reports"); return; }
    if (action === "files-tab") { S.filesTab = node.dataset.tab || "active"; await renderFilesPage(); return; }
    if (action === "refresh-reports") { await renderReports(); toast("Reports refreshed."); return; }
    if (action === "category") {
      S.reportCategory = node.dataset.category || "all";
      S.reportStatus = "all";
      await renderAnalysisPage(S.reportWbId);
      var problems = qs("#ws-select-all");
      if (problems) problems.closest(".ws-card").scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    if (action === "all-categories") { S.reportCategory = "all"; await renderAnalysisPage(S.reportWbId); return; }
    if (action === "status-tab") { S.reportStatus = node.dataset.status || "open"; await renderAnalysisPage(S.reportWbId); return; }
    if (action === "apply-selected") { await applyFindings(selectedFindings()); return; }
    if (action === "reject-selected") { await setFindingsStatus(selectedFindings(), "rejected"); return; }
    if (action === "accept-high") {
      var high = (S.reportAnalysis ? S.reportAnalysis.issues || [] : []).filter(function (issue) {
        return issueStatus(issue) === "open" && issue.suggestion != null && (Number(issue.confidence) || 0) >= 90;
      });
      if (!high.length) { toast("No open high-confidence corrections are available.", "error"); return; }
      await applyFindings(high);
      return;
    }
    if (action === "undo-correction") {
      if (!S.reportWbId) return;
      try {
        var result = await api("/workbooks/" + encodeURIComponent(S.reportWbId) + "/undo", { method: "POST" });
        if (S.wb && S.wb.id === S.reportWbId) {
          adoptWorkbook(result.result.workbook);
          setUndo(result.result.can_undo, result.result.can_redo);
          await reloadGrid({ keepScroll: true });
        }
        await refreshLibrary();
        await renderAnalysisPage(S.reportWbId);
        toast("Undid: " + result.result.label + ". Re-run the analysis to refresh the findings.");
      } catch (error) {
        toast(error.message || "Nothing to undo.", "error");
      }
      return;
    }
    if (action === "analyse-report") {
      if (!S.reportWbId) return;
      await analyseWorkbook(S.reportWbId, S.reportAnalysis ? S.reportAnalysis.sheet_id : null);
      return;
    }
    if (action === "open-sheet") {
      if (!S.reportWbId) return;
      var options = {};
      if (S.reportAnalysis && S.reportAnalysis.sheet_id) options.sheetId = S.reportAnalysis.sheet_id;
      await openWorkbook(S.reportWbId, options);
      return;
    }
    if (action === "export-findings") { await downloadReportFindings(S.reportWbId); return; }
  }

  async function patchWorkbook(id, body) {
    var data = await api("/workbooks/" + encodeURIComponent(id), { method: "PATCH", body: body });
    if (S.wb && S.wb.id === id) { adoptWorkbook(data.workbook); updateHeader(); }
    await refreshLibrary();
    return data.workbook;
  }

  async function toggleStar(id) {
    var wb = S.workbooks.filter(function (item) { return item.id === id; })[0] ||
      S.archived.filter(function (item) { return item.id === id; })[0];
    if (!wb) return;
    await patchWorkbook(id, { starred: !wb.starred });
    if (S.wb && S.wb.id === id) updateHeader();
  }

  async function toggleArchive(id) {
    var wb = S.workbooks.filter(function (item) { return item.id === id; })[0] ||
      S.archived.filter(function (item) { return item.id === id; })[0];
    if (!wb) return;
    if (!wb.archived) {
      var confirmed = await confirmDialog({
        title: "Archive this workbook?",
        text: "\"" + wb.name + "\" moves to Archived. Its data, versions and exports are kept and it can be restored at any time. Nothing is deleted.",
        button: "Archive",
        danger: false,
      });
      if (!confirmed) return;
    }
    if (S.wb && S.wb.id === id) {
      await flushSave();
      if (S.queue.length) { toast("Pending changes must be saved before archiving.", "error"); return; }
    }
    await patchWorkbook(id, { archived: !wb.archived });
    if (S.wb && S.wb.id === id) await closeWorkbook(false);
    toast(wb.archived ? "Workbook restored." : "Workbook archived. Find it under Archived.");
  }

  async function deleteWorkbookById(id) {
    var wb = findWorkbook(id);
    if (!wb) return;
    if (S.wb && S.wb.id === id) {
      await flushSave();
      if (S.queue.length) { toast("Pending changes must be saved before deleting.", "error"); return; }
    }
    var confirmed = await confirmDialog({
      title: "Delete this workbook permanently?",
      text: "\"" + wb.name + "\" and its versions, analysis history and exported copies will be removed from this computer. This cannot be undone. Other workbooks are not touched. Use Archive instead if you only want to hide it.",
      button: "Delete permanently",
      danger: true,
    });
    if (!confirmed) return;
    await api("/workbooks/" + encodeURIComponent(id), { method: "DELETE" });
    if (S.wb && S.wb.id === id) await closeWorkbook(false);
    if (S.reportWbId === id) { S.reportWbId = null; S.reportAnalysis = null; }
    await refreshLibrary();
    await setPage("files");
    toast("Deleted \"" + wb.name + "\". Other workbooks were not changed.");
  }

  async function toggleStarCurrent() {
    if (!S.wb) return;
    await toggleStar(S.wb.id);
  }

  /* -------------------------------------------------------------- workbook */

  function adoptWorkbook(workbook) {
    S.wb = workbook;
    S.sheets = workbook.sheets || [];
    if (!S.sheetId || !S.sheets.some(function (sheet) { return sheet.id === S.sheetId; })) {
      S.sheetId = S.sheets.length ? S.sheets[0].id : null;
    }
    updateHeader();
    renderSheetTabs();
    refreshColumnChoices();
    updateSaveChip();
  }

  function enterEditorMode() {
    S.page = "editor";
    var main = qs(".ws-main");
    if (main) main.classList.remove("is-page");
    $("ws-pages").hidden = true;
    qsa(".ws-page-item").forEach(function (button) { button.classList.remove("is-active"); });
  }

  function updateHeader() {
    if (S.page !== "editor") return;
    var wb = S.wb;
    $("ws-title").textContent = wb ? wb.name : "Workspace";
    $("ws-title").title = wb ? "Double-click to rename this workbook" : "";
    $("ws-star-btn").hidden = !wb;
    $("ws-star-btn").classList.toggle("is-on", Boolean(wb && wb.starred));
    $("ws-star-btn").title = wb && wb.starred ? "Remove star" : "Star this file";
    if (!wb) {
      $("ws-meta").textContent = "Choose a saved workbook or create a new one.";
      document.title = "DataFix Workspace";
      return;
    }
    var parts = [
      (wb.sheet_count || S.sheets.length) + " " + plural(wb.sheet_count || S.sheets.length, "worksheet"),
      (wb.version_count || 0) + " " + plural(wb.version_count || 0, "version"),
      "updated " + when(wb.updated_at),
    ];
    if (wb.original_name) parts.push(wb.original_name);
    $("ws-meta").textContent = parts.join(" · ");
    document.title = wb.name + " — DataFix Workspace";
  }

  function renderSheetTabs() {
    var host = $("ws-sheet-tabs");
    if (!host) return;
    if (!S.wb) { host.innerHTML = ""; return; }
    host.innerHTML = S.sheets.map(function (sheet) {
      return '<button class="ws-sheet-tab' + (sheet.id === S.sheetId ? " is-active" : "") + '" type="button" role="tab" data-sheet="' + sheet.id + '">' + esc(sheet.name) + "</button>";
    }).join("");
  }

  function updateHash() {
    if (!S.wb) {
      if (window.location.hash) window.history.replaceState(null, "", window.location.pathname);
      return;
    }
    var hash = "#/w/" + S.wb.id + "/" + S.sheetId;
    if (window.location.hash !== hash) window.history.replaceState(null, "", hash);
  }

  async function createWorkbook(name) {
    var data = await api("/workbooks", { method: "POST", body: { name: name || "" } });
    await refreshLibrary();
    await openWorkbook(data.workbook.id, { sheetId: data.workbook.sheets[0].id });
    toast("New workbook created in permanent storage.");
  }

  async function openWorkbook(id, options) {
    options = options || {};
    var token = ++S.openToken;
    await flushSave();
    if (S.queue.length) {
      toast("Pending changes could not be saved, so the file was not switched.", "error");
      return;
    }
    var data = await api("/workbooks/" + encodeURIComponent(id));
    if (token !== S.openToken) return;
    var workbook = data.workbook;
    enterEditorMode();
    S.sheetId = null;
    adoptWorkbook(workbook);
    if (options.sheetId != null) {
      var wanted = Number(options.sheetId);
      if (S.sheets.some(function (sheet) { return sheet.id === wanted; })) S.sheetId = wanted;
    }
    S.filters = [];
    S.filterDraft = [];
    S.sort = null;
    S.search = "";
    S.active = { r: 1, c: 1 };
    S.sel = { r1: 1, c1: 1, r2: 1, c2: 1 };
    S.anchor = { r: 1, c: 1 };
    S.pending.clear();
    S.highlight = null;
    S.analysis = null;
    S.analysisStale = false;
    S.issueIndex = -1;
    S.showFormulas = false;
    S.save.error = "";
    $("ws-empty").hidden = true;
    $("ws-analysis-panel").hidden = true;
    $("ws-filter-panel").hidden = true;
    syncFilterUI();
    renderSheetTabs();
    await reloadGrid({ resetScroll: true, resetSelection: true });
    if (token !== S.openToken) return;
    if (options.cell) await revealCell(options.cell.row, options.cell.col, S.sheetId);
    loadVersionState().catch(function () {});
    loadAnalysis().catch(function () {});
    updateHeader();
    updateHash();
    refreshLibrary();
    document.querySelector(".ws-app").classList.remove("sidebar-open");
    var scrim = $("ws-sidebar-scrim");
    if (scrim) scrim.hidden = true;
  }

  async function closeWorkbook(showToast) {
    await flushSave();
    if (S.queue.length) {
      toast("Save the pending changes before closing this workbook.", "error");
      return;
    }
    ++S.openToken;
    S.wb = null;
    S.sheetId = null;
    S.sheets = [];
    S.sheetMeta = null;
    S.rows = [];
    S.columns = [];
    S.cells = new Map();
    S.headers = {};
    S.totalRows = 0;
    S.rowCount = 0;
    S.colCount = 0;
    S.filters = [];
    S.filterDraft = [];
    S.sort = null;
    S.search = "";
    S.analysis = null;
    S.analysisStale = false;
    S.pending.clear();
    S.gridSerial += 1;
    S.lastSaved = null;
    $("ws-empty").hidden = false;
    $("ws-analysis-panel").hidden = true;
    $("ws-filter-panel").hidden = true;
    $("ws-grid-cells").innerHTML = "";
    $("ws-colheads-inner").innerHTML = "";
    $("ws-rowheads-inner").innerHTML = "";
    $("ws-name-box").value = "";
    $("ws-formula-input").value = "";
    $("ws-status-cell").textContent = "No cell selected";
    updateHeader();
    renderSheetTabs();
    renderEmptyRecent();
    renderSidebar();
    updateHash();
    updateSaveChip();
    setUndo(false, false);
    if (S.page === "editor") await setPage("home");
    if (showToast !== false) toast("Workbook closed. Your files remain in permanent storage.");
  }

  async function renameWorkbookFlow() {
    if (!S.wb) return;
    var name = await askName({
      title: "Rename workbook", text: "The saved file keeps its workbook ID, versions and history.",
      value: S.wb.name, button: "Rename",
    });
    if (name == null) return;
    if (!name.trim()) { toast("Enter a workbook name.", "error"); return; }
    var workbook = await patchWorkbook(S.wb.id, { name: name.trim() });
    adoptWorkbook(workbook);
    toast("Workbook renamed.");
  }

  async function saveAsFlow() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    var name = await askName({
      title: "Save a copy", text: "Creates an independent copy with its own ID, cells and history. The original is untouched.",
      value: S.wb.name + " copy", button: "Create copy",
    });
    if (name == null) return;
    await flushSave();
    var data = await api("/workbooks/" + S.wb.id + "/save-as", { method: "POST", body: { name: name.trim() } });
    await refreshLibrary();
    await openWorkbook(data.workbook.id);
    toast("Copy saved as a separate workbook.");
  }

  async function saveNow() {
    if (!S.wb) { toast("Open or create a workbook first.", "error"); return; }
    await flushSave();
    if (S.queue.length) { toast("Autosave failed; use the retry chip in the top bar.", "error"); return; }
    var data = await api("/workbooks/" + S.wb.id + "/versions", {
      method: "POST", body: { label: "Manual save" },
    });
    S.lastSaved = new Date();
    updateSaveChip();
    toast("Saved permanently as version " + data.version.number + ".");
    await refreshLibrary();
    updateHeader();
  }

  async function checkpointFlow(label) {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    await flushSave();
    var data = await api("/workbooks/" + S.wb.id + "/versions", {
      method: "POST", body: { label: label || "Manual checkpoint" },
    });
    toast("Checkpoint v" + data.version.number + " created.");
    await loadVersionState();
  }

  /* ----------------------------------------------------------------- grid */

  function rowHeight(row) {
    return Math.round((S.rowHeights.get(row) || 26) * S.zoom);
  }
  function colWidth(col) {
    return Math.round((S.colWidths.get(col) || 100) * S.zoom);
  }
  function cellAt(row, col) {
    var key = row + ":" + col;
    var base = S.cells.get(key) || null;
    var pending = S.pending.get(key);
    if (pending) return Object.assign({}, base || {}, pending);
    return base;
  }
  function cellDisplay(cell) {
    if (!cell) return "";
    if (S.showFormulas && cell.f) return cell.f;
    var value = cell.v;
    if (value === null || value === undefined || value === "") return "";
    if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
    return String(value);
  }
  function refName(row, col) { return letter(col) + row; }
  function selectionRect() {
    return {
      r1: Math.min(S.sel.r1, S.sel.r2), c1: Math.min(S.sel.c1, S.sel.c2),
      r2: Math.max(S.sel.r1, S.sel.r2), c2: Math.max(S.sel.c1, S.sel.c2),
    };
  }
  function selectionSize() {
    var rect = selectionRect();
    return (rect.r2 - rect.r1 + 1) * (rect.c2 - rect.c1 + 1);
  }

  async function fetchGridPage(settings) {
    var workbook = S.wb;
    if (!workbook) return null;
    var serial = S.gridSerial;
    var params = new URLSearchParams({
      sheet_id: String(S.sheetId),
      start_row: String(settings.startRow || 1),
      rows: String(settings.rowCount || 200),
      start_col: String(settings.startCol || 1),
      cols: String(settings.colCount || 60),
    });
    if (S.filters.length) params.set("filters", JSON.stringify(S.filters));
    if (S.sort) params.set("sort", JSON.stringify(S.sort));
    if (S.search) params.set("search", S.search);
    var data = await api("/workbooks/" + encodeURIComponent(workbook.id) + "/grid?" + params.toString());
    if (!S.wb || S.wb.id !== workbook.id || serial !== S.gridSerial) return null;
    return data;
  }

  function mergeSheetMeta(sheet) {
    S.sheetMeta = sheet;
    S.frozenRows = Number(sheet.frozen_rows) || 0;
    S.frozenCols = Number(sheet.frozen_cols) || 0;
    S.rowCount = Number(sheet.row_count) || 0;
    S.colCount = Number(sheet.col_count) || 0;
    S.sheets = S.sheets.map(function (item) { return item.id === sheet.id ? sheet : item; });
  }

  async function reloadGrid(options) {
    options = options || {};
    if (!S.wb) return;
    var token = ++S.gridSerial;
    var viewport = $("ws-grid-viewport");
    if (!options.keepScroll && viewport) viewport.scrollTop = viewport.scrollLeft = 0;
    var data = await fetchGridPage({ startRow: 1, rowCount: 200, startCol: 1, colCount: 60 });
    if (!data || token !== S.gridSerial) return;
    S.rows = data.rows || [];
    S.columns = data.columns || [];
    S.cells = new Map(Object.keys(data.cells || {}).map(function (key) { return [key, data.cells[key]]; }));
    S.headers = data.headers || {};
    S.totalRows = Number(data.total_rows) || S.rows.length;
    mergeSheetMeta(data.sheet);
    if (options.resetSelection) {
      S.active = { r: 1, c: 1 };
      S.sel = { r1: 1, c1: 1, r2: 1, c2: 1 };
      S.anchor = { r: 1, c: 1 };
    }
    if (!options.keepScroll && viewport) viewport.scrollTop = viewport.scrollLeft = 0;
    S.highlight = S.highlight && S.highlight.sheetId === S.sheetId ? S.highlight : null;
    renderGrid();
    renderSheetTabs();
    refreshColumnChoices();
    syncFormulaBar();
    updateStatusBar();
    updateHash();
  }

  async function loadMoreRows() {
    if (!S.wb || S.loadingRows || S.rows.length >= S.totalRows) return;
    S.loadingRows = true;
    var serial = S.gridSerial;
    var workbook = S.wb;
    var startRow = S.rows.length + 1;
    var lastCol = Math.min(S.colCount, Math.max(60, S.columns[S.columns.length - 1] || 60));
    try {
      var first = null;
      for (var startCol = 1; startCol <= lastCol; startCol += 60) {
        var data = await fetchGridPage({ startRow: startRow, rowCount: 180, startCol: startCol, colCount: 60 });
        if (!data || serial !== S.gridSerial || !S.wb || S.wb.id !== workbook.id) return;
        if (!first) first = data;
        Object.keys(data.cells || {}).forEach(function (key) { S.cells.set(key, data.cells[key]); });
      }
      if (!first) return;
      var known = new Set(S.rows);
      first.rows.forEach(function (row) { if (!known.has(row)) { S.rows.push(row); known.add(row); } });
      S.totalRows = first.total_rows;
      if (first.total_rows <= S.rows.length && !first.rows.length) S.totalRows = S.rows.length;
      if (!S.loadingCols) renderGrid();
    } finally {
      S.loadingRows = false;
    }
  }

  async function loadMoreCols() {
    if (!S.wb || S.loadingCols) return;
    var last = S.columns[S.columns.length - 1] || 0;
    if (last >= S.colCount) return;
    S.loadingCols = true;
    var serial = S.gridSerial;
    var workbook = S.wb;
    try {
      var added = null;
      var guard = 0;
      for (var startRow = 1; startRow <= S.rows.length && guard < 5; startRow += 400) {
        var data = await fetchGridPage({ startRow: startRow, rowCount: 400, startCol: last + 1, colCount: 40 });
        if (!data || serial !== S.gridSerial || !S.wb || S.wb.id !== workbook.id) return;
        if (!added) added = data.columns || [];
        Object.keys(data.cells || {}).forEach(function (key) { S.cells.set(key, data.cells[key]); });
        if (!data.rows.length) break;
        guard += 1;
      }
      var known = new Set(S.columns);
      (added || []).forEach(function (col) { if (!known.has(col)) { S.columns.push(col); known.add(col); } });
      S.colCount = (added && added.length) ? S.colCount : S.colCount;
      renderGrid();
    } finally {
      S.loadingCols = false;
    }
  }

  function scheduleRender() {
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(function () { renderQueued = false; renderGrid(); });
  }

  function renderGrid() {
    var viewport = $("ws-grid-viewport");
    var canvas = $("ws-grid-canvas");
    var cellsHost = $("ws-grid-cells");
    if (!viewport || !canvas || !cellsHost) return;
    if (!S.wb || !S.rows.length || !S.columns.length) {
      cellsHost.innerHTML = "";
      $("ws-colheads-inner").innerHTML = "";
      $("ws-rowheads-inner").innerHTML = "";
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      $("ws-empty").hidden = Boolean(S.wb);
      return;
    }
    var st = viewport.scrollTop, sl = viewport.scrollLeft;
    var vh = viewport.clientHeight, vw = viewport.clientWidth;
    var rowTops = [], rowHeights = [], rowSum = 0;
    S.rows.forEach(function (row) {
      rowTops.push(rowSum);
      var height = rowHeight(row);
      rowHeights.push(height);
      rowSum += height;
    });
    var colLefts = [], colWidths = [], colSum = 0;
    S.columns.forEach(function (col) {
      colLefts.push(colSum);
      var width = colWidth(col);
      colWidths.push(width);
      colSum += width;
    });
    canvas.style.width = Math.max(colSum, vw) + "px";
    canvas.style.height = Math.max(rowSum, vh) + "px";
    $("ws-colheads-inner").style.width = colSum + "px";
    $("ws-rowheads-inner").style.height = rowSum + "px";

    var frozenRows = Math.min(S.frozenRows || 0, S.rows.length);
    var frozenCols = Math.min(S.frozenCols || 0, S.columns.length);
    var rowView = [], colView = [];
    for (var ri = 0; ri < S.rows.length; ri += 1) {
      var top = ri < frozenRows ? rowTops[ri] + st : rowTops[ri];
      var screenTop = top - st;
      if (ri >= frozenRows && (screenTop + rowHeights[ri] < -40 || screenTop > vh + 40)) continue;
      rowView.push({ index: ri, top: top, height: rowHeights[ri], frozen: ri < frozenRows });
    }
    for (var ci = 0; ci < S.columns.length; ci += 1) {
      var left = ci < frozenCols ? colLefts[ci] + sl : colLefts[ci];
      var screenLeft = left - sl;
      if (ci >= frozenCols && (screenLeft + colWidths[ci] < -40 || screenLeft > vw + 40)) continue;
      colView.push({ index: ci, left: left, width: colWidths[ci], frozen: ci < frozenCols });
    }

    var rect = selectionRect();
    var parts = [];
    rowView.forEach(function (rowInfo) {
      var row = S.rows[rowInfo.index];
      colView.forEach(function (colInfo) {
        var col = S.columns[colInfo.index];
        var cell = cellAt(row, col);
        var selected = row >= rect.r1 && row <= rect.r2 && col >= rect.c1 && col <= rect.c2;
        var classes = ["ws-cell"];
        if (row === 1) classes.push("is-header");
        else if (row % 2 === 0) classes.push("is-alt");
        if (row === S.active.r && col === S.active.c) classes.push("is-active");
        if (selected) classes.push("is-selected");
        if (rowInfo.frozen || colInfo.frozen) classes.push("is-frozen");
        if (S.highlight && S.highlight.sheetId === S.sheetId && S.highlight.row === row && S.highlight.col === col) classes.push("is-match");
        var style = "top:" + rowInfo.top + "px;left:" + colInfo.left + "px;width:" + colInfo.width + "px;height:" + rowInfo.height + "px;";
        if (rowInfo.frozen || colInfo.frozen) {
          var level = (rowInfo.frozen ? 4 : 0) + (colInfo.frozen ? 1 : 0);
          style += "z-index:" + (3 + level) + ";";
        }
        var sheetStyle = (cell && cell.s) || {};
        if (sheetStyle.bold) classes.push("bold");
        if (sheetStyle.italic) classes.push("italic");
        if (sheetStyle.underline) classes.push("underline");
        if (sheetStyle.strike) classes.push("strike");
        if (sheetStyle.wrap) classes.push("is-wrap");
        if (sheetStyle.align === "center") classes.push("align-center");
        if (sheetStyle.align === "right") classes.push("align-right");
        if (cell && typeof cell.v === "string" && /^#[A-Z\/]+[!?]$/.test(cell.v)) classes.push("err");
        if (cell && cell.f && !S.showFormulas) classes.push("is-formula");
        var colour = safeColor(sheetStyle.color);
        var fill = safeColor(sheetStyle.fill);
        if (colour) style += "color:" + colour + ";";
        if (fill) style += "background:" + fill + ";";
        if (sheetStyle.font) style += "font-family:" + safeFont(sheetStyle.font) + ";";
        if (sheetStyle.size) style += "font-size:" + clamp(Number(sheetStyle.size) * S.zoom, 8, 48) + "px;";
        var text = cellDisplay(cell);
        parts.push('<div class="' + classes.join(" ") + '" data-r="' + row + '" data-c="' + col + '" style="' + style + '" title="' + esc(text) + '"><span class="ws-cell-text">' + esc(text) + "</span></div>");
      });
    });
    cellsHost.innerHTML = parts.join("");

    var headParts = [];
    colView.forEach(function (info) {
      var col = S.columns[info.index];
      var selected = col >= rect.c1 && col <= rect.c2;
      headParts.push('<div class="ws-colhead' + (selected ? " is-selected" : "") + '" data-col="' + col + '" style="left:' + info.left + 'px;width:' + info.width + 'px;">' + letter(col) +
        '<span class="ws-col-resize" data-resize-col="' + col + '"></span></div>');
    });
    var colHead = $("ws-colheads-inner");
    colHead.innerHTML = headParts.join("");
    colHead.style.transform = "translateX(" + (-sl) + "px)";

    var rowParts = [];
    rowView.forEach(function (info) {
      var row = S.rows[info.index];
      var selected = row >= rect.r1 && row <= rect.r2;
      rowParts.push('<div class="ws-rowhead' + (selected ? " is-selected" : "") + '" data-row="' + row + '" style="top:' + info.top + 'px;height:' + info.height + 'px;">' + row +
        '<span class="ws-row-resize" data-resize-row="' + row + '"></span></div>');
    });
    var rowHead = $("ws-rowheads-inner");
    rowHead.innerHTML = rowParts.join("");
    rowHead.style.transform = "translateY(" + (-st) + "px)";

    $("ws-empty").hidden = true;
    positionEditor();
    updateStatusBar();
  }

  function positionEditor() {
    if (!S.editing) return;
    var ri = S.rows.indexOf(S.active.r), ci = S.columns.indexOf(S.active.c);
    var editor = $("ws-cell-editor");
    if (ri < 0 || ci < 0) { editor.hidden = true; return; }
    var viewport = $("ws-grid-viewport");
    var top = 0, left = 0;
    for (var index = 0; index < ri; index += 1) top += rowHeight(S.rows[index]);
    for (var col = 0; col < ci; col += 1) left += colWidth(S.columns[col]);
    if (ri < S.frozenRows) top += viewport.scrollTop;
    if (ci < S.frozenCols) left += viewport.scrollLeft;
    var width = Math.max(colWidth(S.columns[ci]), 180);
    editor.style.top = top + "px";
    editor.style.left = left + "px";
    editor.style.width = width + "px";
    editor.style.height = rowHeight(S.rows[ri]) + "px";
    editor.hidden = false;
  }

  function syncFormulaBar() {
    var nameBox = $("ws-name-box");
    var formula = $("ws-formula-input");
    if (!S.wb) { nameBox.value = ""; formula.value = ""; return; }
    nameBox.value = refName(S.active.r, S.active.c);
    if (document.activeElement !== formula) {
      var cell = cellAt(S.active.r, S.active.c);
      formula.value = cell ? (cell.f || (cell.v == null ? "" : String(cell.v))) : "";
    }
  }

  function updateStatusBar() {
    if (!S.wb) { $("ws-status-cell").textContent = "No cell selected"; return; }
    var count = selectionSize();
    var cell = cellAt(S.active.r, S.active.c);
    var label = refName(S.active.r, S.active.c);
    if (count > 1) {
      $("ws-status-cell").textContent = label + " · " + count + " cells selected";
    } else {
      var text = cellDisplay(cell);
      $("ws-status-cell").textContent = label + (text ? " · " + text.slice(0, 60) : " · empty");
    }
    $("ws-status-zoom").textContent = Math.round(S.zoom * 100) + "%";
  }

  function selectCell(row, col, extend) {
    S.highlight = null;
    S.active = { r: row, c: col };
    if (extend && S.anchor) {
      S.sel = { r1: S.anchor.r, c1: S.anchor.c, r2: row, c2: col };
    } else {
      S.anchor = { r: row, c: col };
      S.sel = { r1: row, c1: col, r2: row, c2: col };
    }
    syncFormulaBar();
    renderGrid();
    updateStatusBar();
  }

  function scrollActiveIntoView() {
    if (!S.wb) return;
    var viewport = $("ws-grid-viewport");
    var ri = S.rows.indexOf(S.active.r), ci = S.columns.indexOf(S.active.c);
    if (ri < 0 || ci < 0) return;
    var top = 0, left = 0;
    for (var index = 0; index < ri; index += 1) top += rowHeight(S.rows[index]);
    for (var col = 0; col < ci; col += 1) left += colWidth(S.columns[col]);
    var height = rowHeight(S.rows[ri]), width = colWidth(S.columns[ci]);
    if (ri >= S.frozenRows) {
      if (top < viewport.scrollTop) viewport.scrollTop = top;
      else if (top + height > viewport.scrollTop + viewport.clientHeight) viewport.scrollTop = top + height - viewport.clientHeight + 2;
    }
    if (ci >= S.frozenCols) {
      if (left < viewport.scrollLeft) viewport.scrollLeft = left;
      else if (left + width > viewport.scrollLeft + viewport.clientWidth) viewport.scrollLeft = left + width - viewport.clientWidth + 2;
    }
    scheduleRender();
  }

  async function revealCell(row, col, sheetId) {
    if (!S.wb) return;
    if (sheetId != null && Number(sheetId) !== S.sheetId) {
      await selectSheet(Number(sheetId));
    }
    if (!S.wb) return;
    if (S.filters.length || S.sort || S.search) {
      S.filters = [];
      S.filterDraft = [];
      S.sort = null;
      S.search = "";
      syncFilterUI();
      await reloadGrid({ resetScroll: true });
    }
    var guard = 0;
    while (S.rows.indexOf(row) < 0 && S.rows.length < S.totalRows && guard < 60) {
      await loadMoreRows();
      guard += 1;
    }
    guard = 0;
    while (S.columns.length && S.columns[S.columns.length - 1] < col && S.columns.length < S.colCount && guard < 12) {
      await loadMoreCols();
      guard += 1;
    }
    if (S.rows.indexOf(row) < 0 || S.columns.indexOf(col) < 0) {
      toast("That cell is outside the loaded view. Use a filter to narrow the sheet.", "error");
      return;
    }
    selectCell(row, col, false);
    S.highlight = { sheetId: S.sheetId, row: row, col: col };
    scrollActiveIntoView();
    renderGrid();
    $("ws-grid-viewport").focus({ preventScroll: true });
  }

  async function moveSelection(dr, dc, extend) {
    if (!S.wb) return;
    var ri = S.rows.indexOf(S.active.r), ci = S.columns.indexOf(S.active.c);
    if (ri < 0 || ci < 0) return;
    if (dr > 0 && ri + dr >= S.rows.length && S.rows.length < S.totalRows) await loadMoreRows();
    if (dc > 0 && ci + dc >= S.columns.length && S.columns.length < S.colCount) await loadMoreCols();
    var nextRow = clamp(ri + dr, 0, Math.max(0, S.rows.length - 1));
    var nextCol = clamp(ci + dc, 0, Math.max(0, S.columns.length - 1));
    if (nextRow === ri && nextCol === ci) return;
    selectCell(S.rows[nextRow], S.columns[nextCol], extend);
    scrollActiveIntoView();
  }

  function startEdit(initial) {
    if (!S.wb) return;
    var ri = S.rows.indexOf(S.active.r), ci = S.columns.indexOf(S.active.c);
    if (ri < 0 || ci < 0) return;
    var cell = cellAt(S.active.r, S.active.c);
    var raw = cell ? (cell.f || (cell.v == null ? "" : String(cell.v))) : "";
    var editor = $("ws-cell-editor");
    var input = $("ws-cell-editor-input");
    S.editing = true;
    editor.hidden = false;
    input.value = initial == null ? raw : String(initial);
    positionEditor();
    input.focus();
    if (initial == null) input.select();
    else { var end = input.value.length; input.setSelectionRange(end, end); }
  }

  function closeEditor() {
    S.editing = false;
    $("ws-cell-editor").hidden = true;
  }

  function cancelEdit() {
    closeEditor();
    $("ws-grid-viewport").focus({ preventScroll: true });
  }

  function commitEdit(move) {
    if (!S.editing) return;
    var input = $("ws-cell-editor-input");
    var value = input.value;
    var row = S.active.r, col = S.active.c;
    var cell = cellAt(row, col);
    var raw = cell ? (cell.f || (cell.v == null ? "" : String(cell.v))) : "";
    closeEditor();
    if (value !== raw) commitValue(row, col, value);
    $("ws-grid-viewport").focus({ preventScroll: true });
    if (move === "down") moveSelection(1, 0, false);
    if (move === "right") moveSelection(0, 1, false);
    if (move === "up") moveSelection(-1, 0, false);
    if (move === "left") moveSelection(0, -1, false);
  }

  function commitValue(row, col, value) {
    if (!S.wb) return;
    var op = { op: "set_cell", sheet: S.sheetId, row: row, col: col };
    if (value === "") op.value = "";
    else if (String(value).charAt(0) === "=") op.formula = String(value);
    else op.value = value;
    rememberPending([op]);
    queueOps([op], "Edit cell");
    renderGrid();
    syncFormulaBar();
    updateStatusBar();
  }

  function commitFormulaBar() {
    if (!S.wb) return;
    var formula = $("ws-formula-input");
    var cell = cellAt(S.active.r, S.active.c);
    var raw = cell ? (cell.f || (cell.v == null ? "" : String(cell.v))) : "";
    if (formula.value !== raw) commitValue(S.active.r, S.active.c, formula.value);
    $("ws-grid-viewport").focus({ preventScroll: true });
    moveSelection(1, 0, false);
  }

  async function loadVersionState() {
    if (!S.wb) return;
    var data = await api("/workbooks/" + S.wb.id + "/versions");
    S.versions = data;
    setUndo(data.can_undo, data.can_redo);
    return data;
  }

  /* -------------------------------------------------------------- editing */

  function currentStyle() {
    var cell = cellAt(S.active.r, S.active.c);
    return Object.assign({}, (cell && cell.s) || {});
  }

  function applyStyle(patch, label) {
    if (!S.wb) return;
    if (selectionSize() > 20000) { toast("Select fewer cells before formatting.", "error"); return; }
    var rect = selectionRect();
    queueOps([{ op: "set_range", sheet: S.sheetId, range: [rect.r1, rect.c1, rect.r2, rect.c2], style: patch }], label || "Format cells");
  }

  function toggleStyle(key, label) {
    var style = currentStyle();
    var patch = {};
    patch[key] = !style[key];
    if (patch[key] === false) patch[key] = false;
    applyStyle(patch, label);
  }

  function setAlignment(align) { applyStyle({ align: align }, "Align cells"); }

  function clearStyle() {
    applyStyle({
      bold: false, italic: false, underline: false, strike: false,
      color: null, fill: null, align: null, valign: null, wrap: false,
      font: null, size: null, numfmt: null,
    }, "Clear formatting");
  }

  function clearSelectionValues() {
    if (!S.wb) return;
    var rect = selectionRect();
    if (selectionSize() > 40000) { toast("The selection is too large to clear at once.", "error"); return; }
    queueOps([{ op: "clear", sheet: S.sheetId, range: [rect.r1, rect.c1, rect.r2, rect.c2], keep_style: true }], "Clear cells");
    for (var row = rect.r1; row <= rect.r2; row += 1) {
      for (var col = rect.c1; col <= rect.c2; col += 1) S.pending.set(row + ":" + col, { f: null, v: null, n: null });
    }
    renderGrid();
  }

  function copySelection(cut) {
    if (!S.wb) return;
    var rect = selectionRect();
    if (selectionSize() > 5000) { toast("Select up to 5,000 cells to copy.", "error"); return; }
    var lines = [];
    for (var row = rect.r1; row <= rect.r2; row += 1) {
      var line = [];
      for (var col = rect.c1; col <= rect.c2; col += 1) line.push(cellDisplay(cellAt(row, col)));
      lines.push(line.join("\t"));
    }
    S.clipboard = { text: lines.join("\n"), cut: Boolean(cut), rect: rect };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(S.clipboard.text).catch(function () {});
    }
    toast(cut ? "Cut to clipboard." : "Copied to clipboard.");
    if (cut) {
      queueOps([{ op: "clear", sheet: S.sheetId, range: [rect.r1, rect.c1, rect.r2, rect.c2], keep_style: true }], "Cut cells");
      for (var r = rect.r1; r <= rect.r2; r += 1) {
        for (var c = rect.c1; c <= rect.c2; c += 1) S.pending.set(r + ":" + c, { f: null, v: null, n: null });
      }
      renderGrid();
    }
  }

  function pasteText(text, originRow, originCol) {
    if (!S.wb || !text) return;
    var rows = String(text).replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
    while (rows.length && rows[rows.length - 1] === "") rows.pop();
    if (!rows.length) return;
    var baseRow = originRow || S.active.r, baseCol = originCol || S.active.c;
    if (rows.length * rows[0].split("\t").length > 2000) { toast("Paste up to 2,000 cells at once.", "error"); return; }
    var ops = [];
    rows.forEach(function (line, rIndex) {
      line.split("\t").forEach(function (value, cIndex) {
        var row = baseRow + rIndex, col = baseCol + cIndex;
        if (row > 200000 || col > 500) return;
        var op = { op: "set_cell", sheet: S.sheetId, row: row, col: col };
        if (value.charAt(0) === "=") op.formula = value;
        else op.value = value;
        ops.push(op);
      });
    });
    if (!ops.length) return;
    rememberPending(ops);
    queueOps(ops, "Paste");
    renderGrid();
    toast(ops.length + " " + plural(ops.length, "cell") + " pasted.");
  }

  async function pasteFromClipboard() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    if (navigator.clipboard && navigator.clipboard.readText) {
      try {
        var text = await navigator.clipboard.readText();
        if (text) { pasteText(text); return; }
      } catch (error) { /* permission denied */ }
    }
    if (S.clipboard && S.clipboard.text) { pasteText(S.clipboard.text); return; }
    var manual = window.prompt("Paste values (tab-separated rows):");
    if (manual) pasteText(manual);
  }

  function setLayout(patch, label) {
    if (!S.wb) return;
    queueOps([Object.assign({ op: "set_layout", sheet: S.sheetId }, patch)], label || "Sheet layout");
    scheduleFlush(60);
    setTimeout(function () {
      var sheet = S.sheets.filter(function (item) { return item.id === S.sheetId; })[0];
      if (!sheet) return;
      if (patch.frozen_rows != null) { S.frozenRows = Number(patch.frozen_rows); sheet.frozen_rows = S.frozenRows; }
      if (patch.frozen_cols != null) { S.frozenCols = Number(patch.frozen_cols); sheet.frozen_cols = S.frozenCols; }
      if (patch.row_count != null) { S.rowCount = Number(patch.row_count); sheet.row_count = S.rowCount; }
      if (patch.col_count != null) { S.colCount = Number(patch.col_count); sheet.col_count = S.colCount; }
      renderGrid();
    }, 80);
  }

  async function structuralOp(op, label, question) {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    if (question) {
      var confirmed = await confirmDialog({ title: question.title, text: question.text, button: question.button || "Continue" });
      if (!confirmed) return;
    }
    await flushSave();
    if (S.queue.length) { toast("Pending changes must be saved first.", "error"); return; }
    var result = await api("/workbooks/" + S.wb.id + "/ops", { method: "POST", body: { ops: [op], label: label } });
    adoptWorkbook(result.workbook);
    setUndo(result.can_undo, result.can_redo);
    S.lastSaved = new Date();
    S.highlight = null;
    await reloadGrid({ keepScroll: false });
    toast(label + ".");
  }

  async function renameSheetFlow() {
    if (!S.wb) return;
    var sheet = S.sheets.filter(function (item) { return item.id === S.sheetId; })[0];
    if (!sheet) return;
    var name = await askName({ title: "Rename worksheet", text: "Sheet names are limited to 31 characters.", value: sheet.name, button: "Rename" });
    if (name == null || !name.trim() || name.trim() === sheet.name) return;
    await structuralOp({ op: "rename_sheet", sheet: S.sheetId, name: name.trim() }, "Rename sheet");
  }

  async function addSheetFlow() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    var sheet = S.sheets.filter(function (item) { return item.id === S.sheetId; })[0];
    var before = new Set(S.sheets.map(function (item) { return item.id; }));
    await flushSave();
    var result = await api("/workbooks/" + S.wb.id + "/ops", {
      method: "POST", body: { ops: [{ op: "add_sheet", sheet: S.sheetId, after: sheet ? sheet.position + 1 : S.sheets.length }], label: "Add sheet" },
    });
    adoptWorkbook(result.workbook);
    setUndo(result.can_undo, result.can_redo);
    var added = S.sheets.filter(function (item) { return !before.has(item.id); })[0] || S.sheets[S.sheets.length - 1];
    S.sheetId = added.id;
    S.filters = [];
    S.filterDraft = [];
    S.sort = null;
    S.search = "";
    S.active = { r: 1, c: 1 };
    S.sel = { r1: 1, c1: 1, r2: 1, c2: 1 };
    S.anchor = { r: 1, c: 1 };
    syncFilterUI();
    renderSheetTabs();
    await reloadGrid({ resetScroll: true });
    updateHash();
    toast("Worksheet added.");
  }

  async function duplicateSheetFlow() {
    if (!S.wb) return;
    var before = new Set(S.sheets.map(function (item) { return item.id; }));
    await flushSave();
    var result = await api("/workbooks/" + S.wb.id + "/ops", {
      method: "POST", body: { ops: [{ op: "duplicate_sheet", sheet: S.sheetId }], label: "Duplicate sheet" },
    });
    adoptWorkbook(result.workbook);
    setUndo(result.can_undo, result.can_redo);
    var added = S.sheets.filter(function (item) { return !before.has(item.id); })[0];
    if (added) S.sheetId = added.id;
    S.filters = [];
    S.filterDraft = [];
    S.sort = null;
    S.search = "";
    syncFilterUI();
    renderSheetTabs();
    await reloadGrid({ resetScroll: true });
    updateHash();
    toast("Worksheet duplicated.");
  }

  async function deleteSheetFlow() {
    if (!S.wb) return;
    if (S.sheets.length <= 1) { toast("A workbook needs at least one worksheet.", "error"); return; }
    var sheet = S.sheets.filter(function (item) { return item.id === S.sheetId; })[0];
    var confirmed = await confirmDialog({
      title: "Delete this worksheet?",
      text: "\"" + (sheet ? sheet.name : "Sheet") + "\" will be removed from this workbook. Use Undo to restore it; other workbooks are not affected.",
      button: "Delete worksheet",
    });
    if (!confirmed) return;
    await flushSave();
    var result = await api("/workbooks/" + S.wb.id + "/ops", {
      method: "POST", body: { ops: [{ op: "delete_sheet", sheet: S.sheetId }], label: "Delete sheet" },
    });
    adoptWorkbook(result.workbook);
    setUndo(result.can_undo, result.can_redo);
    S.sheetId = S.sheets[0].id;
    S.filters = [];
    S.filterDraft = [];
    S.sort = null;
    S.search = "";
    syncFilterUI();
    renderSheetTabs();
    await reloadGrid({ resetScroll: true });
    updateHash();
    toast("Worksheet deleted. Undo is available.");
  }

  async function selectSheet(id) {
    if (!S.wb || !S.sheets.some(function (sheet) { return sheet.id === Number(id); })) return;
    if (Number(id) === S.sheetId) return;
    await flushSave();
    S.sheetId = Number(id);
    S.filters = [];
    S.filterDraft = [];
    S.sort = null;
    S.search = "";
    S.active = { r: 1, c: 1 };
    S.sel = { r1: 1, c1: 1, r2: 1, c2: 1 };
    S.anchor = { r: 1, c: 1 };
    S.highlight = null;
    S.issueIndex = -1;
    syncFilterUI();
    renderSheetTabs();
    await reloadGrid({ resetScroll: true, resetSelection: true });
    updateHash();
  }

  /* -------------------------------------------------------------- filters */

  var FILTER_OPS = [
    ["contains", "contains"], ["not_contains", "does not contain"], ["equals", "equals"],
    ["not_equals", "does not equal"], ["starts", "starts with"], ["ends", "ends with"],
    ["gt", "greater than"], ["lt", "less than"], ["gte", "greater or equal"], ["lte", "less or equal"],
    ["between", "between"], ["blank", "is blank"], ["not_blank", "is not blank"],
    ["duplicates", "duplicate values"], ["unique", "unique values"],
  ];
  var VALUE_OPS = ["contains", "not_contains", "equals", "not_equals", "starts", "ends", "gt", "lt", "gte", "lte", "between"];

  function headerLabel(col) {
    var text = S.headers[String(col)] || "";
    return text ? letter(col) + " — " + text : "Column " + letter(col);
  }

  function refreshColumnChoices() {
    var sortCol = $("ws-sort-col");
    var previous = S.sort ? String(S.sort.col) : "";
    var options = '<option value="">No sorting</option>';
    for (var col = 1; col <= Math.min(S.colCount || 0, 500); col += 1) {
      options += '<option value="' + col + '">' + esc(headerLabel(col).slice(0, 40)) + "</option>";
    }
    sortCol.innerHTML = options;
    sortCol.value = previous;
    $("ws-sort-dir").value = S.sort ? S.sort.direction : "asc";
  }

  function syncFilterUI() {
    S.filterDraft = JSON.parse(JSON.stringify(S.filters));
    $("ws-sheet-search").value = S.search || "";
    refreshColumnChoices();
    renderFilterRows();
  }

  function renderFilterRows() {
    var host = $("ws-filter-rows");
    if (!S.filterDraft.length) {
      host.innerHTML = '<div class="ws-filter-empty">No filters yet. Add one to narrow the view without changing stored data.</div>';
      return;
    }
    host.innerHTML = S.filterDraft.map(function (filter, index) {
      var colOptions = "";
      for (var col = 1; col <= Math.min(S.colCount || 0, 500); col += 1) {
        colOptions += '<option value="' + col + '"' + (Number(filter.col) === col ? " selected" : "") + ">" + esc(headerLabel(col).slice(0, 40)) + "</option>";
      }
      var opOptions = FILTER_OPS.map(function (pair) {
        return '<option value="' + pair[0] + '"' + (filter.op === pair[0] ? " selected" : "") + ">" + esc(pair[1]) + "</option>";
      }).join("");
      var second = filter.op === "between"
        ? '<input class="ws-filter-value2" type="text" value="' + esc(filter.value2 || "") + '" placeholder="Upper bound">'
        : "";
      return '<div class="ws-filter-row" data-index="' + index + '">' +
        '<select class="ws-filter-col" aria-label="Filter column">' + colOptions + "</select>" +
        '<select class="ws-filter-op" aria-label="Filter operator">' + opOptions + "</select>" +
        '<input class="ws-filter-value" type="text" value="' + esc(filter.value || "") + '" placeholder="Value">' +
        second +
        '<button class="ws-icon-btn" type="button" data-filter-remove="' + index + '" title="Remove filter" aria-label="Remove filter"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="m6 6 12 12M18 6 6 18"/></svg></button>' +
        "</div>";
    }).join("");
  }

  async function applyFilterUI() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    var filters = [];
    for (var index = 0; index < S.filterDraft.length; index += 1) {
      var draft = S.filterDraft[index];
      var op = draft.op || "contains";
      var value = draft.value == null ? "" : String(draft.value);
      if (VALUE_OPS.indexOf(op) >= 0 && !value.trim()) {
        toast("Enter a value for the " + op.replace(/_/g, " ") + " filter.", "error");
        return;
      }
      var filter = { col: Number(draft.col), op: op, value: value };
      if (op === "between") filter.value2 = String(draft.value2 || "");
      filters.push(filter);
    }
    S.filters = filters;
    S.sort = $("ws-sort-col").value ? { col: Number($("ws-sort-col").value), direction: $("ws-sort-dir").value } : null;
    S.search = $("ws-sheet-search").value.trim();
    await reloadGrid({ resetScroll: true });
    toast(filters.length || S.sort || S.search ? "View updated. Stored data is unchanged." : "Filters cleared.");
  }

  async function clearFilters() {
    S.filters = [];
    S.filterDraft = [];
    S.sort = null;
    S.search = "";
    syncFilterUI();
    if (S.wb) await reloadGrid({ resetScroll: true });
    toast("Filters and sorting cleared.");
  }

  function toggleFilterPanel(show) {
    var panel = $("ws-filter-panel");
    var visible = show == null ? panel.hidden : show;
    panel.hidden = !visible;
    if (visible) {
      syncFilterUI();
      $("ws-sheet-search").focus();
    }
  }

  async function quickSort(direction) {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    S.sort = { col: S.active.c, direction: direction };
    $("ws-sort-col").value = String(S.active.c);
    $("ws-sort-dir").value = direction;
    await reloadGrid({ resetScroll: true });
    toast("Sorted by " + headerLabel(S.active.c) + " (" + (direction === "desc" ? "Z→A" : "A→Z") + ").");
  }

  function duplicateFilter() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    toggleFilterPanel(true);
    S.filterDraft = [{ col: S.active.c, op: "duplicates", value: "" }];
    renderFilterRows();
    toast("Duplicate filter ready for " + headerLabel(S.active.c) + ". Click Apply.");
  }

  /* ---------------------------------------------------------------- find */

  async function runFind(reset) {
    if (!S.wb) return;
    var text = $("ws-find-text").value.trim();
    var status = $("ws-find-status");
    if (!text) { status.textContent = "Type something to find."; return; }
    var key = text + "|" + S.sheetId + "|" + ($("ws-find-case").checked ? 1 : 0);
    if (reset || !S.findState || S.findKey !== key) {
      var params = new URLSearchParams({ q: text, sheet_id: String(S.sheetId), limit: "500" });
      var data = await api("/workbooks/" + S.wb.id + "/search?" + params.toString());
      S.findState = data.results || [];
      S.findKey = key;
      S.findIndex = -1;
    }
    if (!S.findState.length) { status.textContent = "No matches in this worksheet."; return; }
    S.findIndex = (S.findIndex + 1) % S.findState.length;
    var hit = S.findState[S.findIndex];
    status.textContent = (S.findIndex + 1) + " of " + S.findState.length + " · " + (hit.header || "Column " + letter(hit.col));
    await revealCell(hit.row, hit.col, hit.sheet_id);
  }

  async function replaceCurrent() {
    if (!S.wb) return;
    if (!S.findState || S.findIndex < 0 || !S.findState.length) { await runFind(true); if (!S.findState || !S.findState.length) return; }
    var hit = S.findState[S.findIndex];
    var replacement = $("ws-replace-text").value;
    commitValue(hit.row, hit.col, replacement);
    await flushSave();
    $("ws-find-status").textContent = "Replaced one value.";
    await reloadGrid({ keepScroll: true });
    runFind(false);
  }

  async function replaceAll() {
    if (!S.wb) return;
    var find = $("ws-find-text").value;
    if (!find) { toast("Type text to find first.", "error"); return; }
    await flushSave();
    var result = await api("/workbooks/" + S.wb.id + "/ops", {
      method: "POST",
      body: {
        ops: [{
          op: "replace", sheet: S.sheetId, find: find, replace: $("ws-replace-text").value,
          match_case: $("ws-find-case").checked, whole: $("ws-find-whole").checked,
          has_header: !$("ws-find-header").checked,
          range: [1, 1, S.rowCount, S.colCount],
        }],
        label: "Find & replace",
      },
    });
    adoptWorkbook(result.workbook);
    setUndo(result.can_undo, result.can_redo);
    S.lastSaved = new Date();
    await reloadGrid({ resetScroll: true });
    $("ws-find-status").textContent = "Replacement saved.";
    toast("Replace all completed. Undo is available.");
  }

  async function gotoCell() {
    var raw = $("ws-goto-input").value.trim();
    if (!raw || !S.wb) return;
    var match = raw.match(/^([A-Za-z]{1,3})?(\d+)?$/);
    if (!match || (!match[1] && !match[2])) { toast("Enter a cell like F42 or a row number.", "error"); return; }
    var col = match[1] ? columnNumber(match[1]) : S.active.c;
    var row = match[2] ? Number(match[2]) : S.active.r;
    if (row < 1 || col < 1) { toast("That cell reference is not valid.", "error"); return; }
    closeModal("ws-modal-goto");
    await revealCell(row, col, S.sheetId);
  }

  /* -------------------------------------------------------------- search */

  var globalSearch = debounce(async function () {
    var input = $("ws-global-search");
    var query = input.value.trim();
    if (query.length < 1) return;
    var seq = ++S.searchSeq;
    try {
      var data = await api("/search?q=" + encodeURIComponent(query));
      if (seq !== S.searchSeq) return;
      renderSearchResults(data, query);
    } catch (error) { if (seq === S.searchSeq) reportError(error); }
  }, 350);

  function renderSearchResults(data, query) {
    var results = data.results || [];
    var books = data.workbooks || [];
    $("ws-search-summary").textContent = results.length + " cell " + plural(results.length, "match", "matches") +
      (books.length ? " · " + books.length + " workbook name " + plural(books.length, "match", "matches") : "") + " for “" + query + "”.";
    var html = "";
    if (books.length) {
      html += '<div class="ws-search-group">Workbooks</div>';
      html += books.map(function (wb) {
        return '<button class="ws-search-row" type="button" data-search-wb="' + wb.id + '"><span class="ws-search-ref">FILE</span><div><strong>' + esc(wb.name) + "</strong><small>Open this saved workbook</small></div></button>";
      }).join("");
    }
    if (results.length) {
      html += '<div class="ws-search-group">Cells</div>';
      html += results.map(function (hit) {
        var value = hit.value == null ? "" : String(hit.value);
        return '<button class="ws-search-row" type="button" data-search-wb="' + hit.workbook_id + '" data-search-sheet="' + hit.sheet_id + '" data-search-row="' + hit.row + '" data-search-col="' + hit.col + '">' +
          '<span class="ws-search-ref">' + esc(hit.ref || "") + "</span>" +
          "<div><strong>" + esc(value.slice(0, 120)) + "</strong><small>" + esc(hit.workbook_name) + " › " + esc(hit.sheet_name) + " · " + esc(hit.header || "") + "</small></div>" +
          "</button>";
      }).join("");
    }
    if (!html) html = '<div class="ws-issue-empty">No matches. Try a shorter or differently spelled keyword.</div>';
    $("ws-search-results").innerHTML = html;
    openModal("ws-modal-search");
  }

  /* ------------------------------------------------------------ analysis */

  function markAnalysisStale() {
    if (!S.analysis) return;
    S.analysisStale = true;
    updateAnalysisSummary();
  }

  async function loadAnalysis() {
    if (!S.wb) return;
    var data = await api("/workbooks/" + S.wb.id + "/analysis");
    S.analysis = data.analysis;
    S.analysisStale = Boolean(data.analysis && data.analysis.stale);
    S.issueIndex = -1;
    if (S.analysis) { $("ws-analysis-panel").hidden = false; renderAnalysis(); }
  }

  async function runAnalysis() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    toast("Running the local cleaning rules…");
    var data = await api("/workbooks/" + S.wb.id + "/analysis", { method: "POST", body: { sheet_id: S.sheetId } });
    S.analysis = data.analysis;
    S.analysisStale = false;
    S.issueIndex = -1;
    $("ws-analysis-panel").hidden = false;
    renderAnalysis();
    var count = (S.analysis.issues || []).length;
    toast(count ? count + " " + plural(count, "finding") + " detected." : "No issues found in the analysed rows.");
    refreshLibrary();
  }

  function updateAnalysisSummary() {
    var summary = $("ws-analysis-summary");
    if (!S.analysis) { summary.textContent = "Run the analysis to inspect detected issues."; return; }
    var issues = S.analysis.issues || [];
    summary.textContent = issues.length + " " + plural(issues.length, "finding") + " in " +
      (S.analysis.sheet_name || "sheet") + " · " + (S.analysis.analyzed_rows || 0) + " rows analysed" +
      (S.analysisStale ? " · out of date, rerun for current data" : "");
  }

  function renderAnalysis() {
    updateAnalysisSummary();
    var host = $("ws-analysis-list");
    if (!S.analysis) { host.innerHTML = '<div class="ws-issue-empty">No stored analysis for this workbook.</div>'; return; }
    var issues = (S.analysis.issues || []).filter(function (issue) { return !issue.applied; });
    if (!issues.length) { host.innerHTML = '<div class="ws-issue-empty">No outstanding findings. Run the analysis again after editing cells.</div>'; return; }
    host.innerHTML = issues.slice(0, 300).map(function (issue, index) {
      var ref = issue.col ? letter(issue.col) + (issue.sheet_row || issue.row || "") : "row " + (issue.sheet_row || issue.row || "?");
      var canApply = issue.suggestion != null && issue.col && (issue.sheet_row || issue.row);
      return '<div class="ws-issue-row" data-issue="' + index + '" data-row="' + (issue.sheet_row || issue.row || "") + '" data-col="' + (issue.col || "") + '">' +
        '<span class="ws-severity ' + esc(issue.severity || "warning") + '"></span>' +
        '<span class="ws-issue-ref">' + esc(ref) + "</span>" +
        '<span class="ws-issue-body"><strong>' + esc(issue.error || issue.category || "Issue") + "</strong><small>" +
        esc(String(issue.original == null ? "" : issue.original).slice(0, 80)) +
        (issue.suggestion != null ? " → " + esc(String(issue.suggestion).slice(0, 80)) : "") +
        "</small></span>" +
        '<span class="ws-issue-actions">' +
        (canApply ? '<button class="ws-small-btn" type="button" data-apply-issue="' + index + '">Apply</button>' : "") +
        '<button class="ws-small-btn ghost" type="button" data-explain-issue="' + index + '">Why</button>' +
        "</span></div>";
    }).join("");
  }

  async function applyIssue(issue) {
    if (!S.wb || !issue) return;
    if (issue.suggestion == null) { toast("This finding has no safe automatic correction.", "error"); return; }
    var row = issue.sheet_row || issue.row;
    var col = issue.col;
    if (!row || !col) { toast("This finding is not attached to a cell.", "error"); return; }
    await flushSave();
    var result = await api("/workbooks/" + S.wb.id + "/analysis/apply", {
      method: "POST",
      body: { sheet_id: S.analysis.sheet_id, row: row, col: col, value: issue.suggestion },
    });
    adoptWorkbook(result.workbook);
    setUndo(result.can_undo, result.can_redo);
    S.lastSaved = new Date();
    issue.applied = true;
    S.analysisStale = true;
    renderAnalysis();
    if (Number(S.analysis.sheet_id) === S.sheetId) {
      await reloadGrid({ keepScroll: true });
      await revealCell(row, col, S.sheetId);
    }
    toast("Correction applied to " + refName(row, col) + ". Undo is available.");
  }

  function exportFindings() {
    if (!S.analysis) { toast("Run the cleaning analysis first.", "error"); return; }
    var issues = S.analysis.issues || [];
    var header = ["sheet", "row", "column", "issue", "original", "suggested", "confidence", "severity", "reason"];
    function csv(value) { return '"' + String(value == null ? "" : value).replace(/"/g, '""') + '"'; }
    var lines = [header.map(csv).join(",")];
    issues.forEach(function (issue) {
      lines.push([
        S.analysis.sheet_name || "", issue.sheet_row || issue.row || "", issue.column || "",
        issue.error || "", issue.original || "", issue.suggestion == null ? "" : issue.suggestion,
        issue.confidence == null ? "" : issue.confidence, issue.severity || "", issue.reason || "",
      ].map(csv).join(","));
    });
    var blob = new Blob(["\ufeff" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
    var anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = (S.wb ? S.wb.name.replace(/[^\w \-]/g, "_") : "datafix") + "-findings.csv";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(function () { URL.revokeObjectURL(anchor.href); }, 4000);
    toast("Findings CSV downloaded. The analysis also stays stored in the backend.");
  }

  /* -------------------------------------------------------------- export */

  function populateExportFormats() {
    var select = $("ws-export-format");
    if (!select) return;
    var current = select.value || "xlsx";
    var formats = S.formats || {};
    var keys = Object.keys(formats);
    if (!keys.length) keys = ["xlsx", "xlsm", "xls", "csv", "ods", "pdf", "docx", "json"];
    select.innerHTML = keys.map(function (key) {
      var label = formats[key] && formats[key].label ? formats[key].label : key.toUpperCase();
      return '<option value="' + esc(key) + '">' + esc(label) + "</option>";
    }).join("");
    select.value = keys.indexOf(current) >= 0 ? current : keys[0];
    updateExportNote();
  }

  function updateExportNote() {
    var select = $("ws-export-format");
    var note = $("ws-export-note");
    var info = (S.formats || {})[select.value];
    note.textContent = info ? info.limits : "Choose a format to see its limitations.";
  }

  async function openExport() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    await flushSave();
    populateExportFormats();
    updateExportNote();
    openModal("ws-modal-export");
    loadExportHistory();
  }

  async function loadExportHistory() {
    var host = $("ws-export-history");
    host.innerHTML = '<div class="ws-issue-empty">Loading previous exports…</div>';
    try {
      var data = await api("/exports?workbook_id=" + encodeURIComponent(S.wb ? S.wb.id : ""));
      var items = data.exports || [];
      if (!items.length) { host.innerHTML = '<div class="ws-issue-empty">No exports yet for this workbook.</div>'; return; }
      host.innerHTML = "<div class=\"ws-search-group\">Exported files</div>" + items.map(function (item) {
        return '<div class="ws-export-row"><span title="' + esc(item.filename) + '">' + esc(item.filename) + "</span><small>" + bytes(item.size) + " · " + esc(when(item.created_at)) + "</small><button class=\"ws-small-btn\" type=\"button\" data-download-export=\"" + item.id + "\">Download</button></div>";
      }).join("");
    } catch (error) { host.innerHTML = '<div class="ws-issue-empty">Export history could not be loaded.</div>'; }
  }

  function triggerDownload(url) {
    var anchor = document.createElement("a");
    anchor.href = url;
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  }

  async function runExport() {
    if (!S.wb) return;
    await flushSave();
    var format = $("ws-export-format").value;
    var scope = $("ws-export-scope").value;
    var body = { format: format };
    if (scope === "workbook") body.sheet_ids = S.sheets.map(function (sheet) { return sheet.id; });
    else body.sheet_ids = [S.sheetId];
    if (scope === "view") {
      body.filters = S.filters;
      body.sort = S.sort;
      body.search = S.search;
      body.include_hidden = true;
    }
    if (scope === "selected") {
      var rect = selectionRect();
      body.rows = S.rows.filter(function (row) { return row >= rect.r1 && row <= rect.r2; });
      body.columns = S.columns.filter(function (col) { return col >= rect.c1 && col <= rect.c2; });
      body.include_hidden = true;
    }
    var data = await api("/workbooks/" + S.wb.id + "/export", { method: "POST", body: body });
    triggerDownload(data.export.url);
    toast("Export created: " + data.export.filename + ". " + (data.export.notes || []).join(" "), "ok", 7000);
    await loadExportHistory();
  }

  function openPrint() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    var win = window.open("about:blank", "_blank");
    flushSave().then(function () {
      var params = new URLSearchParams({ sheet_id: String(S.sheetId) });
      if (S.filters.length) params.set("filters", JSON.stringify(S.filters));
      if (S.sort) params.set("sort", JSON.stringify(S.sort));
      if (S.search) params.set("search", S.search);
      var url = API + "/workbooks/" + S.wb.id + "/print?" + params.toString();
      if (win) win.location.href = url;
      else toast("Allow pop-ups to open the print preview.", "error");
    });
  }

  function downloadOriginal() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    if (!S.wb.original_name) { toast("This workbook was created in DataFix; it has no uploaded original file.", "error"); return; }
    triggerDownload(API + "/workbooks/" + S.wb.id + "/original");
    toast("Downloading the untouched original file.");
  }

  /* -------------------------------------------------------------- upload */

  function filterFiles(list) {
    return Array.prototype.slice.call(list || []).filter(function (file) {
      var dot = file.name.lastIndexOf(".");
      return dot >= 0 && EXTENSIONS.indexOf(file.name.slice(dot).toLowerCase()) >= 0;
    });
  }

  async function uploadFiles(fileList) {
    var files = filterFiles(fileList);
    if (!files.length) { toast("Choose XLSX, XLSM, XLS, CSV or ODS files.", "error"); return; }
    if (files.length > 20) { toast("Upload up to 20 files at a time.", "error"); return; }
    await flushSave();
    var modal = $("ws-modal-upload");
    var list = $("ws-upload-list");
    var bar = $("ws-upload-bar");
    var note = $("ws-upload-note");
    bar.style.width = "0%";
    note.textContent = "Uploading " + files.length + " " + plural(files.length, "file") + " to permanent storage. The originals are copied, never modified.";
    list.innerHTML = files.map(function (file) {
      return '<div class="ws-upload-row" data-name="' + esc(file.name) + '"><span>' + esc(file.name) + '</span><small>Waiting…</small></div>';
    }).join("");
    $("ws-upload-done").hidden = true;
    $("ws-upload-cancel").hidden = false;
    openModal("ws-modal-upload");

    var form = new FormData();
    files.forEach(function (file) { form.append("files", file, file.name); });
    activeXhr = new XMLHttpRequest();
    activeXhr.open("POST", API + "/upload");
    activeXhr.upload.onprogress = function (event) {
      if (event.lengthComputable) bar.style.width = Math.round(event.loaded / event.total * 100) + "%";
    };
    activeXhr.onload = function () {
      activeXhr = null;
      var data = {};
      try { data = JSON.parse(this.responseText || "{}"); } catch (error) { data = {}; }
      bar.style.width = "100%";
      var created = data.created || [];
      (created || []).forEach(function (workbook) {
        var row = qs('[data-name="' + cssEscape(workbook.name) + '"]', list);
        if (!row) {
          var match = qsa(".ws-upload-row", list).filter(function (item) {
            return item.dataset.name && workbook.original_name && item.dataset.name.indexOf(workbook.original_name.replace(/\.[^.]+$/, "")) === 0;
          })[0];
          row = match;
        }
        if (row) { row.classList.add("is-ok"); qs("small", row).textContent = "Imported"; }
      });
      (data.errors || []).forEach(function (error) {
        var row = qs('[data-name="' + cssEscape(error.file) + '"]', list);
        if (row) { row.classList.add("is-bad"); qs("small", row).textContent = error.error; }
      });
      var unmatched = qsa(".ws-upload-row", list).filter(function (item) { return !item.classList.contains("is-ok") && !item.classList.contains("is-bad"); });
      unmatched.forEach(function (row) {
        if (created.length) { row.classList.add("is-ok"); qs("small", row).textContent = "Imported"; }
        else { row.classList.add("is-bad"); qs("small", row).textContent = (data.error || "Not imported"); }
      });
      $("ws-upload-cancel").hidden = true;
      $("ws-upload-done").hidden = false;
      note.textContent = created.length + " imported" + (data.errors && data.errors.length ? ", " + data.errors.length + " failed. Existing workbooks were not changed." : ". The originals remain untouched in storage.");
      refreshLibrary().then(function () {
        if (created.length) {
          closeModal("ws-modal-upload");
          var first = created[0];
          var summary = first.analysis_summary;
          setPage("analysis", { wbId: first.id }).then(function () {
            var detail = summary
              ? " Analysis found " + summary.issues + " " + plural(summary.issues, "finding") + "."
              : " The analysis could not run automatically; use Analyse again.";
            toast("Imported " + created.length + " " + plural(created.length, "workbook") + "." + detail, "ok", 7000);
          }).catch(reportError);
        } else if (data.errors && data.errors.length) {
          toast(data.errors[0].error || "The file could not be imported.", "error");
        }
      });
    };
    activeXhr.onerror = function () {
      activeXhr = null;
      $("ws-upload-cancel").hidden = true;
      $("ws-upload-done").hidden = false;
      note.textContent = "The upload could not reach the backend. Your existing files were not changed.";
      toast("Upload failed. Check that the Python server is running.", "error");
    };
    activeXhr.onabort = function () {
      activeXhr = null;
      $("ws-upload-cancel").hidden = true;
      $("ws-upload-done").hidden = false;
      note.textContent = "Upload cancelled in the browser. A file that already finished arriving may still appear in the library.";
    };
    activeXhr.send(form);
  }

  function cssEscape(value) {
    if (window.CSS && window.CSS.escape) return window.CSS.escape(String(value));
    return String(value).replace(/["\\]/g, "\\$&");
  }

  async function importFromPath() {
    var path = $("ws-import-path").value.trim();
    if (!path) { toast("Enter a full file path first.", "error"); return; }
    var data = await api("/import-path", { method: "POST", body: { path: path } });
    await refreshLibrary();
    closeModal("ws-modal-import");
    await openWorkbook(data.workbook.id);
    toast("Imported an untouched copy of the file.");
  }

  async function nativePick(kind, target) {
    var data = await api("/dialog", { method: "POST", body: { kind: kind } });
    if (!data.cancelled && data.path) target.value = data.path;
  }

  /* -------------------------------------------------------------- storage */

  function updateStorageLabels() {
    var root = S.storage ? S.storage.root : "";
    var mini = $("ws-storage-mini");
    var status = $("ws-status-storage");
    if (mini) { mini.textContent = root || "Local · private"; mini.title = root || ""; }
    if (status) { status.textContent = root || "—"; status.title = root || "Storage root"; }
  }

  function renderStorageInfo() {
    var storage = S.storage || {};
    $("ws-storage-info").innerHTML =
      "<div><strong>Active storage folder</strong><span>" + esc(storage.root || "—") + "</span></div>" +
      "<div><strong>Default folder</strong><span>" + esc(storage.default || "—") + "</span></div>" +
      "<div><strong>Database</strong><span>" + esc(storage.database || "—") + "</span></div>" +
      "<div><strong>Status</strong><span>" + (storage.writable ? "Writable" : "Not writable") + " · originals, edits, exports, reports and backups are stored as real files.</span></div>";
    $("ws-storage-input").value = storage.root || "";
    $("ws-retention").value = (S.limits && S.limits.version_retention) || 120;
  }

  async function openStorage() {
    try { await loadState(); } catch (error) { /* keep last known */ }
    renderStorageInfo();
    openModal("ws-modal-storage");
  }

  async function saveStorage() {
    var root = $("ws-storage-input").value.trim();
    var data = await api("/storage", { method: "POST", body: { root: root } });
    S.storage = data.storage;
    updateStorageLabels();
    renderStorageInfo();
    toast("New workbooks will be stored in " + data.storage.root + ". Existing files stay where they are.");
  }

  async function resetStorage() {
    var data = await api("/storage", { method: "POST", body: { reset: true } });
    S.storage = data.storage;
    updateStorageLabels();
    renderStorageInfo();
    toast("Using the default storage folder again.");
  }

  async function saveRetention() {
    var value = Number($("ws-retention").value);
    var data = await api("/storage/retention", { method: "POST", body: { value: value } });
    S.limits.version_retention = data.version_retention;
    toast("Keeping up to " + data.version_retention + " versions per workbook.");
  }

  /* ----------------------------------------------------------- appearance */

  function renderAppearance() {
    var themeApi = window.DataFixTheme;
    if (!themeApi) return;
    var modeRow = $("ws-mode-row");
    var current = S.themeDraft;
    modeRow.innerHTML = [["system", "System"], ["light", "Light"], ["dark", "Dark"]].map(function (pair) {
      return '<button type="button" data-theme-mode="' + pair[0] + '"' + (current.appearance === pair[0] ? ' class="is-active"' : "") + ">" + pair[1] + "</button>";
    }).join("");
    var grid = $("ws-theme-grid");
    grid.innerHTML = themeApi.themes.map(function (theme) {
      return '<button class="ws-theme-card' + (current.theme === theme.id ? " is-active" : "") + '" type="button" data-theme-id="' + esc(theme.id) + '">' +
        '<span class="ws-theme-swatches"><span style="background:' + esc(theme.primary) + '"></span><span style="background:' + esc(theme.secondary) + '"></span><span style="background:' + esc(theme.accent) + '"></span><span style="background:' + esc(theme.card) + '"></span></span>' +
        "<strong>" + esc(theme.name) + "</strong></button>";
    }).join("");
  }

  function openAppearance() {
    if (!window.DataFixTheme) { toast("The theme manager is still loading. Try again.", "error"); return; }
    S.themeDraft = window.DataFixTheme.getPreferences();
    renderAppearance();
    openModal("ws-modal-appearance");
  }

  async function saveAppearance() {
    if (!window.DataFixTheme || !S.themeDraft) return;
    window.DataFixTheme.selectTheme(S.themeDraft.theme);
    window.DataFixTheme.selectMode(S.themeDraft.appearance);
    await window.DataFixTheme.save();
    closeModal("ws-modal-appearance");
  }

  async function toggleAppearance() {
    if (!window.DataFixTheme) return;
    var mode = document.documentElement.getAttribute("data-mode") === "dark" ? "light" : "dark";
    window.DataFixTheme.selectMode(mode);
    await window.DataFixTheme.save();
  }

  /* ------------------------------------------------------------ versions */

  async function openVersions() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    await flushSave();
    var data = await loadVersionState();
    renderVersions(data);
    openModal("ws-modal-versions");
  }

  function renderVersions(data) {
    var versions = data.versions || [];
    var host = $("ws-version-list");
    host.innerHTML = versions.length ? versions.map(function (version) {
      return '<div class="ws-version-row"><div><strong>v' + version.number + " · " + esc(version.label) + "</strong><small>" +
        esc(when(version.created_at)) + " · " + (version.cell_count || 0) + " cells · " + bytes(version.size) + "</small></div>" +
        '<button class="ws-small-btn" type="button" data-restore-version="' + version.id + '">Restore</button></div>';
    }).join("") : '<div class="ws-issue-empty">No full snapshots yet. One is created automatically during imports and checkpoints.</div>';
    var history = $("ws-batch-list");
    var items = data.history || [];
    history.innerHTML = items.length ? items.map(function (batch) {
      return '<div class="ws-batch-row' + (batch.undone ? " is-undone" : "") + '"><div><strong>' + esc(batch.label) + "</strong><small>" + esc(when(batch.created_at)) + (batch.undone ? " · undone" : "") + "</small></div></div>";
    }).join("") : '<div class="ws-issue-empty">No edit batches yet.</div>';
  }

  async function restoreVersion(versionId) {
    var confirmed = await confirmDialog({
      title: "Restore this version?",
      text: "The current workbook state is snapshotted first, then the selected version is restored. Nothing is lost and the restore itself is versioned.",
      button: "Restore version",
    });
    if (!confirmed) return;
    var data = await api("/workbooks/" + S.wb.id + "/versions/" + versionId + "/restore", { method: "POST" });
    adoptWorkbook(data.workbook);
    S.filters = [];
    S.filterDraft = [];
    S.sort = null;
    S.search = "";
    S.analysis = null;
    S.analysisStale = false;
    $("ws-analysis-panel").hidden = true;
    syncFilterUI();
    await reloadGrid({ resetScroll: true, resetSelection: true });
    closeModal("ws-modal-versions");
    S.lastSaved = new Date();
    updateSaveChip();
    toast("Version restored. A snapshot of the previous state was kept.");
    loadAnalysis().catch(function () {});
    refreshLibrary();
  }

  /* ------------------------------------------------------------- actions */

  async function runAction(action) {
    if (!action) return;
    var needsWorkbook = !["new-file", "upload-open", "import-path", "storage-settings", "toggle-appearance", "toggle-sidebar"].includes(action);
    if (needsWorkbook && !S.wb) { toast("Open or create a workbook first.", "error"); return; }
    switch (action) {
      case "new-file": {
        var name = await askName({ title: "New workbook", text: "The workbook is stored permanently in the backend with one worksheet.", value: "Untitled workbook", button: "Create workbook" });
        if (name != null) await createWorkbook(name.trim());
        break;
      }
      case "upload-open": $("ws-file-input").click(); break;
      case "import-path": $("ws-import-path").value = ""; openModal("ws-modal-import"); break;
      case "save-now": await saveNow(); break;
      case "save-as": await saveAsFlow(); break;
      case "checkpoint": {
        var label = await askName({ title: "Create checkpoint", text: "A labelled full snapshot is written to the backups folder. Versions are never deleted automatically beyond your retention setting.", value: "Checkpoint", button: "Create checkpoint" });
        if (label != null) await checkpointFlow(label.trim() || "Checkpoint");
        break;
      }
      case "export-open": await openExport(); break;
      case "print": openPrint(); break;
      case "download-original": downloadOriginal(); break;
      case "storage-settings": await openStorage(); break;
      case "close-workbook": await closeWorkbook(); break;
      case "undo": await doUndo(); break;
      case "redo": await doRedo(); break;
      case "cut": copySelection(true); break;
      case "copy": copySelection(false); break;
      case "paste": await pasteFromClipboard(); break;
      case "clear-cell": clearSelectionValues(); break;
      case "bold": toggleStyle("bold", "Bold"); break;
      case "italic": toggleStyle("italic", "Italic"); break;
      case "underline": toggleStyle("underline", "Underline"); break;
      case "align-left": setAlignment("left"); break;
      case "align-center": setAlignment("center"); break;
      case "align-right": setAlignment("right"); break;
      case "wrap": toggleStyle("wrap", "Wrap text"); break;
      case "clear-style": clearStyle(); break;
      case "insert-row": await structuralOp({ op: "insert_rows", sheet: S.sheetId, at: S.active.r, count: Math.max(1, selectionRect().r2 - selectionRect().r1 + 1) }, "Insert row(s)"); break;
      case "delete-row": {
        var rows = selectionRect();
        await structuralOp({ op: "delete_rows", sheet: S.sheetId, at: rows.r1, count: rows.r2 - rows.r1 + 1 }, "Delete row(s)", {
          title: "Delete rows?", text: "Delete " + (rows.r2 - rows.r1 + 1) + " row(s) starting at row " + rows.r1 + "? This is undoable.",
        });
        break;
      }
      case "insert-col": await structuralOp({ op: "insert_cols", sheet: S.sheetId, at: S.active.c, count: Math.max(1, selectionRect().c2 - selectionRect().c1 + 1) }, "Insert column(s)"); break;
      case "delete-col": {
        var cols = selectionRect();
        await structuralOp({ op: "delete_cols", sheet: S.sheetId, at: cols.c1, count: cols.c2 - cols.c1 + 1 }, "Delete column(s)", {
          title: "Delete columns?", text: "Delete " + (cols.c2 - cols.c1 + 1) + " column(s) starting at " + letter(cols.c1) + "? This is undoable.",
        });
        break;
      }
      case "find-replace": $("ws-find-text").value = S.search || ""; $("ws-find-status").textContent = ""; openModal("ws-modal-find"); break;
      case "goto": $("ws-goto-input").value = ""; openModal("ws-modal-goto"); break;
      case "add-sheet": await addSheetFlow(); break;
      case "duplicate-sheet": await duplicateSheetFlow(); break;
      case "rename-sheet": await renameSheetFlow(); break;
      case "delete-sheet": await deleteSheetFlow(); break;
      case "versions": await openVersions(); break;
      case "freeze-1": setLayout({ frozen_rows: 1 }, "Freeze first row"); break;
      case "freeze-3": setLayout({ frozen_rows: 3 }, "Freeze three rows"); break;
      case "freeze-col": setLayout({ frozen_cols: 1 }, "Freeze first column"); break;
      case "unfreeze": setLayout({ frozen_rows: 0, frozen_cols: 0 }, "Unfreeze panes"); break;
      case "hide-rows": {
        var hiddenRows = new Set(S.sheetMeta && S.sheetMeta.hidden_rows || []);
        var selected = selectionRect();
        for (var row = selected.r1; row <= selected.r2; row += 1) hiddenRows.add(row);
        setLayout({ hidden_rows: Array.from(hiddenRows) }, "Hide rows");
        break;
      }
      case "hide-cols": {
        var hiddenCols = new Set(S.sheetMeta && S.sheetMeta.hidden_cols || []);
        var selectedCols = selectionRect();
        for (var col = selectedCols.c1; col <= selectedCols.c2; col += 1) hiddenCols.add(col);
        setLayout({ hidden_cols: Array.from(hiddenCols) }, "Hide columns");
        break;
      }
      case "show-hidden": setLayout({ hidden_rows: [], hidden_cols: [] }, "Show hidden rows and columns"); break;
      case "grow-rows": setLayout({ row_count: Math.min(200000, S.rowCount + 50) }, "Add rows"); break;
      case "grow-cols": setLayout({ col_count: Math.min(500, S.colCount + 10) }, "Add columns"); break;
      case "show-formulas":
        S.showFormulas = !S.showFormulas;
        renderGrid();
        toast(S.showFormulas ? "Showing formulas instead of calculated values." : "Showing calculated values.");
        break;
      case "recalc": {
        await flushSave();
        var data = await api("/workbooks/" + S.wb.id + "/recalculate", { method: "POST" });
        adoptWorkbook(data.workbook);
        await reloadGrid({ keepScroll: true });
        toast("Formulas recalculated.");
        break;
      }
      case "toggle-filters": toggleFilterPanel(); break;
      case "sort-asc": await quickSort("asc"); break;
      case "sort-desc": await quickSort("desc"); break;
      case "clear-filters": await clearFilters(); break;
      case "duplicates-report": duplicateFilter(); break;
      case "search-sheet": toggleFilterPanel(true); $("ws-sheet-search").focus(); break;
      case "run-analysis": await runAnalysis(); break;
      case "cleaning-panel":
        $("ws-analysis-panel").hidden = false;
        if (!S.analysis) await runAnalysis(); else renderAnalysis();
        break;
      case "next-issue": await nextIssue(); break;
      case "analysis-export": exportFindings(); break;
      case "analysis-reset": {
        var confirmed = await confirmDialog({ title: "Clear the stored analysis?", text: "Only the saved findings are cleared. Cells in the workbook are not changed.", button: "Clear analysis", danger: false });
        if (!confirmed) return;
        await api("/workbooks/" + S.wb.id + "/analysis", { method: "DELETE" });
        S.analysis = null;
        S.analysisStale = false;
        $("ws-analysis-panel").hidden = true;
        toast("Stored analysis cleared.");
        break;
      }
      case "apply-safe": await applySafeSuggestion(); break;
      case "explain-issue": explainCurrentIssue(); break;
      case "toggle-sidebar": toggleSidebar(); break;
      case "toggle-gridlines":
        S.gridlines = !S.gridlines;
        $("ws-grid-shell").classList.toggle("no-gridlines", !S.gridlines);
        toast(S.gridlines ? "Gridlines on." : "Gridlines hidden.");
        break;
      case "fullscreen":
        if (!document.fullscreenElement) {
          if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(function () { toast("Full screen was blocked by the browser.", "error"); });
        } else if (document.exitFullscreen) {
          document.exitFullscreen();
        }
        break;
      case "zoom-90": setZoom(0.9); break;
      case "zoom-100": setZoom(1); break;
      case "zoom-110": setZoom(1.1); break;
      case "toggle-appearance": await toggleAppearance(); break;
      default: break;
    }
  }

  function setZoom(value) {
    S.zoom = value;
    $("ws-status-zoom").textContent = Math.round(value * 100) + "%";
    renderGrid();
    scrollActiveIntoView();
  }

  function toggleSidebar() {
    var app = document.querySelector(".ws-app");
    if (window.innerWidth <= 820) {
      var open = !app.classList.contains("sidebar-open");
      app.classList.toggle("sidebar-open", open);
      $("ws-sidebar-scrim").hidden = !open;
    } else {
      app.classList.toggle("sidebar-collapsed");
    }
  }

  async function doUndo() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    await flushSave();
    if (S.queue.length) { toast("Pending changes must be saved before undo.", "error"); return; }
    try {
      var data = await api("/workbooks/" + S.wb.id + "/undo", { method: "POST" });
      adoptWorkbook(data.result.workbook);
      setUndo(data.result.can_undo, data.result.can_redo);
      S.lastSaved = new Date();
      S.pending.clear();
      S.highlight = null;
      markAnalysisStale();
      await reloadGrid({ keepScroll: true });
      updateSaveChip();
      toast("Undid: " + data.result.label + ".");
    } catch (error) { toast(error.message || "Nothing to undo.", "error"); }
  }

  async function doRedo() {
    if (!S.wb) { toast("Open a workbook first.", "error"); return; }
    await flushSave();
    if (S.queue.length) { toast("Pending changes must be saved before redo.", "error"); return; }
    try {
      var data = await api("/workbooks/" + S.wb.id + "/redo", { method: "POST" });
      adoptWorkbook(data.result.workbook);
      setUndo(data.result.can_undo, data.result.can_redo);
      S.lastSaved = new Date();
      S.pending.clear();
      markAnalysisStale();
      await reloadGrid({ keepScroll: true });
      updateSaveChip();
      toast("Redid: " + data.result.label + ".");
    } catch (error) { toast(error.message || "Nothing to redo.", "error"); }
  }

  function visibleIssues() {
    return S.analysis ? (S.analysis.issues || []).filter(function (issue) { return !issue.applied; }) : [];
  }

  async function nextIssue() {
    if (!S.wb) return;
    if (!S.analysis) { await runAnalysis(); if (!S.analysis) return; }
    var issues = visibleIssues();
    if (!issues.length) { toast("No outstanding findings. Run the analysis again after edits."); return; }
    S.issueIndex = (S.issueIndex + 1) % issues.length;
    var issue = issues[S.issueIndex];
    $("ws-analysis-panel").hidden = false;
    renderAnalysis();
    var row = issue.sheet_row || issue.row;
    if (row && issue.col) await revealCell(row, issue.col, S.analysis.sheet_id);
    toast(issue.error || "Finding selected.");
  }

  async function applySafeSuggestion() {
    var issues = visibleIssues();
    if (!issues.length) { toast("No suggestions are available. Run the analysis first.", "error"); return; }
    var issue = issues[S.issueIndex >= 0 ? S.issueIndex : 0];
    var target = issues.filter(function (item) { return item.suggestion != null; })[0];
    if (!target) { toast("No finding has a safe automatic correction.", "error"); return; }
    await applyIssue(target);
  }

  function explainCurrentIssue() {
    var issues = visibleIssues();
    if (!issues.length) { toast("Run the analysis first.", "error"); return; }
    var issue = issues[S.issueIndex >= 0 ? S.issueIndex : 0];
    showInfo("Why this was flagged",
      (issue.error || "Detected issue") + " in " + (issue.column || "a column") + " at row " + (issue.sheet_row || issue.row || "?") + ".\n\n" +
      (issue.reason || "The local cleaning rules flagged this value for review.") +
      "\n\nOriginal: " + (issue.original == null ? "" : issue.original) +
      (issue.suggestion != null ? "\nSuggested: " + issue.suggestion : "\nNo automatic correction was suggested."));
  }

  /* ---------------------------------------------------------- interaction */

  function cellFromEvent(event) {
    var node = event.target.closest(".ws-cell");
    if (!node) return null;
    return { row: Number(node.dataset.r), col: Number(node.dataset.c) };
  }

  function bindEvents() {
    $("ws-new-file").addEventListener("click", function () { runAction("new-file").catch(reportError); });
    $("ws-new-create").addEventListener("click", function () {
      var value = $("ws-new-name").value;
      finishName(value.trim());
    });
    $("ws-new-name").addEventListener("keydown", function (event) {
      if (event.key === "Enter") { event.preventDefault(); $("ws-new-create").click(); }
    });

    $("ws-file-input").addEventListener("change", function () {
      if (this.files && this.files.length) uploadFiles(this.files).catch(reportError);
      this.value = "";
    });

    $("ws-file-search").addEventListener("input", debounce(async function () {
      S.fileQuery = this.value.trim();
      if (!S.fileQuery) { S.fileResults = null; renderSidebar(); return; }
      try {
        var data = await api("/workbooks?query=" + encodeURIComponent(S.fileQuery) + "&scope=" + (S.scope === "archived" ? "archived" : "all"));
        S.fileResults = data.workbooks || [];
        renderSidebar();
      } catch (error) { reportError(error); }
    }, 260));

    $("ws-file-list").addEventListener("click", function (event) {
      var star = event.target.closest("[data-star]");
      var archive = event.target.closest("[data-archive]");
      var open = event.target.closest("[data-open]");
      try {
        if (star) { toggleStar(star.dataset.star).catch(reportError); return; }
        if (archive) { toggleArchive(archive.dataset.archive).catch(reportError); return; }
        if (open) openWorkbook(open.dataset.open).catch(reportError);
      } catch (error) { reportError(error); }
    });

    $("ws-recent-mini").addEventListener("click", function (event) {
      var button = event.target.closest("[data-open]");
      if (button) openWorkbook(button.dataset.open).catch(reportError);
    });

    var homeSearchTimer = null, homeSearchSeq = 0;
    function runHomeSearch(value) {
      var host = $("ws-home-results");
      if (!host) return;
      if (!value) { host.innerHTML = ""; return; }
      clearTimeout(homeSearchTimer);
      homeSearchTimer = setTimeout(async function () {
        var seq = ++homeSearchSeq;
        try {
          var data = await api("/workbooks?scope=all&query=" + encodeURIComponent(value));
          if (seq !== homeSearchSeq || !$("ws-home-results")) return;
          $("ws-home-results").innerHTML = homeSearchRows(data.workbooks || [], value);
        } catch (error) {
          if (seq === homeSearchSeq) $("ws-home-results").innerHTML = '<div class="ws-empty-box">' + esc(error.message) + "</div>";
        }
      }, 280);
    }

    $("ws-pages").addEventListener("input", function (event) {
      if (event.target.id === "ws-home-search") runHomeSearch(event.target.value.trim());
    });
    $("ws-pages").addEventListener("change", function (event) {
      var target = event.target;
      if (target.id === "ws-report-wb") { setPage("analysis", { wbId: target.value }).catch(reportError); return; }
      if (target.id === "ws-select-all") {
        filteredFindings().forEach(function (issue) {
          if (target.checked) S.reportSelected[issue.id] = true;
          else delete S.reportSelected[issue.id];
        });
        qsa("[data-issue-select]", $("ws-pages")).forEach(function (box) { box.checked = target.checked; });
        return;
      }
      if (target.hasAttribute("data-issue-select")) {
        if (target.checked) S.reportSelected[target.dataset.issueSelect] = true;
        else delete S.reportSelected[target.dataset.issueSelect];
      }
    });
    $("ws-pages").addEventListener("click", function (event) {
      var menuButton = event.target.closest("[data-file-menu]");
      if (menuButton) {
        event.preventDefault();
        var trigger = menuButton.tagName === "TR"
          ? (menuButton.querySelector("[data-file-menu]") || menuButton) : menuButton;
        if (trigger.getAttribute("aria-expanded") === "true") closeFileMenu();
        else openFileMenu(trigger, trigger.dataset.fileMenu);
        return;
      }
      var fileButton = event.target.closest("[data-file-action]");
      if (fileButton) {
        event.preventDefault();
        if (fileButton.closest("#ws-file-menu")) closeFileMenu();
        handleFileAction(fileButton.dataset.fileAction, fileButton.dataset.id).catch(reportError);
        return;
      }
      var findingButton = event.target.closest("[data-finding-action]");
      if (findingButton) {
        event.preventDefault();
        var issue = findingById(findingButton.dataset.id);
        if (!issue) return;
        var action = findingButton.dataset.findingAction;
        if (action === "accept") applyFindings([issue]).catch(reportError);
        else if (action === "edit") editFinding(issue).catch(reportError);
        else if (action === "reject") setFindingsStatus([issue], "rejected").catch(reportError);
        else if (action === "reopen") setFindingsStatus([issue], "open").catch(reportError);
        else if (action === "view" && S.reportWbId) {
          openWorkbook(S.reportWbId, {
            sheetId: S.reportAnalysis ? S.reportAnalysis.sheet_id : null,
            cell: { row: issue.sheet_row || issue.row, col: issue.col },
          }).catch(reportError);
        }
        return;
      }
      var actionButton = event.target.closest("[data-page-action]");
      if (actionButton) {
        event.preventDefault();
        handlePageAction(actionButton.dataset.pageAction, actionButton).catch(reportError);
      }
    });

    qsa(".ws-nav-item[data-scope]").forEach(function (button) {
      button.addEventListener("click", function () {
        S.scope = button.dataset.scope;
        S.fileResults = null;
        S.fileQuery = "";
        $("ws-file-search").value = "";
        renderSidebar();
      });
    });
    qsa(".ws-page-item").forEach(function (button) {
      button.addEventListener("click", function () { setPage(button.dataset.page).catch(reportError); });
    });
    $("ws-history-nav").addEventListener("click", function () { openVersions().catch(reportError); });
    $("ws-refresh-files").addEventListener("click", function () { refreshLibrary().then(function () { toast("File list refreshed."); }); });
    $("ws-open-storage").addEventListener("click", function () { openStorage().catch(reportError); });
    $("ws-open-appearance").addEventListener("click", openAppearance);
    $("ws-sidebar-collapse").addEventListener("click", toggleSidebar);
    $("ws-mobile-menu").addEventListener("click", toggleSidebar);
    $("ws-sidebar-scrim").addEventListener("click", function () {
      document.querySelector(".ws-app").classList.remove("sidebar-open");
      this.hidden = true;
    });

    qsa(".ws-ribbon-tab").forEach(function (tab) {
      tab.addEventListener("click", function () {
        qsa(".ws-ribbon-tab").forEach(function (item) { item.classList.toggle("is-active", item === tab); });
        qsa(".ws-ribbon-panel").forEach(function (panel) {
          panel.hidden = panel.dataset.panel !== tab.dataset.tab;
        });
      });
    });

    document.addEventListener("click", function (event) {
      var disabled = event.target.closest("[data-disabled]");
      if (disabled) {
        event.preventDefault();
        toast(disabled.title || "This command is not implemented yet.");
        return;
      }
      var close = event.target.closest("[data-close]");
      if (close) {
        var modal = close.closest(".ws-modal");
        if (modal) closeModal(modal.id);
        return;
      }
      var action = event.target.closest("[data-action]");
      if (!action) return;
      event.preventDefault();
      runAction(action.dataset.action).catch(reportError);
    });

    document.addEventListener("click", function (event) {
      if (!event.target || !event.target.closest) return;
      if (event.target.closest("#ws-file-menu") || event.target.closest("[data-file-menu]")) return;
      closeFileMenu();
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") closeFileMenu();
    });
    window.addEventListener("resize", closeFileMenu);
    $("ws-pages").addEventListener("scroll", closeFileMenu);

    MODAL_IDS.forEach(function (id) {
      var modal = $(id);
      modal.addEventListener("mousedown", function (event) {
        if (event.target !== modal) return;
        if (id === "ws-modal-upload" && activeXhr) return;
        closeModal(id);
      });
    });

    $("ws-confirm-cancel").addEventListener("click", function () { settleConfirm(false); });
    $("ws-confirm-ok").addEventListener("click", function () {
      if (confirmMode === "info") { $("ws-modal-confirm").hidden = true; }
      else settleConfirm(true);
    });

    $("ws-star-btn").addEventListener("click", function () { toggleStarCurrent().catch(reportError); });
    $("ws-title").addEventListener("dblclick", function () { renameWorkbookFlow().catch(reportError); });
    $("ws-save-status").addEventListener("click", function () {
      if (S.save.error) { S.save.error = ""; flushSave().catch(reportError); }
    });
    $("ws-undo").addEventListener("click", function () { doUndo().catch(reportError); });
    $("ws-redo").addEventListener("click", function () { doRedo().catch(reportError); });
    $("ws-export-open").addEventListener("click", function () { openExport().catch(reportError); });

    $("ws-sheet-tabs").addEventListener("click", function (event) {
      var tab = event.target.closest("[data-sheet]");
      if (tab) selectSheet(tab.dataset.sheet).catch(reportError);
    });
    $("ws-sheet-tabs").addEventListener("dblclick", function (event) {
      var tab = event.target.closest("[data-sheet]");
      if (tab) renameSheetFlow().catch(reportError);
    });
    $("ws-sheet-add").addEventListener("click", function () { addSheetFlow().catch(reportError); });

    var viewport = $("ws-grid-viewport");
    viewport.addEventListener("scroll", function () {
      $("ws-colheads-inner").style.transform = "translateX(" + (-viewport.scrollLeft) + "px)";
      $("ws-rowheads-inner").style.transform = "translateY(" + (-viewport.scrollTop) + "px)";
      scheduleRender();
      if (viewport.scrollTop + viewport.clientHeight >= viewport.scrollHeight - 420 && S.rows.length < S.totalRows) loadMoreRows().catch(reportError);
      var lastCol = S.columns[S.columns.length - 1] || 0;
      if (viewport.scrollLeft + viewport.clientWidth >= viewport.scrollWidth - 320 && lastCol < S.colCount) loadMoreCols().catch(reportError);
    });
    window.addEventListener("resize", scheduleRender);

    $("ws-grid-cells").addEventListener("mousedown", function (event) {
      if (event.button !== 0) return;
      var cell = cellFromEvent(event);
      if (!cell) return;
      if (S.editing) commitEdit(null);
      S.dragging = true;
      selectCell(cell.row, cell.col, event.shiftKey);
      viewport.focus({ preventScroll: true });
      event.preventDefault();
    });
    document.addEventListener("mousemove", function (event) {
      if (!S.dragging) return;
      var node = document.elementFromPoint(event.clientX, event.clientY);
      var cellNode = node && node.closest ? node.closest(".ws-cell") : null;
      if (!cellNode) return;
      var row = Number(cellNode.dataset.r), col = Number(cellNode.dataset.c);
      if (row !== S.sel.r2 || col !== S.sel.c2) {
        S.sel.r2 = row;
        S.sel.c2 = col;
        S.active = { r: row, c: col };
        syncFormulaBar();
        renderGrid();
        updateStatusBar();
      }
    });
    document.addEventListener("mouseup", function () { S.dragging = false; });

    $("ws-grid-cells").addEventListener("dblclick", function (event) {
      var cell = cellFromEvent(event);
      if (cell) { selectCell(cell.row, cell.col, false); startEdit(null); }
    });

    viewport.addEventListener("keydown", function (event) {
      if (S.editing) return;
      var handled = true;
      if (event.key === "ArrowDown") moveSelection(1, 0, event.shiftKey);
      else if (event.key === "ArrowUp") moveSelection(-1, 0, event.shiftKey);
      else if (event.key === "ArrowRight") moveSelection(0, 1, event.shiftKey);
      else if (event.key === "ArrowLeft") moveSelection(0, -1, event.shiftKey);
      else if (event.key === "Tab") moveSelection(0, event.shiftKey ? -1 : 1, false);
      else if (event.key === "Enter") moveSelection(1, 0, false);
      else if (event.key === "F2") startEdit(null);
      else if (event.key === "Delete" || event.key === "Backspace") clearSelectionValues();
      else if (event.key === "Home") {
        if (event.ctrlKey) selectCell(S.rows[0] || 1, S.columns[0] || 1, false);
        else selectCell(S.active.r, S.columns[0] || 1, event.shiftKey);
        scrollActiveIntoView();
      } else if (event.key === "End") {
        selectCell(S.active.r, S.columns[S.columns.length - 1] || 1, event.shiftKey);
        scrollActiveIntoView();
      } else if (event.key === "PageDown") {
        var step = Math.max(3, Math.floor(viewport.clientHeight / Math.max(1, rowHeight(S.active.r))) - 2);
        moveSelection(step, 0, event.shiftKey);
      } else if (event.key === "PageUp") {
        var back = Math.max(3, Math.floor(viewport.clientHeight / Math.max(1, rowHeight(S.active.r))) - 2);
        moveSelection(-back, 0, event.shiftKey);
      } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
        startEdit(event.key);
      } else handled = false;
      if (handled) event.preventDefault();
    });

    var editorInput = $("ws-cell-editor-input");
    editorInput.addEventListener("keydown", function (event) {
      if (event.key === "Enter") { event.preventDefault(); commitEdit("down"); }
      else if (event.key === "Tab") { event.preventDefault(); commitEdit(event.shiftKey ? "left" : "right"); }
      else if (event.key === "Escape") { event.preventDefault(); cancelEdit(); }
      else if (event.key === "ArrowDown") { event.preventDefault(); commitEdit("down"); }
      else if (event.key === "ArrowUp") { event.preventDefault(); commitEdit("up"); }
    });
    editorInput.addEventListener("blur", function () { if (S.editing) commitEdit(null); });

    viewport.addEventListener("paste", function (event) {
      if (S.editing) return;
      var text = event.clipboardData && event.clipboardData.getData("text");
      if (text) { event.preventDefault(); pasteText(text); }
    });

    $("ws-formula-commit").addEventListener("click", function () { commitFormulaBar(); $("ws-grid-viewport").focus({ preventScroll: true }); });
    $("ws-formula-input").addEventListener("keydown", function (event) {
      if (event.key === "Enter") { event.preventDefault(); commitFormulaBar(); }
      else if (event.key === "Escape") { event.preventDefault(); syncFormulaBar(); $("ws-grid-viewport").focus({ preventScroll: true }); }
    });
    $("ws-name-box").addEventListener("keydown", function (event) {
      if (event.key !== "Enter") return;
      event.preventDefault();
      $("ws-goto-input").value = this.value;
      gotoCell().catch(reportError);
    });

    $("ws-text-color").addEventListener("change", function () { applyStyle({ color: this.value }, "Text colour"); });
    $("ws-fill-color").addEventListener("change", function () { applyStyle({ fill: this.value }, "Fill colour"); });

    $("ws-function-select").addEventListener("change", function () {
      var fn = this.value;
      if (!fn) return;
      this.value = "";
      var formula = $("ws-formula-input");
      formula.value = "=" + fn + "(";
      formula.focus();
      var end = formula.value.length;
      formula.setSelectionRange(end, end);
    });

    $("ws-filter-add").addEventListener("click", function () {
      if (!S.wb) { toast("Open a workbook first.", "error"); return; }
      S.filterDraft.push({ col: S.active.c || 2, op: "contains", value: "" });
      renderFilterRows();
    });
    $("ws-filter-apply").addEventListener("click", function () { applyFilterUI().catch(reportError); });
    $("ws-filter-clear").addEventListener("click", function () { clearFilters().catch(reportError); });
    $("ws-filter-rows").addEventListener("input", function (event) {
      var row = event.target.closest("[data-index]");
      if (!row) return;
      var index = Number(row.dataset.index);
      if (!S.filterDraft[index]) return;
      if (event.target.classList.contains("ws-filter-value")) S.filterDraft[index].value = event.target.value;
      if (event.target.classList.contains("ws-filter-value2")) S.filterDraft[index].value2 = event.target.value;
    });
    $("ws-filter-rows").addEventListener("change", function (event) {
      var row = event.target.closest("[data-index]");
      if (!row) return;
      var index = Number(row.dataset.index);
      if (!S.filterDraft[index]) return;
      if (event.target.classList.contains("ws-filter-col")) S.filterDraft[index].col = Number(event.target.value);
      if (event.target.classList.contains("ws-filter-op")) { S.filterDraft[index].op = event.target.value; renderFilterRows(); }
    });
    $("ws-filter-rows").addEventListener("click", function (event) {
      var remove = event.target.closest("[data-filter-remove]");
      if (!remove) return;
      S.filterDraft.splice(Number(remove.dataset.filterRemove), 1);
      renderFilterRows();
    });
    $("ws-sheet-search").addEventListener("keydown", function (event) {
      if (event.key === "Enter") applyFilterUI().catch(reportError);
    });

    $("ws-grid-corner").addEventListener("click", function () {
      if (!S.wb || !S.rows.length || !S.columns.length) return;
      S.anchor = { r: S.rows[0], c: S.columns[0] };
      S.active = { r: S.rows[0], c: S.columns[0] };
      S.sel = { r1: S.rows[0], c1: S.columns[0], r2: S.rows[S.rows.length - 1], c2: S.columns[S.columns.length - 1] };
      syncFormulaBar(); renderGrid(); updateStatusBar();
    });

    $("ws-colheads-inner").addEventListener("click", function (event) {
      if (event.target.closest(".ws-col-resize")) return;
      var head = event.target.closest("[data-col]");
      if (!head || !S.wb || !S.rows.length) return;
      var col = Number(head.dataset.col);
      var row = S.rows[0];
      if (event.shiftKey) { S.sel = { r1: S.sel.r1, c1: S.sel.c1, r2: S.rows[S.rows.length - 1], c2: col }; S.active = { r: row, c: col }; }
      else { S.anchor = { r: row, c: col }; S.active = { r: row, c: col }; S.sel = { r1: row, c1: col, r2: S.rows[S.rows.length - 1], c2: col }; }
      syncFormulaBar(); renderGrid(); updateStatusBar();
    });
    $("ws-rowheads-inner").addEventListener("click", function (event) {
      if (event.target.closest(".ws-row-resize")) return;
      var head = event.target.closest("[data-row]");
      if (!head || !S.wb || !S.columns.length) return;
      var row = Number(head.dataset.row);
      var col = S.columns[0];
      if (event.shiftKey) { S.sel = { r1: S.sel.r1, c1: S.sel.c1, r2: row, c2: S.columns[S.columns.length - 1] }; S.active = { r: row, c: col }; }
      else { S.anchor = { r: row, c: col }; S.active = { r: row, c: col }; S.sel = { r1: row, c1: col, r2: row, c2: S.columns[S.columns.length - 1] }; }
      syncFormulaBar(); renderGrid(); updateStatusBar();
    });

    document.addEventListener("mousedown", function (event) {
      var colHandle = event.target.closest("[data-resize-col]");
      var rowHandle = event.target.closest("[data-resize-row]");
      if (colHandle) {
        event.preventDefault();
        var col = Number(colHandle.dataset.resizeCol);
        var startX = event.clientX, startWidth = colWidth(col);
        var move = function (moveEvent) {
          S.colWidths.set(col, clamp(startWidth + (moveEvent.clientX - startX) / S.zoom, 40, 500));
          renderGrid();
        };
        var up = function () { document.removeEventListener("mousemove", move); document.removeEventListener("mouseup", up); };
        document.addEventListener("mousemove", move);
        document.addEventListener("mouseup", up);
      }
      if (rowHandle) {
        event.preventDefault();
        var row = Number(rowHandle.dataset.resizeRow);
        var startY = event.clientY, startHeight = rowHeight(row);
        var moveRow = function (moveEvent) {
          S.rowHeights.set(row, clamp(startHeight + (moveEvent.clientY - startY) / S.zoom, 16, 200));
          renderGrid();
        };
        var upRow = function () { document.removeEventListener("mousemove", moveRow); document.removeEventListener("mouseup", upRow); };
        document.addEventListener("mousemove", moveRow);
        document.addEventListener("mouseup", upRow);
      }
    });

    $("ws-global-search").addEventListener("input", globalSearch);
    $("ws-global-search").addEventListener("keydown", function (event) {
      if (event.key === "Enter") { event.preventDefault(); globalSearch(); }
    });
    $("ws-search-results").addEventListener("click", function (event) {
      var row = event.target.closest("[data-search-wb]");
      if (!row) return;
      closeModal("ws-modal-search");
      var options = {};
      if (row.dataset.searchSheet) options.sheetId = Number(row.dataset.searchSheet);
      if (row.dataset.searchRow) options.cell = { row: Number(row.dataset.searchRow), col: Number(row.dataset.searchCol) };
      openWorkbook(row.dataset.searchWb, options).catch(reportError);
    });

    $("ws-analysis-run").addEventListener("click", function () { runAnalysis().catch(reportError); });
    $("ws-analysis-close").addEventListener("click", function () { $("ws-analysis-panel").hidden = true; });
    $("ws-analysis-list").addEventListener("click", function (event) {
      var apply = event.target.closest("[data-apply-issue]");
      var explain = event.target.closest("[data-explain-issue]");
      var issues = visibleIssues();
      if (apply) { applyIssue(issues[Number(apply.dataset.applyIssue)]).catch(reportError); return; }
      if (explain) {
        var issue = issues[Number(explain.dataset.explainIssue)];
        if (issue) {
          S.issueIndex = Number(explain.dataset.explainIssue);
          explainCurrentIssue();
        }
        return;
      }
      var row = event.target.closest("[data-issue]");
      if (row) {
        var item = issues[Number(row.dataset.issue)];
        if (item && (item.sheet_row || item.row) && item.col) revealCell(item.sheet_row || item.row, item.col, S.analysis.sheet_id).catch(reportError);
      }
    });

    $("ws-find-next").addEventListener("click", function () { runFind(false).catch(reportError); });
    $("ws-replace-one").addEventListener("click", function () { replaceCurrent().catch(reportError); });
    $("ws-replace-all").addEventListener("click", function () { replaceAll().catch(reportError); });
    $("ws-find-text").addEventListener("keydown", function (event) {
      if (event.key === "Enter") { event.preventDefault(); runFind(true).catch(reportError); }
    });
    $("ws-goto-run").addEventListener("click", function () { gotoCell().catch(reportError); });
    $("ws-goto-input").addEventListener("keydown", function (event) {
      if (event.key === "Enter") { event.preventDefault(); gotoCell().catch(reportError); }
    });

    $("ws-import-browse").addEventListener("click", function () {
      var button = this;
      button.disabled = true;
      nativePick("file", $("ws-import-path")).catch(function (error) {
        toast(error.message || "The picker could not open; type the path instead.", "error");
      }).finally(function () { button.disabled = false; });
    });
    $("ws-import-run").addEventListener("click", function () { importFromPath().catch(reportError); });
    $("ws-import-path").addEventListener("keydown", function (event) {
      if (event.key === "Enter") { event.preventDefault(); importFromPath().catch(reportError); }
    });

    $("ws-storage-browse").addEventListener("click", function () {
      var button = this;
      button.disabled = true;
      nativePick("folder", $("ws-storage-input")).catch(function (error) {
        toast(error.message || "The folder picker could not open; type the path instead.", "error");
      }).finally(function () { button.disabled = false; });
    });
    $("ws-storage-save").addEventListener("click", function () { saveStorage().catch(reportError); });
    $("ws-storage-reset").addEventListener("click", function () { resetStorage().catch(reportError); });
    $("ws-retention-save").addEventListener("click", function () { saveRetention().catch(reportError); });

    $("ws-export-format").addEventListener("change", updateExportNote);
    $("ws-export-run").addEventListener("click", function () { runExport().catch(reportError); });
    $("ws-export-history").addEventListener("click", function (event) {
      var button = event.target.closest("[data-download-export]");
      if (button) triggerDownload(API + "/export/" + button.dataset.downloadExport + "/download");
    });

    $("ws-checkpoint-create").addEventListener("click", function () {
      checkpointFlow($("ws-checkpoint-label").value.trim() || "Manual checkpoint").then(function () {
        $("ws-checkpoint-label").value = "";
        openVersions().catch(reportError);
      }).catch(reportError);
    });
    $("ws-version-list").addEventListener("click", function (event) {
      var button = event.target.closest("[data-restore-version]");
      if (button) restoreVersion(Number(button.dataset.restoreVersion)).catch(reportError);
    });

    $("ws-mode-row").addEventListener("click", function (event) {
      var button = event.target.closest("[data-theme-mode]");
      if (!button || !S.themeDraft) return;
      S.themeDraft.appearance = button.dataset.themeMode;
      renderAppearance();
    });
    $("ws-theme-grid").addEventListener("click", function (event) {
      var card = event.target.closest("[data-theme-id]");
      if (!card || !S.themeDraft) return;
      S.themeDraft.theme = card.dataset.themeId;
      renderAppearance();
    });
    $("ws-theme-save").addEventListener("click", function () { saveAppearance().catch(reportError); });
    $("ws-theme-reset").addEventListener("click", function () {
      S.themeDraft = Object.assign({}, window.DataFixTheme.defaults);
      renderAppearance();
    });

    $("ws-upload-cancel").addEventListener("click", function () {
      if (activeXhr) { activeXhr.abort(); }
      else closeModal("ws-modal-upload");
    });
    $("ws-upload-done").addEventListener("click", function () { closeModal("ws-modal-upload"); });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        if (S.editing) { cancelEdit(); event.preventDefault(); return; }
        var open = lastOpenModal();
        if (open) {
          if (open === "ws-modal-upload" && activeXhr) return;
          closeModal(open);
          event.preventDefault();
        }
        return;
      }
      var tag = (event.target.tagName || "").toLowerCase();
      var typing = tag === "input" || tag === "textarea" || tag === "select" || event.target.isContentEditable;
      var mod = event.ctrlKey || event.metaKey;
      if (!mod) {
        if (event.key === "/" && !typing && !anyModalOpen()) {
          event.preventDefault();
          $("ws-global-search").focus();
        }
        return;
      }
      var key = String(event.key).toLowerCase();
      if (typing) return;
      if (key === "z" && !event.shiftKey) { event.preventDefault(); doUndo().catch(reportError); }
      else if (key === "y" || (key === "z" && event.shiftKey)) { event.preventDefault(); doRedo().catch(reportError); }
      else if (key === "s") { event.preventDefault(); saveNow().catch(reportError); }
      else if (key === "f") { event.preventDefault(); $("ws-find-text").value = S.search || ""; openModal("ws-modal-find"); }
      else if (key === "c") { if (!anyModalOpen()) { event.preventDefault(); copySelection(false); } }
      else if (key === "x") { if (!anyModalOpen()) { event.preventDefault(); copySelection(true); } }
      else if (key === "v") { if (!anyModalOpen()) { event.preventDefault(); pasteFromClipboard().catch(reportError); } }
    });

    window.addEventListener("dragover", function (event) {
      if (!event.dataTransfer || Array.prototype.indexOf.call(event.dataTransfer.types || [], "Files") < 0) return;
      event.preventDefault();
      $("ws-drop-overlay").hidden = false;
      dragDepth += 1;
    });
    window.addEventListener("dragleave", function (event) {
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) $("ws-drop-overlay").hidden = true;
    });
    window.addEventListener("drop", function (event) {
      event.preventDefault();
      dragDepth = 0;
      $("ws-drop-overlay").hidden = true;
      if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length) {
        uploadFiles(event.dataTransfer.files).catch(reportError);
      }
    });

    window.addEventListener("beforeunload", function (event) {
      if (S.queue.length || flushing) {
        event.preventDefault();
        event.returnValue = "Changes are still saving. Stay on this page until the save completes.";
      }
    });
    window.addEventListener("online", function () { if (S.queue.length) { S.save.error = ""; flushSave().catch(reportError); } });
    window.addEventListener("offline", function () { if (S.queue.length) toast("You are offline. Changes stay queued in this tab until the backend reconnects.", "error"); });

    window.addEventListener("datafix:toast", function (event) {
      var detail = event.detail || {};
      toast(detail.message || "", detail.kind === "bad" ? "error" : "ok");
    });

    window.addEventListener("hashchange", function () {
      if (!S.wb) return;
      var match = window.location.hash.match(/^#\/w\/([a-f0-9]{32})(?:\/(\d+))?/);
      if (!match) return;
      if (match[1] !== S.wb.id) { openWorkbook(match[1], { sheetId: match[2] ? Number(match[2]) : null }).catch(reportError); return; }
      if (match[2] && Number(match[2]) !== S.sheetId) selectSheet(Number(match[2])).catch(reportError);
    });
  }

  /* ------------------------------------------------------------------ boot */

  async function boot() {
    S.maxMb = Number(qs('meta[name="datafix-max-mb"]') && qs('meta[name="datafix-max-mb"]').content) || 0;
    bindEvents();
    updateHeader();
    renderSheetTabs();
    updateSaveChip();
    setUndo(false, false);
    setZoom(1);
    try {
      await loadState();
    } catch (error) {
      toast("The backend could not be reached. Start the Python server and refresh.", "error", 9000);
      console.error(error);
      return;
    }
    var match = window.location.hash.match(/^#\/w\/([a-f0-9]{32})(?:\/(\d+))?/);
    if (match) {
      try { await openWorkbook(match[1], { sheetId: match[2] ? Number(match[2]) : null }); }
      catch (error) {
        toast("That saved workbook could not be reopened.", "error");
        renderSidebar();
        await setPage("home");
      }
    } else {
      await setPage("home");
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.DataFixWorkspace = {
    state: S, api: api, openWorkbook: openWorkbook, reloadGrid: reloadGrid, revealCell: revealCell,
    runAction: runAction, flushSave: flushSave, applyIssue: applyIssue,
  };
})();

==============================================================================
FILE: templates/index.html
==============================================================================
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DataFix — Smart Excel Data Validation &amp; Correction</title>
  <meta name="description" content="Upload a messy Excel file, let DataFix find and fix data-entry mistakes, review every correction, and download a clean workbook.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap" rel="stylesheet">
  <meta name="datafix-max-mb" content="{{ max_mb }}">
  <meta name="theme-color" content="#0a0713">
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%237c6cff'/%3E%3Cg fill='white'%3E%3Crect x='15' y='34' width='6' height='15' rx='3'/%3E%3Crect x='25' y='24' width='6' height='25' rx='3'/%3E%3Crect x='35' y='30' width='6' height='19' rx='3'/%3E%3Crect x='45' y='17' width='6' height='32' rx='3'/%3E%3C/g%3E%3C/svg%3E">
  <script>window.DATAFIX_INITIAL_PREFS = {{ initial_prefs|tojson }};</script>
  <script src="/static/theme.js"></script>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
<div class="backdrop" aria-hidden="true">
  <span class="orb orb-a"></span>
  <span class="orb orb-b"></span>
  <span class="grid-lines"></span>
</div>

<div class="app">
  <aside class="sidebar">
    <a class="brand" href="/" aria-label="DataFix home">
      <span class="brand-mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
          <path d="M5 19V11M12 19V5M19 19v-5"/>
        </svg>
      </span>
      <span class="brand-text"><strong>DataFix</strong><small>Smart Excel care</small></span>
    </a>

    <nav class="nav" aria-label="Dashboard sections">
      <button class="nav-item" data-panel="overview" title="Overview" aria-label="Overview" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3.5" y="3.5" width="7" height="7" rx="2"/><rect x="13.5" y="3.5" width="7" height="7" rx="2"/><rect x="3.5" y="13.5" width="7" height="7" rx="2"/><rect x="13.5" y="13.5" width="7" height="7" rx="2"/></svg>
        <span class="nav-label">Overview</span>
      </button>
      <button class="nav-item" data-panel="review" title="Review &amp; compare" aria-label="Review and compare" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M3.5 9.5h17M9.5 9.5v10"/></svg>
        <span class="nav-label">Review &amp; compare</span>
      </button>
      <button class="nav-item" data-panel="issues" title="Corrections" aria-label="Corrections" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3.8l8 4.6v7.2l-8 4.6-8-4.6V8.4l8-4.6z"/><path d="M12 8.2v4.6M12 16h.01"/></svg>
        <span class="nav-label">Corrections</span>
      </button>
      <button class="nav-item" data-panel="downloads" title="Downloads" aria-label="Downloads" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 4v11M7.8 11l4.2 4.2L16.2 11"/><path d="M5 19h14"/></svg>
        <span class="nav-label">Downloads</span>
      </button>
      <button class="nav-item nav-settings" data-panel="appearance" title="Theme settings" aria-label="Theme settings">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3.5a8.5 8.5 0 1 0 0 17c1.4 0 2-.9 2-1.9 0-1.2-.9-1.7-.9-2.7 0-.9.7-1.4 1.7-1.4h1.7a3.9 3.9 0 0 0 3.9-3.9C20.4 6.4 16.8 3.5 12 3.5z"/><circle cx="7.7" cy="10.2" r="1.1" fill="currentColor" stroke="none"/><circle cx="11.4" cy="7.4" r="1.1" fill="currentColor" stroke="none"/><circle cx="15.6" cy="8.6" r="1.1" fill="currentColor" stroke="none"/></svg>
        <span class="nav-label">Appearance</span>
      </button>
    </nav>

    <div class="side-card">
      <span class="side-label">Current file</span>
      <div class="side-file" id="side-file">Nothing uploaded yet</div>
      <div class="side-meta" id="side-meta">Upload a workbook to begin.</div>
    </div>

    <div class="side-tip">
      <span class="side-tip-dot"></span>
      Your original file is never modified.
    </div>

    <button class="btn ghost block" id="new-file-btn" type="button">＋ New file</button>
    <div class="side-foot">Local · private · v1.0</div>
  </aside>

  <main class="main">
    <header class="topbar">
      <div class="topbar-copy">
        <h1 id="page-title">Upload workbook</h1>
        <p id="page-sub">Find mistakes, approve fixes, export clean data.</p>
      </div>
      <div class="topbar-actions">
        <button class="btn ghost sm" id="undo-btn" type="button" hidden title="Undo the last change">↺ Undo</button>
        <button class="btn ghost sm" id="reset-btn" type="button" hidden title="Discard all corrections">Reset</button>
        <a class="btn primary sm" id="quick-download" href="/api/download/cleaned" hidden>Download clean file</a>
      </div>
    </header>

    <!-- Upload / landing ------------------------------------------------- -->
    <section class="view" id="view-upload">
      <div class="hero card">
        <div class="hero-copy">
          <span class="eyebrow">Excel intelligence</span>
          <h2>Messy spreadsheets,<br><span class="grad">cleaned automatically.</span></h2>
          <p>DataFix reads your workbook, learns the patterns your data should follow, fixes high-confidence
            mistakes and asks you about everything else. Nothing is guessed, nothing is lost.</p>
          <div class="hero-chips">
            <span class="chip">Case &amp; spacing</span>
            <span class="chip">Emails &amp; phones</span>
            <span class="chip">IDs &amp; dates</span>
            <span class="chip">Duplicates</span>
            <span class="chip">Missing data</span>
          </div>
          <button class="btn primary" id="hero-cta" type="button">Upload a workbook</button>
        </div>
        <div class="hero-art" aria-hidden="true">
          <div class="mock-table">
            <div class="mock-head"><span>name</span><span>email</span><span>class</span></div>
            <div class="mock-row"><span class="bad">arjun J</span><span class="bad">arjun#gmail.com</span><span class="bad">1 cse21</span></div>
            <div class="mock-row"><span class="good">Arjun J</span><span class="good">arjun@gmail.com</span><span class="good">1CSE21</span></div>
            <div class="mock-row"><span class="bad">Banglore</span><span class="warn">asha@gmail</span><span class="good">2CSE22</span></div>
            <div class="mock-row"><span class="good">Bangalore</span><span class="warn">asha@gmail.com ?</span><span class="good">2CSE22</span></div>
          </div>
          <div class="mock-badge"><span class="pulse"></span> 3 fixes suggested</div>
        </div>
      </div>

      <div class="upload-grid">
        <div class="card upload-card">
          <div class="card-head">
            <div>
              <h3>Upload your workbook</h3>
              <p class="muted">We read the first sheet and keep your file untouched.</p>
            </div>
            <span class="pill">.xlsx · .xls · .csv</span>
          </div>

          <div class="dropzone" id="dropzone" tabindex="0" role="button" aria-label="Choose a file to upload">
            <div class="dz-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 16V4.5"/><path d="M7.5 9L12 4.5 16.5 9"/>
                <path d="M5 15v3.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V15"/>
              </svg>
            </div>
            <div class="dz-title">Drag &amp; drop your file here</div>
            <p class="dz-sub">or <button class="link-btn" id="browse-btn" type="button">browse files</button> from your computer</p>
            <span class="dz-note">Maximum {{ max_mb }} MB · first sheet is analyzed</span>
            <input type="file" id="file-input" accept=".xlsx,.xls,.xlsm,.csv" hidden>
          </div>

          <div class="file-chip" id="file-chip" hidden>
            <span class="file-badge" id="file-badge">XLS</span>
            <span class="file-info"><strong id="file-name"></strong><small id="file-size"></small></span>
            <button class="icon-btn" id="clear-file" type="button" aria-label="Remove file">×</button>
          </div>

          <div class="upload-error" id="upload-error" hidden></div>

          <div class="upload-actions">
            <span class="hint" id="upload-hint">Nothing leaves your computer.</span>
            <button class="btn primary" id="analyze-btn" type="button" disabled>Upload &amp; analyze</button>
          </div>
          <button class="btn ghost block" id="back-btn" type="button" hidden>← Back to dashboard</button>
        </div>

        <div class="card checks-card">
          <h3>What gets checked</h3>
          <ul class="check-list">
            <li><span class="dot violet"></span><div><strong>Text &amp; spelling</strong><small>Casing, missing letters, extra spaces, repeated words.</small></div></li>
            <li><span class="dot cyan"></span><div><strong>Emails &amp; phones</strong><small>Broken @ symbols, domains, digit counts, country codes.</small></div></li>
            <li><span class="dot green"></span><div><strong>IDs, dates &amp; numbers</strong><small>Format patterns, invalid dates, numbers stored as text.</small></div></li>
            <li><span class="dot amber"></span><div><strong>Duplicates &amp; gaps</strong><small>Repeated rows, emails, IDs, sequence gaps — never removed silently.</small></div></li>
            <li><span class="dot rose"></span><div><strong>Missing data</strong><small>Empty cells, NA/NULL placeholders, incomplete rows.</small></div></li>
          </ul>
        </div>
      </div>

      <div class="feature-row">
        <div class="feature"><span class="feature-icon">01</span><div><strong>Learn</strong><p>Column types and dominant formats are inferred from your data, not a fixed schema.</p></div></div>
        <div class="feature"><span class="feature-icon">02</span><div><strong>Fix &amp; ask</strong><p>High-confidence errors are corrected automatically; uncertain ones wait for your approval.</p></div></div>
        <div class="feature"><span class="feature-icon">03</span><div><strong>Export</strong><p>Download a clean workbook with highlighted fixes, plus a full correction report.</p></div></div>
      </div>
    </section>

    <!-- Dashboard -------------------------------------------------------- -->
    <section class="view" id="view-dashboard" hidden>
      <div class="welcome card">
        <div class="welcome-copy">
          <span class="eyebrow">Analysis complete</span>
          <h2 id="welcome-title">Your workbook is ready</h2>
          <p id="welcome-sub"></p>
          <div class="welcome-actions">
            <button class="btn primary sm" type="button" data-jump="review">Review &amp; compare</button>
            <button class="btn ghost sm" type="button" data-jump="issues" id="welcome-review">See remaining issues</button>
          </div>
        </div>
        <div class="health-wrap">
          <div class="health-ring" id="health-ring">
            <div class="health-inner"><strong id="health-value">0%</strong><span>data health</span></div>
          </div>
          <small id="health-note">Calculated from records with no open issues.</small>
        </div>
      </div>

      <div class="stats" id="stats"></div>

      <div class="panels">
        <section class="panel" data-panel="overview">
          <div class="grid-2">
            <div class="card">
              <div class="card-head">
                <div><h3>Open issues by category</h3><p class="muted" id="cat-note">Issues still waiting for a decision.</p></div>
                <span class="pill" id="cat-total">0 open</span>
              </div>
              <div class="cat-bars" id="cat-bars"></div>
            </div>
            <div class="card">
              <div class="card-head">
                <div><h3>Column profile</h3><p class="muted">Types detected from names and content.</p></div>
                <span class="pill">auto-detected</span>
              </div>
              <div class="table-scroll compact">
                <table class="mini-table" id="profile-table"></table>
              </div>
            </div>
          </div>

          <div class="card next-card">
            <div class="next-copy">
              <span class="eyebrow">Next step</span>
              <h3 id="next-title">Review the corrections</h3>
              <p id="next-text">Every change is shown with the original value, the suggestion and the evidence behind it.</p>
            </div>
            <div class="next-actions">
              <button class="btn primary" type="button" data-jump="review">Open review table</button>
              <a class="btn ghost" href="/api/download/report">Download error report</a>
            </div>
          </div>
        </section>

        <section class="panel" data-panel="review">
          <div class="card toolbar-card">
            <div class="legend">
              <span><i class="sw fixed"></i>Auto / approved</span>
              <span><i class="sw manual"></i>Manual edit</span>
              <span><i class="sw pending"></i>Suggested</span>
              <span><i class="sw flagged"></i>Needs attention</span>
            </div>
            <div class="review-tools">
              <label class="search">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4 4"/></svg>
                <input id="table-search" type="search" placeholder="Search rows…" autocomplete="off">
              </label>
              <select id="table-mode" aria-label="Filter rows">
                <option value="all">All rows</option>
                <option value="problems">With open issues</option>
                <option value="review">Needs review</option>
                <option value="corrected">Changed cells</option>
              </select>
            </div>
          </div>

          <div class="card table-card">
            <div class="table-scroll" id="table-scroll">
              <table class="data-table" id="data-table"></table>
            </div>
            <div class="table-foot">
              <span class="muted" id="table-note"></span>
              <div class="pager">
                <button class="icon-btn" id="prev-page" type="button" aria-label="Previous page">‹</button>
                <span id="page-label">Page 1</span>
                <button class="icon-btn" id="next-page" type="button" aria-label="Next page">›</button>
              </div>
            </div>
          </div>
          <p class="hint">Double-click a cell to edit it. The old value stays visible until you save. Deleted duplicate rows are hidden.</p>
        </section>

        <section class="panel" data-panel="issues">
          <div class="card toolbar-card wrap">
            <div class="chips" id="issue-tabs"></div>
            <div class="review-tools">
              <select id="issue-category" aria-label="Filter by category"></select>
              <label class="search">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4 4"/></svg>
                <input id="issue-search" type="search" placeholder="Search corrections…" autocomplete="off">
              </label>
            </div>
          </div>

          <div class="bulk-bar">
            <div class="bulk-copy"><strong id="issue-count">0 corrections</strong><span id="issue-note" class="muted"></span></div>
            <div class="bulk-actions">
              <button class="btn ghost sm" id="accept-all" type="button">Approve all suggestions</button>
              <button class="btn ghost sm" id="undo-btn-2" type="button">Undo last</button>
              <button class="btn danger sm" id="reset-btn-2" type="button">Reset all</button>
            </div>
          </div>

          <div class="issue-list" id="issue-list"></div>
          <div class="load-more"><button class="btn ghost" id="issue-more" type="button" hidden>Show more</button></div>
        </section>

        <section class="panel" data-panel="downloads">
          <div class="download-grid">
            <div class="card download-card accent">
              <div class="download-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 4v11M7.8 11l4.2 4.2L16.2 11"/><path d="M5 19h14"/></svg>
              </div>
              <h3>Cleaned Excel file</h3>
              <p>Your data with every accepted correction applied. Column names, row order and valid values are preserved.</p>
              <ul class="download-list">
                <li><span class="dot green"></span>Green cells: auto-corrected or approved</li>
                <li><span class="dot amber"></span>Amber cells: manual edits</li>
                <li><span class="dot violet"></span>Duplicate rows are removed only after approval</li>
              </ul>
              <a class="btn primary block" id="dl-cleaned" href="/api/download/cleaned">Download cleaned .xlsx</a>
            </div>

            <div class="card download-card">
              <div class="download-icon soft">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M7 3.5h7l4.5 4.5V20a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4.5a1 1 0 0 1 1-1z"/><path d="M14 3.5V8h4.5M9 12h6M9 15.5h6"/></svg>
              </div>
              <h3>Correction report</h3>
              <p>A separate workbook listing every detected issue, its row, confidence and current status.</p>
              <ul class="download-list">
                <li>Summary metrics</li>
                <li>All corrections with reasons</li>
                <li>Detected column types</li>
              </ul>
              <a class="btn ghost block" id="dl-report" href="/api/download/report">Download report .xlsx</a>
            </div>

            <div class="card download-card">
              <div class="download-icon soft">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M7 3.5h7l4.5 4.5V20a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4.5a1 1 0 0 1 1-1z"/><path d="M14 3.5V8h4.5M9 16.5l2-2 2 2 2.5-2.5"/></svg>
              </div>
              <h3>Original file</h3>
              <p>The exact file you uploaded, byte for byte. DataFix never edits your source workbook.</p>
              <ul class="download-list">
                <li id="original-name">—</li>
                <li id="original-size">—</li>
              </ul>
              <a class="btn ghost block" id="dl-original" href="/api/download/original">Download original</a>
            </div>
          </div>

          <div class="card session-card">
            <div class="card-head"><h3>Session summary</h3><span class="pill" id="session-sheet">—</span></div>
            <div class="session-grid" id="session-summary"></div>
          </div>
        </section>
      </div>
    </section>

    <!-- Appearance / theme settings -------------------------------------- -->
    <section class="view" id="view-appearance" hidden>
      <div class="appearance-grid">
        <div class="card appearance-card">
          <div class="card-head">
            <div>
              <h3>Appearance mode</h3>
              <p class="muted">Pick how DataFix looks, or follow your operating system.</p>
            </div>
            <span class="pill" id="mode-pill">System</span>
          </div>

          <div class="mode-options" id="mode-options" role="radiogroup" aria-label="Appearance mode">
            <button class="mode-option" type="button" role="radio" data-mode="light" aria-checked="false">
              <span class="mode-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2.6v2.1M12 19.3v2.1M4.4 4.4l1.5 1.5M18.1 18.1l1.5 1.5M2.6 12h2.1M19.3 12h2.1M4.4 19.6l1.5-1.5M18.1 5.9l1.5-1.5"/></svg>
              </span>
              <span class="mode-copy"><strong>Light</strong><small>Bright, clean surfaces</small></span>
              <span class="mode-check" aria-hidden="true">✓</span>
            </button>
            <button class="mode-option" type="button" role="radio" data-mode="dark" aria-checked="false">
              <span class="mode-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.2A8.2 8.2 0 0 1 9.8 4a8.5 8.5 0 1 0 10.2 10.2z"/></svg>
              </span>
              <span class="mode-copy"><strong>Dark</strong><small>Soft midnight shades</small></span>
              <span class="mode-check" aria-hidden="true">✓</span>
            </button>
            <button class="mode-option" type="button" role="radio" data-mode="system" aria-checked="false">
              <span class="mode-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4.5" width="18" height="12" rx="2"/><path d="M9 20h6M12 16.5V20"/></svg>
              </span>
              <span class="mode-copy"><strong>System</strong><small>Follows your device</small></span>
              <span class="mode-check" aria-hidden="true">✓</span>
            </button>
          </div>

          <div class="settings-divider"></div>

          <div class="card-head settings-head">
            <div>
              <h3>Light colour themes</h3>
              <p class="muted">Choose a palette for the light appearance. Selecting one previews it instantly.</p>
            </div>
            <span class="pill" id="theme-pill">Royal Purple</span>
          </div>
          <div class="theme-grid" id="theme-grid" role="radiogroup" aria-label="Light colour theme"></div>
          <p class="hint tiny" id="theme-hint">Colour palettes live in Light mode. Dark mode keeps its own midnight surfaces.</p>
        </div>

        <aside class="appearance-side">
          <div class="card preview-card">
            <div class="card-head">
              <div>
                <h3>Live preview</h3>
                <p class="muted">Real components in the selected palette.</p>
              </div>
              <span class="pill" id="preview-pill">Live</span>
            </div>

            <div class="live-preview" aria-hidden="true">
              <div class="lp-head">
                <span class="lp-logo"><i></i><i></i><i></i></span>
                <span class="lp-title"><strong id="preview-theme-name">Royal Purple</strong><small id="preview-mode-label">System · Dark</small></span>
                <span class="lp-badge">Preview</span>
              </div>
              <div class="lp-stats">
                <span class="lp-stat"><small>Data health</small><strong>33%</strong></span>
                <span class="lp-stat"><small>Issues</small><strong>36</strong></span>
                <span class="lp-stat"><small>Rows</small><strong>15</strong></span>
              </div>
              <div class="lp-bars"><i style="--h:36%"></i><i style="--h:62%"></i><i style="--h:48%"></i><i style="--h:84%"></i><i style="--h:57%"></i><i style="--h:72%"></i></div>
              <div class="lp-legend"><span><i></i>Open issues</span><span><i></i>Corrected</span></div>
              <div class="lp-actions"><span class="lp-btn primary">Download clean file</span><span class="lp-btn ghost">Review</span></div>
            </div>

            <div class="pref-line">
              <div class="pref-status" id="pref-status" role="status" aria-live="polite">Loading saved preferences…</div>
              <span class="pill" id="sync-pill">Local</span>
            </div>

            <div class="settings-actions">
              <button class="btn primary block" id="save-prefs" type="button">Save preferences</button>
              <div class="actions-row">
                <button class="btn ghost" id="apply-theme" type="button">Apply theme</button>
                <button class="btn ghost danger-text" id="reset-theme" type="button">Reset to default</button>
              </div>
            </div>
            <p class="hint tiny">Apply keeps the theme on this device. Save also syncs your local DataFix profile. No account needed.</p>
          </div>
        </aside>
      </div>
    </section>
  </main>
</div>

<div class="toasts" id="toasts" role="status" aria-live="polite"></div>

<div class="overlay" id="overlay" hidden>
  <div class="overlay-card">
    <div class="spinner"></div>
    <h3 id="overlay-title">Reading your workbook…</h3>
    <p id="overlay-sub">Learning column patterns and checking every cell.</p>
    <div class="overlay-bar"><span></span></div>
  </div>
</div>

<script src="/static/app.js"></script>
</body>
</html>

==============================================================================
FILE: templates/workspace.html
==============================================================================
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DataFix Workspace — Excel editor &amp; permanent file workspace</title>
  <meta name="description" content="A permanent Excel workspace: upload and reopen workbooks, edit cells, clean data, search, filter, track versions and export.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap" rel="stylesheet">
  <meta name="datafix-max-mb" content="{{ max_mb }}">
  <meta name="theme-color" content="#0a0713">
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%237c6cff'/%3E%3Cg fill='white'%3E%3Crect x='15' y='34' width='6' height='15' rx='3'/%3E%3Crect x='25' y='24' width='6' height='25' rx='3'/%3E%3Crect x='35' y='30' width='6' height='19' rx='3'/%3E%3Crect x='45' y='17' width='6' height='32' rx='3'/%3E%3C/g%3E%3C/svg%3E">
  <script>window.DATAFIX_INITIAL_PREFS = {{ initial_prefs|tojson }};</script>
  <script src="/static/theme.js"></script>
  <link rel="stylesheet" href="/static/styles.css">
  <link rel="stylesheet" href="/static/workspace.css">
</head>
<body class="ws-body">
<div class="ws-app" id="ws-app">
  <aside class="ws-sidebar" id="ws-sidebar" aria-label="Workbook library">
    <div class="ws-brand-row">
      <a class="ws-brand" href="/" aria-label="DataFix workspace home">
        <span class="ws-brand-mark">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 19V11M12 19V5M19 19v-5"/></svg>
        </span>
        <span class="ws-brand-text"><strong>DataFix</strong><small>Permanent workspace</small></span>
      </a>
      <button class="ws-icon-btn" id="ws-sidebar-collapse" type="button" title="Collapse sidebar" aria-label="Collapse sidebar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M14.5 6.5 9 12l5.5 5.5"/></svg>
      </button>
    </div>

    <button class="ws-new-file" id="ws-new-file" type="button">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>
      <span>New file</span>
    </button>

    <label class="ws-file-search">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4 4"/></svg>
      <input type="search" id="ws-file-search" placeholder="Search files" autocomplete="off" aria-label="Search saved files">
    </label>

    <nav class="ws-page-nav" aria-label="Workspace sections">
      <button class="ws-page-item is-active" type="button" data-page="home">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m4 10.5 8-6.5 8 6.5V20H4z"/><path d="M9.5 20v-6h5v6"/></svg>
        <span>Home</span>
      </button>
      <button class="ws-page-item" type="button" data-page="upload">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 16V5m0 0L7.5 9.5M12 5l4.5 4.5"/><path d="M5 15v3.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V15"/></svg>
        <span>Upload dataset</span>
      </button>
      <button class="ws-page-item" type="button" data-page="files">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 6.5A2.5 2.5 0 0 1 6.5 4H10l2 2h5.5A2.5 2.5 0 0 1 20 8.5v9A2.5 2.5 0 0 1 17.5 20h-11A2.5 2.5 0 0 1 4 17.5z"/></svg>
        <span>My files</span>
      </button>
      <button class="ws-page-item" type="button" data-page="analysis">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 19V5M4 19h16"/><path d="m7.5 15 3-4 3 2 4-6"/></svg>
        <span>Data analysis</span>
      </button>
      <button class="ws-page-item" type="button" data-page="reports">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M6 3.5h8l4 4V20H6z"/><path d="M14 3.5V8h4M9 12h6M9 15.5h6"/></svg>
        <span>Reports</span>
      </button>
    </nav>

    <nav class="ws-nav" aria-label="File scopes">
      <button class="ws-nav-item is-active" type="button" data-scope="all">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 6.5A2.5 2.5 0 0 1 6.5 4H10l2 2h5.5A2.5 2.5 0 0 1 20 8.5v9A2.5 2.5 0 0 1 17.5 20h-11A2.5 2.5 0 0 1 4 17.5z"/></svg>
        <span>All files</span><em id="ws-count-all">0</em>
      </button>
      <button class="ws-nav-item" type="button" data-scope="recent">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="8"/><path d="M12 7.5V12l3 2"/></svg>
        <span>Recent</span>
      </button>
      <button class="ws-nav-item" type="button" data-scope="starred">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m12 4 2.4 4.9 5.4.8-3.9 3.8.9 5.4-4.8-2.5-4.8 2.5.9-5.4L4.2 9.7l5.4-.8z"/></svg>
        <span>Starred</span><em id="ws-count-starred">0</em>
      </button>
      <button class="ws-nav-item" type="button" data-scope="opened">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 7h16M4 12h10M4 17h7"/></svg>
        <span>Recently opened</span>
      </button>
      <button class="ws-nav-item" type="button" data-scope="archived">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="5" width="16" height="4" rx="1.5"/><path d="M6 9v8.5A1.5 1.5 0 0 0 7.5 19h9a1.5 1.5 0 0 0 1.5-1.5V9M10 13h4"/></svg>
        <span>Archived</span><em id="ws-count-archived">0</em>
      </button>
      <button class="ws-nav-item" type="button" id="ws-history-nav">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 12a8 8 0 1 0 2.3-5.6M4 4v4h4"/><path d="M12 8v4l2.5 1.5"/></svg>
        <span>File history</span>
      </button>
    </nav>

    <div class="ws-files-head"><span id="ws-files-label">All files</span><button class="ws-icon-btn" id="ws-refresh-files" type="button" title="Refresh file list" aria-label="Refresh file list"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20 12a8 8 0 1 1-2.3-5.6M20 4v4h-4"/></svg></button></div>
    <div class="ws-file-list" id="ws-file-list" role="list"></div>

    <div class="ws-sidebar-footer">
      <button class="ws-nav-item" type="button" id="ws-open-storage">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 7.5A2.5 2.5 0 0 1 6.5 5h11A2.5 2.5 0 0 1 20 7.5v9a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 16.5z"/><path d="M4 10h16"/></svg>
        <span>Storage &amp; settings</span>
      </button>
      <a class="ws-nav-item" href="/cleaner">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3.8l8 4.6v7.2l-8 4.6-8-4.6V8.4l8-4.6z"/><path d="M12 8.2v4.6M12 16h.01"/></svg>
        <span>Classic cleaner</span>
      </a>
      <button class="ws-nav-item" type="button" id="ws-open-appearance">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3.5a8.5 8.5 0 1 0 0 17c1.4 0 2-.9 2-1.9 0-1.2-.9-1.7-.9-2.7 0-.9.7-1.4 1.7-1.4h1.7a3.9 3.9 0 0 0 3.9-3.9C20.4 6.4 16.8 3.5 12 3.5z"/><circle cx="7.7" cy="10.2" r="1.1" fill="currentColor" stroke="none"/><circle cx="11.4" cy="7.4" r="1.1" fill="currentColor" stroke="none"/><circle cx="15.6" cy="8.6" r="1.1" fill="currentColor" stroke="none"/></svg>
        <span>Appearance</span>
      </button>
      <div class="ws-storage-mini" id="ws-storage-mini">Local · private</div>
    </div>
  </aside>

  <div class="ws-sidebar-scrim" id="ws-sidebar-scrim" hidden></div>

  <main class="ws-main">
    <header class="ws-topbar">
      <button class="ws-icon-btn ws-mobile-menu" id="ws-mobile-menu" type="button" aria-label="Open file sidebar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
      </button>
      <div class="ws-title-block">
        <div class="ws-title-row">
          <h1 id="ws-title">Workspace</h1>
          <button class="ws-star-btn" id="ws-star-btn" type="button" title="Star this file" aria-label="Star this file" hidden>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m12 4 2.4 4.9 5.4.8-3.9 3.8.9 5.4-4.8-2.5-4.8 2.5.9-5.4L4.2 9.7l5.4-.8z"/></svg>
          </button>
        </div>
        <p class="ws-meta" id="ws-meta">Choose a saved workbook or create a new one.</p>
      </div>
      <div class="ws-global-search">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4 4"/></svg>
        <input type="search" id="ws-global-search" placeholder="Search all workbooks, sheets and cells" autocomplete="off" aria-label="Search all workbooks">
        <kbd>/</kbd>
      </div>
      <div class="ws-top-actions">
        <span class="ws-save-chip" id="ws-save-status" data-state="idle" title="Autosave status">Ready</span>
        <button class="ws-icon-btn" id="ws-undo" type="button" title="Undo (Ctrl+Z)" aria-label="Undo" disabled><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M9 7 4.5 11.5 9 16"/><path d="M4.5 11.5h9a5.5 5.5 0 0 1 0 11h-2"/></svg></button>
        <button class="ws-icon-btn" id="ws-redo" type="button" title="Redo (Ctrl+Y)" aria-label="Redo" disabled><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="m15 7 4.5 4.5L15 16"/><path d="M19.5 11.5h-9a5.5 5.5 0 0 0 0 11h2"/></svg></button>
        <button class="ws-primary-btn" id="ws-export-open" type="button">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M12 4v11M7.8 11l4.2 4.2L16.2 11"/><path d="M5 19h14"/></svg>
          <span>Export</span>
        </button>
      </div>
    </header>

    <div class="ws-pages" id="ws-pages" hidden></div>

    <section class="ws-ribbon" id="ws-ribbon" aria-label="Ribbon">
      <div class="ws-ribbon-tabs" role="tablist">
        <button class="ws-ribbon-tab" role="tab" data-tab="file">FILE</button>
        <button class="ws-ribbon-tab is-active" role="tab" data-tab="home">HOME</button>
        <button class="ws-ribbon-tab" role="tab" data-tab="insert">INSERT</button>
        <button class="ws-ribbon-tab" role="tab" data-tab="page">PAGE LAYOUT</button>
        <button class="ws-ribbon-tab" role="tab" data-tab="formulas">FORMULAS</button>
        <button class="ws-ribbon-tab" role="tab" data-tab="data">DATA</button>
        <button class="ws-ribbon-tab" role="tab" data-tab="review">REVIEW</button>
        <button class="ws-ribbon-tab" role="tab" data-tab="view">VIEW</button>
        <button class="ws-ribbon-tab" role="tab" data-tab="cleaning">DATA CLEANING</button>
        <button class="ws-ribbon-tab" role="tab" data-tab="ai">AI CORRECTION</button>
      </div>
      <div class="ws-ribbon-body">
        <div class="ws-ribbon-panel" data-panel="file" hidden>
          <div class="ws-group"><button class="ws-rbtn" data-action="new-file">New</button><button class="ws-rbtn" data-action="upload-open">Open files…</button><button class="ws-rbtn" data-action="import-path">Import path…</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="save-now">Save</button><button class="ws-rbtn" data-action="save-as">Save as…</button><button class="ws-rbtn" data-action="checkpoint">Checkpoint</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="export-open">Export…</button><button class="ws-rbtn" data-action="print">Print</button><button class="ws-rbtn" data-action="download-original">Original file</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="storage-settings">Storage location…</button><button class="ws-rbtn" data-action="close-workbook">Close file</button></div>
        </div>
        <div class="ws-ribbon-panel is-active" data-panel="home">
          <div class="ws-group"><button class="ws-rbtn" data-action="undo"><span class="ws-k">Ctrl</span>Z</button><button class="ws-rbtn" data-action="redo"><span class="ws-k">Ctrl</span>Y</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="cut">Cut</button><button class="ws-rbtn" data-action="copy">Copy</button><button class="ws-rbtn" data-action="paste">Paste</button><button class="ws-rbtn" data-action="clear-cell">Clear</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="bold"><strong>B</strong></button><button class="ws-rbtn" data-action="italic"><em>I</em></button><button class="ws-rbtn" data-action="underline"><u>U</u></button><label class="ws-rbtn ws-color" title="Text colour">A<input type="color" id="ws-text-color" value="#111827"></label><label class="ws-rbtn ws-color" title="Fill colour">Fill<input type="color" id="ws-fill-color" value="#fff3bf"></label></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="align-left">Left</button><button class="ws-rbtn" data-action="align-center">Center</button><button class="ws-rbtn" data-action="align-right">Right</button><button class="ws-rbtn" data-action="wrap">Wrap</button><button class="ws-rbtn" data-action="clear-style">Clear style</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="insert-row">Insert row</button><button class="ws-rbtn" data-action="delete-row">Delete row</button><button class="ws-rbtn" data-action="insert-col">Insert col</button><button class="ws-rbtn" data-action="delete-col">Delete col</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="find-replace">Find &amp; replace</button><button class="ws-rbtn" data-action="goto">Go to cell…</button></div>
        </div>
        <div class="ws-ribbon-panel" data-panel="insert" hidden>
          <div class="ws-group"><button class="ws-rbtn" data-action="add-sheet">New sheet</button><button class="ws-rbtn" data-action="duplicate-sheet">Duplicate sheet</button><button class="ws-rbtn" data-action="rename-sheet">Rename sheet</button><button class="ws-rbtn" data-action="delete-sheet">Delete sheet</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="checkpoint">Save checkpoint</button><button class="ws-rbtn" data-action="versions">Version history</button><button class="ws-rbtn" data-disabled title="Not implemented yet: insert charts">Chart <small>soon</small></button><button class="ws-rbtn" data-disabled title="Not implemented yet: insert images">Image <small>soon</small></button></div>
        </div>
        <div class="ws-ribbon-panel" data-panel="page" hidden>
          <div class="ws-group"><button class="ws-rbtn" data-action="freeze-1">Freeze first row</button><button class="ws-rbtn" data-action="freeze-3">Freeze 3 rows</button><button class="ws-rbtn" data-action="freeze-col">Freeze first column</button><button class="ws-rbtn" data-action="unfreeze">Unfreeze</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="hide-rows">Hide selected row(s)</button><button class="ws-rbtn" data-action="hide-cols">Hide selected col(s)</button><button class="ws-rbtn" data-action="show-hidden">Show hidden</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="grow-rows">More rows</button><button class="ws-rbtn" data-action="grow-cols">More columns</button><button class="ws-rbtn" data-disabled title="Not implemented yet: page margins and paper size">Margins <small>soon</small></button></div>
        </div>
        <div class="ws-ribbon-panel" data-panel="formulas" hidden>
          <div class="ws-group">
            <select class="ws-select" id="ws-function-select" aria-label="Insert function">
              <option value="">Insert function…</option>
              <option value="SUM">SUM</option><option value="AVERAGE">AVERAGE</option>
              <option value="COUNT">COUNT</option><option value="COUNTA">COUNTA</option>
              <option value="MIN">MIN</option><option value="MAX">MAX</option>
              <option value="IF">IF</option><option value="COUNTIF">COUNTIF</option>
              <option value="SUMIF">SUMIF</option><option value="VLOOKUP">VLOOKUP</option>
              <option value="XLOOKUP">XLOOKUP</option><option value="CONCAT">CONCAT</option>
              <option value="TEXTJOIN">TEXTJOIN</option><option value="UPPER">UPPER</option>
              <option value="TRIM">TRIM</option><option value="TODAY">TODAY</option>
            </select>
            <button class="ws-rbtn" data-action="show-formulas">Show formulas</button>
            <button class="ws-rbtn" data-action="recalc">Recalculate</button>
          </div>
          <div class="ws-group"><button class="ws-rbtn" data-disabled title="Not implemented yet: named ranges">Name manager <small>soon</small></button><button class="ws-rbtn" data-disabled title="Not implemented yet: formula auditing">Trace precedents <small>soon</small></button></div>
        </div>
        <div class="ws-ribbon-panel" data-panel="data" hidden>
          <div class="ws-group"><button class="ws-rbtn" data-action="toggle-filters">Filter panel</button><button class="ws-rbtn" data-action="sort-asc">Sort A→Z</button><button class="ws-rbtn" data-action="sort-desc">Sort Z→A</button><button class="ws-rbtn" data-action="clear-filters">Clear filters</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="find-replace">Find &amp; replace</button><button class="ws-rbtn" data-action="duplicates-report">Duplicate values</button><button class="ws-rbtn" data-disabled title="Not implemented yet: remove duplicate rows">Remove duplicates <small>soon</small></button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="search-sheet">Search in sheet</button><button class="ws-rbtn" data-action="goto">Go to cell…</button></div>
        </div>
        <div class="ws-ribbon-panel" data-panel="review" hidden>
          <div class="ws-group"><button class="ws-rbtn" data-action="run-analysis">Clean data…</button><button class="ws-rbtn" data-action="versions">Versions</button><button class="ws-rbtn" data-action="print">Print preview</button><button class="ws-rbtn" data-action="download-original">Download original</button></div>
          <div class="ws-group"><a class="ws-rbtn" href="/cleaner">Classic cleaner</a><button class="ws-rbtn" data-disabled title="Not implemented yet: lock individual ranges">Protect <small>soon</small></button><button class="ws-rbtn" data-disabled title="Not implemented yet: cell comments">Comment <small>soon</small></button></div>
        </div>
        <div class="ws-ribbon-panel" data-panel="view" hidden>
          <div class="ws-group"><button class="ws-rbtn" data-action="toggle-filters">Filters</button><button class="ws-rbtn" data-action="toggle-sidebar">Sidebar</button><button class="ws-rbtn" data-action="toggle-gridlines">Gridlines</button><button class="ws-rbtn" data-action="fullscreen">Full screen</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="zoom-90">90%</button><button class="ws-rbtn" data-action="zoom-100">100%</button><button class="ws-rbtn" data-action="zoom-110">110%</button><button class="ws-rbtn" data-action="toggle-appearance">Light / dark</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="show-hidden">Show hidden</button><button class="ws-rbtn" data-action="freeze-1">Freeze top row</button></div>
        </div>
        <div class="ws-ribbon-panel" data-panel="cleaning" hidden>
          <div class="ws-group"><button class="ws-rbtn" data-action="run-analysis">Run cleaning analysis</button><button class="ws-rbtn" data-action="cleaning-panel">Open findings</button><button class="ws-rbtn" data-action="next-issue">Next finding</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-action="analysis-export">Export findings CSV</button><button class="ws-rbtn" data-action="analysis-reset">Clear analysis</button><a class="ws-rbtn" href="/cleaner">Full cleaning reports</a></div>
        </div>
        <div class="ws-ribbon-panel" data-panel="ai" hidden>
          <div class="ws-group"><button class="ws-rbtn" data-action="run-analysis">Analyse with local rules</button><button class="ws-rbtn" data-action="apply-safe">Apply selected suggestion</button><button class="ws-rbtn" data-action="explain-issue">Why was this flagged?</button></div>
          <div class="ws-group"><button class="ws-rbtn" data-disabled title="Not implemented yet: external AI service. Corrections run locally.">Cloud AI <small>off</small></button><span class="ws-ribbon-note">Suggestions come from the local cleaning engine. Nothing is sent to a cloud service.</span></div>
        </div>
      </div>
    </section>

    <section class="ws-formula-bar" id="ws-formula-bar">
      <input class="ws-name-box" id="ws-name-box" aria-label="Name box" placeholder="A1" spellcheck="false">
      <span class="ws-fx">fx</span>
      <input class="ws-formula-input" id="ws-formula-input" aria-label="Formula bar" placeholder="Select a cell to see its value or formula" spellcheck="false">
      <button class="ws-icon-btn" id="ws-formula-commit" type="button" title="Commit the formula bar value" aria-label="Commit value">✓</button>
    </section>

    <section class="ws-filter-panel" id="ws-filter-panel" hidden>
      <div class="ws-filter-head">
        <strong>Filters &amp; sorting</strong>
        <span class="ws-filter-note">Filters change only the view — stored data is not modified.</span>
        <div class="ws-filter-actions">
          <button class="ws-small-btn" id="ws-filter-add" type="button">Add filter</button>
          <button class="ws-small-btn" id="ws-filter-apply" type="button">Apply</button>
          <button class="ws-small-btn ghost" id="ws-filter-clear" type="button">Clear</button>
        </div>
      </div>
      <div class="ws-filter-rows" id="ws-filter-rows"></div>
      <div class="ws-filter-foot">
        <label class="ws-inline-field">Sort
          <select id="ws-sort-col"><option value="">No sorting</option></select>
        </label>
        <select id="ws-sort-dir" aria-label="Sort direction"><option value="asc">Ascending</option><option value="desc">Descending</option></select>
        <label class="ws-inline-field grow">In this sheet <input type="search" id="ws-sheet-search" placeholder="Only show rows containing…"></label>
        <span class="ws-inline-note">Filtered-out rows stay in the workbook; only the view is narrowed.</span>
      </div>
    </section>

    <section class="ws-sheet-area" id="ws-sheet-area">
      <div class="ws-grid-shell" id="ws-grid-shell">
        <div class="ws-grid-viewport" id="ws-grid-viewport" tabindex="0" role="grid" aria-label="Spreadsheet grid">
          <div class="ws-grid-canvas" id="ws-grid-canvas">
            <div class="ws-grid-cells" id="ws-grid-cells"></div>
            <div class="ws-cell-editor" id="ws-cell-editor" hidden><input type="text" id="ws-cell-editor-input" spellcheck="false"></div>
          </div>
        </div>
        <div class="ws-grid-corner" id="ws-grid-corner" title="Select all"></div>
        <div class="ws-grid-colheads" id="ws-grid-colheads"><div class="ws-grid-colheads-inner" id="ws-colheads-inner"></div></div>
        <div class="ws-grid-rowheads" id="ws-grid-rowheads"><div class="ws-grid-rowheads-inner" id="ws-rowheads-inner"></div></div>
        <div class="ws-empty" id="ws-empty">
          <div class="ws-empty-card">
            <div class="ws-empty-mark">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3.5" y="3.5" width="17" height="17" rx="3"/><path d="M3.5 9.5h17M9.5 9.5v11M3.5 15h17"/></svg>
            </div>
            <h2>Your permanent Excel workspace</h2>
            <p>Create a workbook, upload spreadsheets, or reopen a saved file. Everything is saved to the Python backend — never only in the browser.</p>
            <div class="ws-empty-actions">
              <button class="ws-primary-btn" type="button" data-action="new-file">New workbook</button>
              <button class="ws-small-btn" type="button" data-action="upload-open">Upload files</button>
              <button class="ws-small-btn" type="button" data-action="import-path">Import from path</button>
            </div>
            <div class="ws-drop-hint">Drop XLSX, XLSM, XLS, CSV or ODS files anywhere on this page.</div>
            <div class="ws-recent-mini" id="ws-recent-mini"></div>
          </div>
        </div>
        <div class="ws-drop-overlay" id="ws-drop-overlay" hidden><div>Drop spreadsheets to import them</div></div>
      </div>
    </section>

    <section class="ws-analysis-panel" id="ws-analysis-panel" hidden>
      <div class="ws-analysis-head">
        <div><strong>Cleaning &amp; AI correction</strong><span id="ws-analysis-summary">Run the analysis to inspect detected issues.</span></div>
        <div><button class="ws-small-btn" id="ws-analysis-run" type="button">Run analysis</button><button class="ws-small-btn ghost" id="ws-analysis-close" type="button">Close</button></div>
      </div>
      <div class="ws-analysis-list" id="ws-analysis-list"></div>
    </section>

    <footer class="ws-statusbar">
      <div class="ws-sheet-tabs" id="ws-sheet-tabs" role="tablist"></div>
      <button class="ws-sheet-add" id="ws-sheet-add" type="button" title="Add worksheet" aria-label="Add worksheet">+</button>
      <div class="ws-status-spacer"></div>
      <span id="ws-status-cell">No cell selected</span>
      <span id="ws-status-zoom">100%</span>
      <span id="ws-status-storage" title="Storage root">—</span>
    </footer>
  </main>
</div>

<input type="file" id="ws-file-input" accept=".xlsx,.xlsm,.xls,.csv,.ods" multiple hidden>

<!-- Dialogs -->
<div class="ws-modal" id="ws-modal-new" hidden>
  <div class="ws-modal-card" role="dialog" aria-modal="true" aria-labelledby="ws-new-title">
    <h2 id="ws-new-title">New workbook</h2>
    <p>Creates a blank workbook in permanent storage with one worksheet.</p>
    <label class="ws-field">File name<input type="text" id="ws-new-name" placeholder="Untitled workbook" maxlength="120"></label>
    <div class="ws-modal-actions"><button class="ws-small-btn ghost" data-close type="button">Cancel</button><button class="ws-primary-btn" id="ws-new-create" type="button">Create workbook</button></div>
  </div>
</div>

<div class="ws-modal" id="ws-modal-upload" hidden>
  <div class="ws-modal-card" role="dialog" aria-modal="true" aria-labelledby="ws-upload-title">
    <h2 id="ws-upload-title">Uploading files</h2>
    <p id="ws-upload-note">Saving the original file untouched, then validating and importing it.</p>
    <div class="ws-progress"><div class="ws-progress-bar" id="ws-upload-bar"></div></div>
    <div class="ws-upload-list" id="ws-upload-list"></div>
    <div class="ws-modal-actions"><button class="ws-small-btn" id="ws-upload-cancel" type="button">Cancel upload</button><button class="ws-primary-btn" id="ws-upload-done" type="button" hidden>Done</button></div>
  </div>
</div>

<div class="ws-modal" id="ws-modal-export" hidden>
  <div class="ws-modal-card wide" role="dialog" aria-modal="true" aria-labelledby="ws-export-title">
    <h2 id="ws-export-title">Export workbook</h2>
    <div class="ws-export-grid">
      <label class="ws-field">Format
        <select id="ws-export-format"></select>
      </label>
      <label class="ws-field">What to export
        <select id="ws-export-scope">
          <option value="workbook">Whole workbook (all sheets)</option>
          <option value="sheet">Current sheet only</option>
          <option value="view">Current sheet with filters / search</option>
          <option value="selected">Selected rows and columns</option>
        </select>
      </label>
    </div>
    <p class="ws-format-note" id="ws-export-note"></p>
    <div class="ws-export-history" id="ws-export-history"></div>
    <div class="ws-modal-actions"><button class="ws-small-btn ghost" data-close type="button">Close</button><button class="ws-primary-btn" id="ws-export-run" type="button">Create export</button></div>
  </div>
</div>

<div class="ws-modal" id="ws-modal-versions" hidden>
  <div class="ws-modal-card wide" role="dialog" aria-modal="true" aria-labelledby="ws-versions-title">
    <h2 id="ws-versions-title">Version history</h2>
    <p>Every save keeps an undoable edit batch. Full gzip snapshots are also stored on disk; restoring one creates a new version so nothing is lost.</p>
    <div class="ws-version-actions"><input type="text" id="ws-checkpoint-label" placeholder="Checkpoint label (optional)" maxlength="80"><button class="ws-small-btn" id="ws-checkpoint-create" type="button">Create checkpoint</button></div>
    <div class="ws-version-list" id="ws-version-list"></div>
    <h3 class="ws-subhead">Recent edit batches</h3>
    <div class="ws-batch-list" id="ws-batch-list"></div>
    <div class="ws-modal-actions"><button class="ws-small-btn ghost" data-close type="button">Close</button></div>
  </div>
</div>

<div class="ws-modal" id="ws-modal-find" hidden>
  <div class="ws-modal-card" role="dialog" aria-modal="true" aria-labelledby="ws-find-title">
    <h2 id="ws-find-title">Find &amp; replace</h2>
    <label class="ws-field">Find<input type="text" id="ws-find-text" placeholder="Text to find"></label>
    <label class="ws-field">Replace with<input type="text" id="ws-replace-text" placeholder="Replacement text"></label>
    <div class="ws-check-row">
      <label><input type="checkbox" id="ws-find-case"> Match case</label>
      <label><input type="checkbox" id="ws-find-whole"> Whole cell only</label>
      <label><input type="checkbox" id="ws-find-header"> Include header row</label>
    </div>
    <div class="ws-modal-actions"><span class="ws-find-status" id="ws-find-status"></span><button class="ws-small-btn" id="ws-find-next" type="button">Find next</button><button class="ws-small-btn" id="ws-replace-one" type="button">Replace</button><button class="ws-primary-btn" id="ws-replace-all" type="button">Replace all</button></div>
  </div>
</div>

<div class="ws-modal" id="ws-modal-goto" hidden>
  <div class="ws-modal-card" role="dialog" aria-modal="true" aria-labelledby="ws-goto-title">
    <h2 id="ws-goto-title">Go to cell</h2>
    <label class="ws-field">Cell or row<input type="text" id="ws-goto-input" placeholder="e.g. F42"></label>
    <div class="ws-modal-actions"><button class="ws-small-btn ghost" data-close type="button">Cancel</button><button class="ws-primary-btn" id="ws-goto-run" type="button">Go</button></div>
  </div>
</div>

<div class="ws-modal" id="ws-modal-import" hidden>
  <div class="ws-modal-card" role="dialog" aria-modal="true" aria-labelledby="ws-import-title">
    <h2 id="ws-import-title">Import a local workbook</h2>
    <p>Type a full path or use the desktop file picker. DataFix copies the file into permanent storage; the original file is never modified.</p>
    <label class="ws-field">File path<div class="ws-path-row"><input type="text" id="ws-import-path" placeholder="C:\\Users\\me\\Documents\\data.xlsx"><button class="ws-small-btn" id="ws-import-browse" type="button">Browse…</button></div></label>
    <div class="ws-modal-actions"><button class="ws-small-btn ghost" data-close type="button">Cancel</button><button class="ws-primary-btn" id="ws-import-run" type="button">Import</button></div>
  </div>
</div>

<div class="ws-modal" id="ws-modal-storage" hidden>
  <div class="ws-modal-card" role="dialog" aria-modal="true" aria-labelledby="ws-storage-title">
    <h2 id="ws-storage-title">Storage &amp; settings</h2>
    <p>Existing workbooks keep their original storage location. New workbooks are created in the selected folder.</p>
    <div class="ws-storage-info" id="ws-storage-info"></div>
    <label class="ws-field">New storage location<div class="ws-path-row"><input type="text" id="ws-storage-input" placeholder="C:\\DataFix"><button class="ws-small-btn" id="ws-storage-browse" type="button">Browse…</button></div></label>
    <label class="ws-field">Keep this many versions (5–1000)<input type="number" id="ws-retention" min="5" max="1000" step="1"></label>
    <div class="ws-modal-actions"><button class="ws-small-btn" id="ws-storage-reset" type="button">Use default folder</button><button class="ws-small-btn ghost" id="ws-storage-save" type="button">Save location</button><button class="ws-small-btn" id="ws-retention-save" type="button">Save retention</button><button class="ws-primary-btn" data-close type="button">Close</button></div>
  </div>
</div>

<div class="ws-modal" id="ws-modal-appearance" hidden>
  <div class="ws-modal-card wide" role="dialog" aria-modal="true" aria-labelledby="ws-appearance-title">
    <h2 id="ws-appearance-title">Appearance</h2>
    <div class="ws-mode-row" id="ws-mode-row"></div>
    <div class="ws-theme-grid" id="ws-theme-grid"></div>
    <div class="ws-modal-actions"><button class="ws-small-btn ghost" id="ws-theme-reset" type="button">Reset</button><button class="ws-small-btn ghost" data-close type="button">Cancel</button><button class="ws-primary-btn" id="ws-theme-save" type="button">Save appearance</button></div>
  </div>
</div>

<div class="ws-modal" id="ws-modal-search" hidden>
  <div class="ws-modal-card wide" role="dialog" aria-modal="true" aria-labelledby="ws-search-title">
    <h2 id="ws-search-title">Search results</h2>
    <p id="ws-search-summary"></p>
    <div class="ws-search-results" id="ws-search-results"></div>
    <div class="ws-modal-actions"><button class="ws-small-btn ghost" data-close type="button">Close</button></div>
  </div>
</div>

<div class="ws-modal" id="ws-modal-confirm" hidden>
  <div class="ws-modal-card" role="dialog" aria-modal="true" aria-labelledby="ws-confirm-title">
    <h2 id="ws-confirm-title">Are you sure?</h2>
    <p id="ws-confirm-text"></p>
    <div class="ws-modal-actions"><button class="ws-small-btn ghost" id="ws-confirm-cancel" type="button">Cancel</button><button class="ws-primary-btn danger" id="ws-confirm-ok" type="button">Continue</button></div>
  </div>
</div>

<div class="ws-toasts" id="ws-toasts" aria-live="polite"></div>
<div class="ws-upload-dock" id="ws-upload-dock" hidden></div>
<script src="/static/workspace.js"></script>
</body>
</html>

==============================================================================
FILE: templates/workspace_print.html
==============================================================================
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ workbook.name }} — {{ sheet.name }} · Print</title>
  <style>
    :root { color-scheme: light; }
    * { box-sizing: border-box; }
    body { font-family: "Segoe UI", system-ui, sans-serif; margin: 18px; color: #111827; background: #fff; }
    .toolbar { display: flex; gap: 10px; align-items: center; margin-bottom: 14px; }
    .toolbar button { padding: 8px 16px; border-radius: 8px; border: 1px solid #cbd5e1;
      background: #0f766e; color: #fff; cursor: pointer; font-weight: 600; }
    .toolbar button.secondary { background: #fff; color: #0f172a; }
    h1 { font-size: 18px; margin: 0; }
    .meta { color: #475569; font-size: 12px; margin: 2px 0 0; }
    table { border-collapse: collapse; width: 100%; font-size: 11px; }
    th, td { border: 1px solid #cbd5e1; padding: 3px 5px; vertical-align: top; white-space: pre-wrap; }
    th { background: #f1f5f9; text-align: left; position: sticky; top: 0; }
    td.row-no { background: #f8fafc; color: #64748b; text-align: right; width: 48px; }
    @media print {
      .toolbar { display: none; }
      body { margin: 0; }
      th { position: static; }
      @page { size: landscape; margin: 10mm; }
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <button type="button" onclick="window.print()">Print</button>
    <button type="button" class="secondary" onclick="window.close()">Close</button>
    <div>
      <h1>{{ workbook.name }} — {{ sheet.name }}</h1>
      <p class="meta">{{ total }} row(s)
        {% if query %} · search “{{ query }}”{% endif %}
        {% if filters %} · {{ filters|length }} filter(s) applied{% endif %}
      </p>
    </div>
  </div>
  <table>
    <thead>
      <tr>
        <th class="row-no">#</th>
        {% for label in labels %}<th>{{ label }}</th>{% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for row in rows %}
      <tr>
        <td class="row-no">{{ row.number }}</td>
        {% for cell in row.cells %}
        <td style="{% if cell.style %}{% if cell.style.bold %}font-weight:700;{% endif %}{% if cell.style.italic %}font-style:italic;{% endif %}{% if cell.style.color %}color:{{ cell.style.color }};{% endif %}{% if cell.style.fill %}background:{{ cell.style.fill }};{% endif %}{% if cell.style.align %}text-align:{{ cell.style.align }};{% endif %}{% endif %}">{{ cell.value }}</td>
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>
</body>
</html>

==============================================================================
FILE: tests/test_api.py
==============================================================================
"""End-to-end tests for the permanent workspace REST API."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from datafix import config

_TMP = Path(tempfile.mkdtemp(prefix="datafix_api_"))
_ORIGINAL = {
    "DATABASE_PATH": config.DATABASE_PATH,
    "STORAGE_CONFIG": config.STORAGE_CONFIG,
    "DEFAULT_STORAGE_ROOT": config.DEFAULT_STORAGE_ROOT,
    "DEFAULT_VERSION_RETENTION": config.DEFAULT_VERSION_RETENTION,
}
config.DATABASE_PATH = _TMP / "api.db"
config.STORAGE_CONFIG = _TMP / "storage.json"
config.DEFAULT_STORAGE_ROOT = _TMP / "storage"

from datafix import db  # noqa: E402

db.init_db()

from app import app  # noqa: E402

PREFIX = "/api/workspace"


def setUpModule():
    # Re-apply the temporary paths; another test module may have changed config.
    config.DATABASE_PATH = _TMP / "api.db"
    config.STORAGE_CONFIG = _TMP / "storage.json"
    config.DEFAULT_STORAGE_ROOT = _TMP / "storage"
    db.init_db()


def tearDownModule():
    for name, value in _ORIGINAL.items():
        setattr(config, name, value)
    shutil.rmtree(_TMP, ignore_errors=True)


def make_xlsx(rows):
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.title = "People"
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    buffer.seek(0)
    return buffer


class WorkspaceApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        response = self.client.post(f"{PREFIX}/workbooks", json={"name": "API test workbook"})
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        self.workbook = response.get_json()["workbook"]
        self.wb_id = self.workbook["id"]
        self.sheet_id = self.workbook["sheets"][0]["id"]

    def ops(self, *operations, label="Test edit", version=False):
        response = self.client.post(
            f"{PREFIX}/workbooks/{self.wb_id}/ops",
            json={"ops": list(operations), "label": label, "version": version},
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def grid(self, **params):
        params.setdefault("sheet_id", self.sheet_id)
        response = self.client.get(f"{PREFIX}/workbooks/{self.wb_id}/grid", query_string=params)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def seed(self):
        return self.ops(
            {"op": "set_cell", "sheet": self.sheet_id, "row": 1, "col": 1, "value": "Name"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 1, "col": 2, "value": "Score"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 1, "col": 3, "value": "Total"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 2, "col": 1, "value": "Ada"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 2, "col": 2, "value": 2},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 3, "col": 1, "value": "Grace"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 3, "col": 2, "value": 3},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 2, "col": 3, "formula": "=B2+B3"},
        )

    def test_state_and_library_scopes(self):
        state = self.client.get(f"{PREFIX}/state").get_json()
        self.assertTrue(state["ok"])
        self.assertIn("root", state["storage"])
        self.assertIn("xlsx", state["formats"])
        self.assertGreaterEqual(state["limits"]["max_upload_mb"], 1)

        listing = self.client.get(f"{PREFIX}/workbooks").get_json()
        self.assertTrue(any(item["id"] == self.wb_id for item in listing["workbooks"]))

        patched = self.client.patch(
            f"{PREFIX}/workbooks/{self.wb_id}", json={"starred": True, "name": "Starred book"}
        ).get_json()
        self.assertTrue(patched["workbook"]["starred"])
        self.assertEqual(patched["workbook"]["name"], "Starred book")
        starred = self.client.get(f"{PREFIX}/workbooks?scope=starred").get_json()
        self.assertEqual(starred["workbooks"][0]["id"], self.wb_id)

        archived = self.client.patch(
            f"{PREFIX}/workbooks/{self.wb_id}", json={"archived": True}
        ).get_json()
        self.assertTrue(archived["workbook"]["archived"])
        archived_list = self.client.get(f"{PREFIX}/workbooks?scope=archived").get_json()
        self.assertTrue(any(item["id"] == self.wb_id for item in archived_list["workbooks"]))
        self.client.patch(f"{PREFIX}/workbooks/{self.wb_id}", json={"archived": False})

    def test_editing_formulas_undo_redo_and_grid(self):
        result = self.seed()
        self.assertTrue(result["can_undo"])
        self.assertTrue(result["workbook"]["sheets"])
        grid = self.grid(rows=10, cols=5)
        self.assertIn(2, grid["rows"])
        self.assertEqual(grid["columns"], [1, 2, 3, 4, 5])
        self.assertEqual(float(grid["cells"]["2:3"]["v"]), 5)
        self.assertEqual(grid["cells"]["1:1"]["v"], "Name")
        self.assertEqual(grid["headers"]["2"], "Score")

        undone = self.client.post(f"{PREFIX}/workbooks/{self.wb_id}/undo").get_json()["result"]
        self.assertEqual(undone["workbook"]["id"], self.wb_id)
        after_undo = self.grid(rows=10, cols=5)
        self.assertNotIn("2:3", after_undo["cells"])
        redone = self.client.post(f"{PREFIX}/workbooks/{self.wb_id}/redo").get_json()["result"]
        self.assertTrue(redone["can_undo"])
        self.assertEqual(float(self.grid(rows=10, cols=5)["cells"]["2:3"]["v"]), 5)

        recalculated = self.client.post(f"{PREFIX}/workbooks/{self.wb_id}/recalculate").get_json()
        self.assertIn("workbook", recalculated)

    def test_filters_sort_and_search(self):
        self.seed()
        filters = json.dumps([{"col": 1, "op": "contains", "value": "Ad"}])
        filtered = self.grid(filters=filters)
        self.assertLess(filtered["total_rows"], 100)
        values = [cell["v"] for key, cell in filtered["cells"].items() if key.endswith(":1")]
        self.assertTrue(any(value == "Ada" for value in values))

        sorted_grid = self.grid(sort=json.dumps({"col": 2, "direction": "desc"}))
        self.assertEqual(sorted_grid["total_rows"], 100)

        search = self.client.get(
            f"{PREFIX}/workbooks/{self.wb_id}/search", query_string={"q": "Grace"}
        ).get_json()
        self.assertTrue(search["results"])
        hit = search["results"][0]
        self.assertEqual((hit["row"], hit["col"]), (3, 1))

        global_hits = self.client.get(f"{PREFIX}/search", query_string={"q": "Grace"}).get_json()
        self.assertTrue(any(item["workbook_id"] == self.wb_id for item in global_hits["results"]))

    def test_versions_checkpoint_and_restore(self):
        self.seed()
        checkpoint = self.client.post(
            f"{PREFIX}/workbooks/{self.wb_id}/versions", json={"label": "Known good"}
        )
        self.assertEqual(checkpoint.status_code, 201)
        version = checkpoint.get_json()["version"]
        self.ops({"op": "set_cell", "sheet": self.sheet_id, "row": 2, "col": 1, "value": "Changed"})
        self.assertEqual(self.grid()["cells"]["2:1"]["v"], "Changed")

        restored = self.client.post(
            f"{PREFIX}/workbooks/{self.wb_id}/versions/{version['id']}/restore"
        )
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(self.grid()["cells"]["2:1"]["v"], "Ada")

        listed = self.client.get(f"{PREFIX}/workbooks/{self.wb_id}/versions").get_json()
        self.assertTrue(listed["versions"])
        self.assertIn("can_undo", listed)

    def test_layout_sheets_and_structural_ops(self):
        self.seed()
        self.ops(
            {"op": "set_layout", "sheet": self.sheet_id, "frozen_rows": 1, "frozen_cols": 1},
            {"op": "add_sheet", "sheet": self.sheet_id, "after": 1},
        )
        workbook = self.client.get(f"{PREFIX}/workbooks/{self.wb_id}").get_json()["workbook"]
        self.assertEqual(len(workbook["sheets"]), 2)
        sheet = next(item for item in workbook["sheets"] if item["id"] == self.sheet_id)
        self.assertEqual(sheet["frozen_rows"], 1)
        second = [item for item in workbook["sheets"] if item["id"] != self.sheet_id][0]
        self.ops({"op": "rename_sheet", "sheet": second["id"], "name": "Renamed"})
        renamed = self.client.get(f"{PREFIX}/workbooks/{self.wb_id}").get_json()["workbook"]
        self.assertIn("Renamed", [item["name"] for item in renamed["sheets"]])

        self.ops({"op": "insert_rows", "sheet": self.sheet_id, "at": 2, "count": 1})
        shifted = self.grid(rows=10)
        self.assertEqual(shifted["cells"]["3:1"]["v"], "Ada")
        self.ops({"op": "delete_rows", "sheet": self.sheet_id, "at": 2, "count": 1})
        self.assertEqual(self.grid(rows=10)["cells"]["2:1"]["v"], "Ada")

    def test_analysis_apply_and_clear(self):
        self.ops(
            {"op": "set_cell", "sheet": self.sheet_id, "row": 1, "col": 1, "value": "Name"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 1, "col": 2, "value": "Email"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 1, "col": 3, "value": "Score"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 2, "col": 1, "value": "Ada"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 2, "col": 2, "value": "not-an-email"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 2, "col": 3, "value": "abc"},
        )
        response = self.client.post(
            f"{PREFIX}/workbooks/{self.wb_id}/analysis", json={"sheet_id": self.sheet_id}
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        analysis = response.get_json()["analysis"]
        self.assertEqual(analysis["sheet_id"], self.sheet_id)
        self.assertIn("issues", analysis)
        self.assertGreater(analysis["analyzed_rows"], 0)

        stored = self.client.get(f"{PREFIX}/workbooks/{self.wb_id}/analysis").get_json()
        self.assertEqual(stored["analysis"]["sheet_id"], self.sheet_id)

        issues = [issue for issue in analysis["issues"] if issue.get("suggestion") is not None and issue.get("col")]
        if issues:
            issue = issues[0]
            applied = self.client.post(
                f"{PREFIX}/workbooks/{self.wb_id}/analysis/apply",
                json={
                    "sheet_id": self.sheet_id,
                    "row": issue.get("sheet_row"),
                    "col": issue["col"],
                    "value": issue["suggestion"],
                },
            )
            self.assertEqual(applied.status_code, 200, applied.get_data(as_text=True))
            self.assertTrue(applied.get_json()["stale"])

        cleared = self.client.delete(f"{PREFIX}/workbooks/{self.wb_id}/analysis")
        self.assertEqual(cleared.status_code, 200)
        self.assertIsNone(
            self.client.get(f"{PREFIX}/workbooks/{self.wb_id}/analysis").get_json()["analysis"]
        )

    def test_dashboard_reports_and_finding_status(self):
        self.ops(
            {"op": "set_cell", "sheet": self.sheet_id, "row": 1, "col": 1, "value": "Name"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 1, "col": 2, "value": "Email"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 1, "col": 3, "value": "Score"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 2, "col": 1, "value": "Ada"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 2, "col": 2, "value": "not-an-email"},
            {"op": "set_cell", "sheet": self.sheet_id, "row": 2, "col": 3, "value": "abc"},
        )
        analysis = self.client.post(
            f"{PREFIX}/workbooks/{self.wb_id}/analysis", json={"sheet_id": self.sheet_id}
        ).get_json()["analysis"]
        issues = analysis["issues"]
        self.assertTrue(issues)
        rejected = self.client.post(
            f"{PREFIX}/workbooks/{self.wb_id}/analysis/status",
            json={"status": "rejected", "ids": [issues[0]["id"]]},
        )
        self.assertEqual(rejected.status_code, 200, rejected.get_data(as_text=True))
        self.assertEqual(rejected.get_json()["changed"], 1)
        stored = self.client.get(f"{PREFIX}/workbooks/{self.wb_id}/analysis").get_json()["analysis"]
        self.assertTrue(next(item for item in stored["issues"] if item["id"] == issues[0]["id"])["rejected"])

        correctable = [item for item in issues
                       if item.get("suggestion") is not None and item.get("col")][:2]
        if correctable:
            applied = self.client.post(
                f"{PREFIX}/workbooks/{self.wb_id}/analysis/apply",
                json={"corrections": [
                    {"sheet_id": self.sheet_id, "row": item["sheet_row"],
                     "col": item["col"], "value": item["suggestion"]}
                    for item in correctable
                ]},
            )
            self.assertEqual(applied.status_code, 200, applied.get_data(as_text=True))
            self.assertEqual(applied.get_json()["applied"], len(correctable))

        dashboard = self.client.get(f"{PREFIX}/dashboard")
        self.assertEqual(dashboard.status_code, 200, dashboard.get_data(as_text=True))
        totals = dashboard.get_json()["totals"]
        self.assertGreaterEqual(totals["files"], 1)
        self.assertGreaterEqual(totals["records"], 1)
        self.assertGreaterEqual(totals["issues"], len(issues))

        reports = self.client.get(f"{PREFIX}/reports").get_json()
        report = next(item for item in reports["reports"] if item["workbook_id"] == self.wb_id)
        self.assertEqual(report["total"], len(issues))
        self.assertEqual(report["rejected"], 1)
        if correctable:
            self.assertEqual(report["applied"], len(correctable))

    def test_exports_print_and_history(self):
        self.seed()
        formats = self.client.get(f"{PREFIX}/export/formats").get_json()["formats"]
        self.assertIn("csv", formats)

        for fmt in ("csv", "xlsx", "xls", "ods", "pdf", "docx", "json"):
            response = self.client.post(
                f"{PREFIX}/workbooks/{self.wb_id}/export",
                json={"format": fmt, "sheet_ids": [self.sheet_id]},
            )
            self.assertEqual(response.status_code, 201, f"{fmt}: {response.get_data(as_text=True)}")
            record = response.get_json()["export"]
            self.assertTrue(record["filename"].lower().endswith("." + ("xlsx" if fmt == "xlsm" else fmt)))
            download = self.client.get(f"{PREFIX}/export/{record['id']}/download")
            self.assertEqual(download.status_code, 200)
            self.assertGreater(len(download.data), 20)

        history = self.client.get(f"{PREFIX}/exports", query_string={"workbook_id": self.wb_id}).get_json()
        self.assertGreaterEqual(len(history["exports"]), 7)

        printed = self.client.get(
            f"{PREFIX}/workbooks/{self.wb_id}/print", query_string={"sheet_id": self.sheet_id}
        )
        self.assertEqual(printed.status_code, 200)
        self.assertIn("Ada", printed.get_data(as_text=True))

        no_original = self.client.get(f"{PREFIX}/workbooks/{self.wb_id}/original")
        self.assertEqual(no_original.status_code, 404)

    def test_upload_export_and_original(self):
        payload = make_xlsx([
            ["Name", "Email", "Phone"],
            ["Ada Lovelace", "ada@example.com", "555-0100"],
            ["Grace Hopper", "grace@example.com", "555-0101"],
        ])
        response = self.client.post(
            f"{PREFIX}/upload",
            data={"files": (payload, "people.xlsx")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        created = response.get_json()["created"][0]
        self.assertEqual(created["original_name"], "people.xlsx")
        self.assertEqual(created["sheet_count"], 1)
        self.assertIsNotNone(created.get("analysis_summary"))
        stored = self.client.get(f"{PREFIX}/workbooks/{created['id']}/analysis").get_json()["analysis"]
        self.assertIsNotNone(stored)
        self.assertGreaterEqual(stored["analyzed_rows"], 2)

        grid = self.client.get(
            f"{PREFIX}/workbooks/{created['id']}/grid", query_string={"rows": 10, "cols": 5}
        ).get_json()
        self.assertEqual(grid["cells"]["2:1"]["v"], "Ada Lovelace")

        original = self.client.get(f"{PREFIX}/workbooks/{created['id']}/original")
        self.assertEqual(original.status_code, 200)
        self.assertGreater(len(original.data), 100)

        exported = self.client.post(
            f"{PREFIX}/workbooks/{created['id']}/export", json={"format": "csv"}
        ).get_json()["export"]
        downloaded = self.client.get(f"{PREFIX}/export/{exported['id']}/download")
        self.assertIn("Ada Lovelace", downloaded.data.decode("utf-8-sig"))

    def test_upload_rejects_unsupported_and_empty(self):
        response = self.client.post(
            f"{PREFIX}/upload",
            data={"files": (io.BytesIO(b"hello"), "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["created"], [])

        empty = self.client.post(
            f"{PREFIX}/upload",
            data={"files": (io.BytesIO(b""), "empty.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(empty.status_code, 400)

    def test_import_path_save_as_and_duplicate(self):
        source = _TMP / "import_me.xlsx"
        source.write_bytes(make_xlsx([["Alpha", "Beta"], [1, 2]]).getvalue())
        response = self.client.post(f"{PREFIX}/import-path", json={"path": str(source)})
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        imported = response.get_json()["workbook"]
        self.assertEqual(imported["original_name"], "import_me.xlsx")

        rel = self.client.post(f"{PREFIX}/import-path", json={"path": "relative.xlsx"})
        self.assertEqual(rel.status_code, 400)

        copy = self.client.post(
            f"{PREFIX}/workbooks/{self.wb_id}/save-as", json={"name": "Copy of API test"}
        )
        self.assertEqual(copy.status_code, 201)
        copy_id = copy.get_json()["workbook"]["id"]
        self.assertNotEqual(copy_id, self.wb_id)
        duplicated = self.client.post(
            f"{PREFIX}/workbooks/{self.wb_id}/duplicate", json={"name": "Duplicate"}
        )
        self.assertEqual(duplicated.status_code, 201)
        self.assertNotEqual(duplicated.get_json()["workbook"]["id"], self.wb_id)

    def test_delete_workbook_removes_only_that_workbook(self):
        other = self.client.post(
            f"{PREFIX}/workbooks", json={"name": "Keep this workbook"}
        ).get_json()["workbook"]
        folder = _TMP / "storage" / "workbooks" / self.wb_id
        backups = _TMP / "storage" / "backups" / self.wb_id
        self.assertTrue(folder.is_dir())
        self.assertTrue(backups.is_dir())

        deleted = self.client.delete(f"{PREFIX}/workbooks/{self.wb_id}")
        self.assertEqual(deleted.status_code, 200, deleted.get_data(as_text=True))
        self.assertEqual(deleted.get_json()["deleted"]["id"], self.wb_id)
        self.assertEqual(self.client.get(f"{PREFIX}/workbooks/{self.wb_id}").status_code, 404)
        self.assertFalse(folder.exists())
        self.assertFalse(backups.exists())

        listing = self.client.get(f"{PREFIX}/workbooks").get_json()["workbooks"]
        ids = [item["id"] for item in listing]
        self.assertNotIn(self.wb_id, ids)
        self.assertIn(other["id"], ids)
        self.assertTrue((_TMP / "storage" / "workbooks" / other["id"]).is_dir())

        missing = self.client.delete(f"{PREFIX}/workbooks/missing")
        self.assertEqual(missing.status_code, 404)

    def test_storage_settings_and_retention(self):
        root = self.client.get(f"{PREFIX}/state").get_json()["storage"]["root"]
        self.assertTrue(Path(root).is_dir())
        retention = self.client.post(f"{PREFIX}/storage/retention", json={"value": 42}).get_json()
        self.assertEqual(retention["version_retention"], 42)
        reset = self.client.post(f"{PREFIX}/storage", json={"reset": True}).get_json()
        self.assertIn("root", reset["storage"])
        rejected = self.client.post(f"{PREFIX}/storage", json={"root": "relative/path"})
        self.assertEqual(rejected.status_code, 400)


if __name__ == "__main__":
    unittest.main()

==============================================================================
FILE: tests/test_core.py
==============================================================================
"""Core tests for the DataFix workspace backend (formulas, import, editing)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

TEMP = Path(tempfile.mkdtemp(prefix="datafix-core-"))
os.environ["DATAFIX_DB"] = str(TEMP / "test.db")

from datafix import config  # noqa: E402

config.CONFIG_DIR = TEMP / "config"
config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
config.STORAGE_CONFIG = config.CONFIG_DIR / "storage.json"
config.DEFAULT_STORAGE_ROOT = TEMP / "storage"
config.STORAGE_CONFIG.write_text(
    '{"root": "' + str(TEMP / "storage").replace("\\", "\\\\") + '"}', encoding="utf-8"
)

from datafix import db, formulas, importer, workbooks, exporters  # noqa: E402


def setUpModule():
    # Re-apply the temporary paths; another test module may have changed config.
    config.DATABASE_PATH = TEMP / "test.db"
    config.CONFIG_DIR = TEMP / "config"
    config.STORAGE_CONFIG = config.CONFIG_DIR / "storage.json"
    config.DEFAULT_STORAGE_ROOT = TEMP / "storage"
    config.STORAGE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    config.STORAGE_CONFIG.write_text(
        '{"root": "' + str(TEMP / "storage").replace("\\", "\\\\") + '"}',
        encoding="utf-8",
    )
    db.init_db()


def grid(cells):
    """cells: {(row, col): raw} -> engine input."""
    out = {}
    for name, data in cells.items():
        entries = {}
        for key, value in data.items():
            if isinstance(value, str) and value.startswith("="):
                entries[key] = {"f": value}
            else:
                entries[key] = {"v": value}
        out[name] = entries
    return out


class FormulaTests(unittest.TestCase):
    def test_arithmetic_and_precedence(self):
        result = formulas.compute(grid({"S": {(1, 1): 2, (2, 1): 3, (3, 1): "=A1+A2*2",
                                              (4, 1): "=(A1+A2)*2"}}))
        self.assertEqual(result[("S", 3, 1)], "8")
        self.assertEqual(result[("S", 4, 1)], "10")

    def test_sum_if_vlookup(self):
        sheets = grid({"S": {
            (1, 1): "name", (1, 2): "score",
            (2, 1): "Ann", (2, 2): 90,
            (3, 1): "Bob", (3, 2): 40,
            (4, 1): "Cara", (4, 2): 70,
            (6, 1): "=SUM(B2:B4)",
            (7, 1): "=AVERAGE(B2:B4)",
            (8, 1): "=COUNTIF(B2:B4,\">50\")",
            (9, 1): "=SUMIF(B2:B4,\">50\")",
            (10, 1): "=VLOOKUP(\"Bob\",A2:B4,2,FALSE)",
            (11, 1): "=XLOOKUP(\"Cara\",A2:A4,B2:B4)",
            (12, 1): "=IF(B2>80,\"pass\",\"fail\")",
            (13, 1): "=UPPER(A2)&\"!\"",
            (14, 1): "=ROUND(AVERAGE(B2:B4),0)",
        }})
        result = formulas.compute(sheets)
        self.assertEqual(result[("S", 6, 1)], "200")
        self.assertEqual(result[("S", 7, 1)], "66.66666666666667")
        self.assertEqual(result[("S", 8, 1)], "2")
        self.assertEqual(result[("S", 9, 1)], "160")
        self.assertEqual(result[("S", 10, 1)], "40")
        self.assertEqual(result[("S", 11, 1)], "70")
        self.assertEqual(result[("S", 12, 1)], "pass")
        self.assertEqual(result[("S", 13, 1)], "ANN!")
        self.assertEqual(result[("S", 14, 1)], "67")

    def test_cross_sheet_and_cycle(self):
        result = formulas.compute({
            "Sheet1": {(1, 1): {"v": 3}, (2, 1): {"f": "=A1", "v": None}},
            "Sheet2": {(1, 1): {"f": "=Sheet1!A1*2"}, (2, 1): {"f": "=A3"},
                       (3, 1): {"f": "=A2"}},
        })
        self.assertEqual(result[("Sheet2", 1, 1)], "6")
        self.assertEqual(result[("Sheet2", 2, 1)], "#CYCLE!")
        self.assertEqual(result[("Sheet1", 2, 1)], "3")

    def test_errors(self):
        result = formulas.compute(grid({"S": {
            (1, 1): "=1/0", (2, 1): "=NOPE(1)", (3, 1): "=A9+1", (4, 1): "=\"a\"+1",
        }}))
        self.assertEqual(result[("S", 1, 1)], "#DIV/0!")
        self.assertEqual(result[("S", 2, 1)], "#NAME?")
        self.assertEqual(result[("S", 3, 1)], "1")
        self.assertEqual(result[("S", 4, 1)], "#VALUE!")


class ImporterTests(unittest.TestCase):
    def test_csv_leading_zero(self):
        path = TEMP / "people.csv"
        path.write_text("phone,name\n09876543211,Ann\n, Bob\n", encoding="utf-8")
        doc = importer.import_file(path, ".csv")
        cells = doc["sheets"][0]["cells"]
        self.assertEqual(cells[(2, 1)]["v"], "09876543211")
        self.assertEqual(cells[(2, 2)]["v"], "Ann")

    def test_xlsx_styles_and_formulas(self):
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill

        path = TEMP / "styled.xlsx"
        book = Workbook()
        sheet = book.active
        sheet.title = "Data"
        sheet["A1"] = "n"
        sheet["A2"] = 5
        sheet["A3"] = 7
        sheet["A4"] = "=SUM(A2:A3)"
        sheet["A1"].font = Font(bold=True)
        sheet["A2"].fill = PatternFill("solid", fgColor="FFEEDD")
        sheet.merge_cells("C1:D1")
        book.save(path)

        doc = importer.import_file(path, ".xlsx")
        sheet_doc = doc["sheets"][0]
        self.assertTrue(sheet_doc["cells"][(1, 1)]["s"]["bold"])
        self.assertEqual(sheet_doc["cells"][(2, 1)]["s"]["fill"], "#FFEEDD")
        self.assertEqual(sheet_doc["cells"][(4, 1)]["f"], "=SUM(A2:A3)")
        self.assertEqual(sheet_doc["merges"][0], {"r1": 1, "c1": 3, "r2": 1, "c2": 4})

    def test_rejects_garbage(self):
        path = TEMP / "bad.xlsx"
        path.write_bytes(b"this is not a workbook")
        with self.assertRaises(ValueError):
            importer.import_file(path, ".xlsx")


class WorkbookTests(unittest.TestCase):
    def setUp(self):
        config.reset_storage_root()
        import sqlite3
        conn = sqlite3.connect(config.DATABASE_PATH)
        conn.execute("DELETE FROM cells")
        conn.execute("DELETE FROM sheets")
        conn.execute("DELETE FROM batches")
        conn.execute("DELETE FROM versions")
        conn.execute("DELETE FROM exports")
        conn.execute("DELETE FROM workbooks")
        conn.commit()
        conn.close()

    def test_blank_edit_undo_redo(self):
        wb = workbooks.create_blank("Budget")
        self.assertEqual(wb["name"], "Budget")
        sheet = wb["sheets"][0]
        meta = workbooks.apply_ops(wb["id"], [
            {"op": "set_cell", "sheet": sheet["id"], "row": 1, "col": 1, "value": "item"},
            {"op": "set_cell", "sheet": sheet["id"], "row": 1, "col": 2, "value": "cost"},
            {"op": "set_cell", "sheet": sheet["id"], "row": 2, "col": 1, "value": "tea"},
            {"op": "set_cell", "sheet": sheet["id"], "row": 2, "col": 2, "value": 3.5},
            {"op": "set_cell", "sheet": sheet["id"], "row": 3, "col": 2, "formula": "=B2*2"},
        ])
        self.assertTrue(meta["can_undo"])
        page = workbooks.read_grid(wb["id"], sheet["id"], rows=10, cols=5)
        self.assertEqual(page["cells"]["2:2"]["v"], "3.5")
        self.assertEqual(page["cells"]["3:2"]["v"], "7")
        self.assertEqual(page["cells"]["3:2"]["f"], "=B2*2")
        self.assertEqual(meta["affected_rows"], [1, 2, 3])

        workbooks.undo(wb["id"])
        page = workbooks.read_grid(wb["id"], sheet["id"], rows=10, cols=5)
        self.assertNotIn("3:2", page["cells"])
        workbooks.redo(wb["id"])
        page = workbooks.read_grid(wb["id"], sheet["id"], rows=10, cols=5)
        self.assertEqual(page["cells"]["3:2"]["v"], "7")

        # second edit keeps undo stack ordered
        workbooks.apply_ops(wb["id"], [
            {"op": "set_cell", "sheet": sheet["id"], "row": 4, "col": 1, "value": "coffee"},
        ])
        workbooks.undo(wb["id"])
        page = workbooks.read_grid(wb["id"], sheet["id"], rows=10, cols=5)
        self.assertNotIn("4:1", page["cells"])
        self.assertIn("2:2", page["cells"])

    def test_structure_undo_redo(self):
        wb = workbooks.create_blank()
        sid = wb["sheets"][0]["id"]
        workbooks.apply_ops(wb["id"], [
            {"op": "set_cell", "sheet": sid, "row": 5, "col": 1, "value": "five"},
            {"op": "set_cell", "sheet": sid, "row": 5, "col": 2, "value": "B5"},
        ])
        workbooks.apply_ops(wb["id"], [
            {"op": "insert_rows", "sheet": sid, "at": 2, "count": 2},
        ])
        page = workbooks.read_grid(wb["id"], sid, rows=20, cols=5)
        self.assertEqual(page["cells"]["7:1"]["v"], "five")
        workbooks.undo(wb["id"])
        page = workbooks.read_grid(wb["id"], sid, rows=20, cols=5)
        self.assertEqual(page["cells"]["5:1"]["v"], "five")
        workbooks.redo(wb["id"])
        page = workbooks.read_grid(wb["id"], sid, rows=20, cols=5)
        self.assertEqual(page["cells"]["7:2"]["v"], "B5")

        workbooks.apply_ops(wb["id"], [
            {"op": "insert_cols", "sheet": sid, "at": 1, "count": 1},
        ])
        page = workbooks.read_grid(wb["id"], sid, rows=20, cols=8)
        self.assertEqual(page["cells"]["7:2"]["v"], "five")
        workbooks.undo(wb["id"])
        page = workbooks.read_grid(wb["id"], sid, rows=20, cols=8)
        self.assertEqual(page["cells"]["7:1"]["v"], "five")

        workbooks.apply_ops(wb["id"], [
            {"op": "delete_rows", "sheet": sid, "at": 2, "count": 1},
        ])
        page = workbooks.read_grid(wb["id"], sid, rows=20, cols=5)
        self.assertEqual(page["cells"]["6:1"]["v"], "five")
        workbooks.undo(wb["id"])
        page = workbooks.read_grid(wb["id"], sid, rows=20, cols=5)
        self.assertEqual(page["cells"]["7:1"]["v"], "five")

        workbooks.apply_ops(wb["id"], [
            {"op": "delete_cols", "sheet": sid, "at": 1, "count": 1},
        ])
        page = workbooks.read_grid(wb["id"], sid, rows=20, cols=8)
        self.assertEqual(page["cells"]["7:1"]["v"], "B5")
        workbooks.undo(wb["id"])
        page = workbooks.read_grid(wb["id"], sid, rows=20, cols=8)
        self.assertEqual(page["cells"]["7:1"]["v"], "five")

    def test_sheets_undo_redo_keep_ids(self):
        wb = workbooks.create_blank()
        first = wb["sheets"][0]["id"]
        meta = workbooks.apply_ops(wb["id"], [
            {"op": "add_sheet", "name": "Extra", "after": 1},
        ])
        wb = workbooks.get_workbook(wb["id"])
        extra = [s for s in wb["sheets"] if s["name"] == "Extra"][0]
        extra_id = extra["id"]
        workbooks.apply_ops(wb["id"], [
            {"op": "set_cell", "sheet": extra_id, "row": 1, "col": 1, "value": "hello"},
        ])
        workbooks.undo(wb["id"])  # undo cell
        workbooks.undo(wb["id"])  # undo add_sheet
        wb = workbooks.get_workbook(wb["id"])
        self.assertEqual([s["name"] for s in wb["sheets"]], ["Sheet1"])
        workbooks.redo(wb["id"])  # redo add_sheet
        wb = workbooks.get_workbook(wb["id"])
        extra = [s for s in wb["sheets"] if s["name"] == "Extra"][0]
        self.assertEqual(extra["id"], extra_id)
        workbooks.redo(wb["id"])  # redo cell
        page = workbooks.read_grid(wb["id"], extra_id, rows=5, cols=5)
        self.assertEqual(page["cells"]["1:1"]["v"], "hello")

        workbooks.apply_ops(wb["id"], [
            {"op": "rename_sheet", "sheet": first, "name": "Main"},
            {"op": "move_sheet", "sheet": extra_id, "to": 0},
        ])
        wb = workbooks.get_workbook(wb["id"])
        self.assertEqual(wb["sheets"][0]["id"], extra_id)
        workbooks.undo(wb["id"])
        workbooks.undo(wb["id"])
        wb = workbooks.get_workbook(wb["id"])
        self.assertEqual(wb["sheets"][1]["name"], "Extra")

    def test_versions_restore(self):
        wb = workbooks.create_blank("History")
        sid = wb["sheets"][0]["id"]
        workbooks.apply_ops(wb["id"], [
            {"op": "set_cell", "sheet": sid, "row": 1, "col": 1, "value": "one"},
        ], version=True)
        versions, pending = workbooks.list_versions(wb["id"])
        self.assertGreaterEqual(len(versions), 2)  # Imported/Created + edit
        self.assertGreaterEqual(pending, 1)

        workbooks.apply_ops(wb["id"], [
            {"op": "set_cell", "sheet": sid, "row": 1, "col": 1, "value": "two"},
        ], version=True)
        page = workbooks.read_grid(wb["id"], sid, rows=3, cols=3)
        self.assertEqual(page["cells"]["1:1"]["v"], "two")

        target = [v for v in versions if v["label"] == "Edit"][0]
        workbooks.restore_version(wb["id"], target["id"])
        page = workbooks.read_grid(wb["id"], sid, rows=3, cols=3)
        self.assertEqual(page["cells"]["1:1"]["v"], "one")

    def test_search_filter_archive_duplicate(self):
        wb = workbooks.create_blank("Find me")
        sid = wb["sheets"][0]["id"]
        workbooks.apply_ops(wb["id"], [
            {"op": "set_cell", "sheet": sid, "row": 1, "col": 1, "value": "name"},
            {"op": "set_cell", "sheet": sid, "row": 1, "col": 2, "value": "city"},
            {"op": "set_cell", "sheet": sid, "row": 2, "col": 1, "value": "Ann"},
            {"op": "set_cell", "sheet": sid, "row": 2, "col": 2, "value": "Mumbai"},
            {"op": "set_cell", "sheet": sid, "row": 3, "col": 1, "value": "Bob"},
            {"op": "set_cell", "sheet": sid, "row": 3, "col": 2, "value": "Pune"},
        ])
        hits = workbooks.search_in_workbook(wb["id"], "mum")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["header"], "city")
        page = workbooks.read_grid(wb["id"], sid, rows=10, cols=5, filters=[
            {"col": 2, "op": "contains", "value": "u"},
        ])
        self.assertEqual(page["total_rows"], 3)  # header + 2 matches

        workbooks.rename_workbook(wb["id"], "Searchable")
        workbooks.set_flags(wb["id"], starred=True)
        starred = workbooks.list_workbooks("starred", "search")
        self.assertEqual([w["id"] for w in starred], [wb["id"]])
        workbooks.set_flags(wb["id"], archived=True)
        self.assertEqual(workbooks.list_workbooks("archived")[0]["id"], wb["id"])
        self.assertEqual(workbooks.list_workbooks("all"), [])

        copy = workbooks.duplicate_workbook(wb["id"], "Copy")
        page = workbooks.read_grid(copy["id"], copy["sheets"][0]["id"], rows=5, cols=5)
        self.assertEqual(page["cells"]["2:1"]["v"], "Ann")
        workbooks.apply_ops(copy["id"], [
            {"op": "set_cell", "sheet": copy["sheets"][0]["id"], "row": 2, "col": 1,
             "value": "Zoe"},
        ])
        page = workbooks.read_grid(wb["id"], sid, rows=5, cols=5)
        self.assertEqual(page["cells"]["2:1"]["v"], "Ann")

    def test_original_file_untouched(self):
        wb = workbooks.create_blank("Originals")
        sid = wb["sheets"][0]["id"]
        workbooks.apply_ops(wb["id"], [
            {"op": "set_cell", "sheet": sid, "row": 1, "col": 1, "value": "x"},
        ])
        original = workbooks.original_file(wb["id"])
        self.assertIsNone(original)

        src = TEMP / "keep.csv"
        src.write_text("a,b\n1,2\n", encoding="utf-8")
        checksum = hashlib.sha256(src.read_bytes()).hexdigest()
        imported = workbooks.create_from_file(src, "keep.csv", ".csv")
        saved = workbooks.original_file(imported["id"])
        self.assertIsNotNone(saved)
        self.assertEqual(hashlib.sha256(saved.read_bytes()).hexdigest(), checksum)
        workbooks.apply_ops(imported["id"], [
            {"op": "set_cell", "sheet": imported["sheets"][0]["id"], "row": 1, "col": 1,
             "value": "changed"},
        ])
        self.assertEqual(hashlib.sha256(saved.read_bytes()).hexdigest(), checksum)
        self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(), checksum)


class ExportTests(unittest.TestCase):
    def setUp(self):
        config.reset_storage_root()
        import sqlite3
        conn = sqlite3.connect(config.DATABASE_PATH)
        for table in ("cells", "sheets", "batches", "versions", "exports", "workbooks"):
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
        conn.close()
        wb = workbooks.create_blank("Report card")
        self.wb_id = wb["id"]
        self.sid = wb["sheets"][0]["id"]
        workbooks.apply_ops(self.wb_id, [
            {"op": "set_cell", "sheet": self.sid, "row": 1, "col": 1, "value": "name",
             "style": {"bold": True}},
            {"op": "set_cell", "sheet": self.sid, "row": 1, "col": 2, "value": "city"},
            {"op": "set_cell", "sheet": self.sid, "row": 1, "col": 3, "value": "score"},
            {"op": "set_cell", "sheet": self.sid, "row": 2, "col": 1, "value": "Ann"},
            {"op": "set_cell", "sheet": self.sid, "row": 2, "col": 2, "value": "Mumbai"},
            {"op": "set_cell", "sheet": self.sid, "row": 2, "col": 3, "value": 90},
            {"op": "set_cell", "sheet": self.sid, "row": 3, "col": 1, "value": "Bob"},
            {"op": "set_cell", "sheet": self.sid, "row": 3, "col": 2, "value": "Pune"},
            {"op": "set_cell", "sheet": self.sid, "row": 3, "col": 3, "value": 40},
            {"op": "set_cell", "sheet": self.sid, "row": 4, "col": 1, "value": "Cara"},
            {"op": "set_cell", "sheet": self.sid, "row": 4, "col": 2, "value": "Lucknow"},
            {"op": "set_cell", "sheet": self.sid, "row": 4, "col": 3, "value": 70},
            {"op": "set_cell", "sheet": self.sid, "row": 5, "col": 3, "formula": "=SUM(C2:C4)"},
            {"op": "add_sheet", "name": "Notes", "after": 1},
        ])

    def _path(self, record):
        return exporters.get_export(record["id"])["path"]

    def test_all_formats(self):
        from openpyxl import load_workbook

        xlsx = exporters.export(self.wb_id, "xlsx")
        book = load_workbook(self._path(xlsx))
        self.assertEqual(book.sheetnames, ["Sheet1", "Notes"])
        sheet = book["Sheet1"]
        self.assertEqual(sheet["A2"].value, "Ann")
        self.assertEqual(sheet["C5"].value, "=SUM(C2:C4)")
        self.assertTrue(sheet["A1"].font.bold)
        self.assertEqual(sheet["C2"].value, 90)

        xlsm = exporters.export(self.wb_id, "xlsm")
        self.assertTrue(xlsm["filename"].endswith(".xlsm"))
        self.assertIn("macros", xlsm["notes"][0].lower())

        xls = exporters.export(self.wb_id, "xls")
        import xlrd
        old = xlrd.open_workbook(str(self._path(xls)))
        self.assertEqual(old.sheet_by_index(0).cell_value(1, 0), "Ann")

        csv_record = exporters.export(self.wb_id, "csv", [self.sid])
        raw = self._path(csv_record).read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        text = raw.decode("utf-8-sig")
        self.assertIn("Ann,Mumbai,90", text)

        ods = exporters.export(self.wb_id, "ods")
        round_trip = importer.import_file(self._path(ods), ".ods")
        values = round_trip["sheets"][0]["cells"]
        self.assertEqual(values[(2, 1)]["v"], "Ann")
        self.assertEqual(values[(2, 3)]["v"], 90)

        pdf = exporters.export(self.wb_id, "pdf")
        self.assertTrue(self._path(pdf).read_bytes().startswith(b"%PDF"))

        docx = exporters.export(self.wb_id, "docx")
        from docx import Document
        document = Document(str(self._path(docx)))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        self.assertIn("Report card", text)
        table_text = "\n".join(cell.text for row in document.tables[0].rows for cell in row.cells)
        self.assertIn("Mumbai", table_text)

        data = exporters.export(self.wb_id, "json")
        payload = json.loads(self._path(data).read_text(encoding="utf-8"))
        self.assertEqual(payload["workbook"]["name"], "Report card")
        self.assertEqual(len(payload["sheets"]), 2)
        self.assertTrue(any(cell.get("formula") == "=SUM(C2:C4)"
                            for cell in payload["sheets"][0]["cells"]))

        history = exporters.history(self.wb_id)
        self.assertEqual(len(history), 8)
        self.assertEqual(history[0]["format"], "json")

    def test_filtered_and_selected_export(self):
        record = exporters.export(self.wb_id, "csv", [self.sid], {
            "filters": [{"col": 2, "op": "equals", "value": "Mumbai"}],
        })
        text = self._path(record).read_text(encoding="utf-8-sig")
        lines = [line for line in text.splitlines() if line]
        self.assertEqual(lines[0], "name,city,score")
        self.assertIn("Ann,Mumbai,90", text)
        self.assertNotIn("Pune", text)
        self.assertNotIn("Lucknow", text)

        row_record = exporters.export(self.wb_id, "xlsx", [self.sid], {
            "rows": [1, 3], "columns": [1, 2],
        })
        from openpyxl import load_workbook
        sheet = load_workbook(self._path(row_record))["Sheet1"]
        self.assertEqual([sheet["A1"].value, sheet["B1"].value], ["name", "city"])
        self.assertEqual([sheet["A2"].value, sheet["B2"].value], ["Bob", "Pune"])
        self.assertEqual(sheet.max_row, 2)

    def test_export_history_and_missing_file(self):
        record = exporters.export(self.wb_id, "csv", [self.sid])
        path = self._path(record)
        self.assertTrue(path.is_file())
        path.unlink()
        with self.assertRaises(LookupError):
            exporters.get_export(record["id"])
        with self.assertRaises(ValueError):
            exporters.export(self.wb_id, "csv", [999999])


class RestartTests(unittest.TestCase):
    def test_data_survives_new_process(self):
        config.reset_storage_root()
        wb = workbooks.create_blank("Persistent")
        workbooks.apply_ops(wb["id"], [
            {"op": "set_cell", "sheet": wb["sheets"][0]["id"], "row": 1, "col": 1,
             "value": "survives"},
        ])
        import subprocess
        import sys
        script = (
            "import os; os.environ['DATAFIX_DB']=r'%s';"
            "from datafix import workbooks;"
            "wb=workbooks.require_workbook(r'%s');"
            "page=workbooks.read_grid(r'%s', wb['sheets'][0]['id']);"
            "print(page['cells']['1:1']['v'])" % (config.DATABASE_PATH, wb["id"], wb["id"])
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=str(Path(__file__).parent.parent),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "survives")


if __name__ == "__main__":
    unittest.main()


