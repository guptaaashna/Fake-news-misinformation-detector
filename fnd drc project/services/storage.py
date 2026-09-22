"""SQLite persistence for completed URL analyses."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "truthshield.db"


def _connect(database_path: str | Path | None = None) -> sqlite3.Connection:
	path = Path(database_path) if database_path else DEFAULT_DATABASE_PATH
	path.parent.mkdir(parents=True, exist_ok=True)
	connection = sqlite3.connect(path)
	connection.row_factory = sqlite3.Row
	return connection


def _initialize_database(connection: sqlite3.Connection) -> None:
	connection.execute(
		"""
		CREATE TABLE IF NOT EXISTS analyses (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL,
			url TEXT NOT NULL,
			title TEXT NOT NULL,
			result_json TEXT NOT NULL,
			text_statistics_json TEXT NOT NULL
		)
		"""
	)
	connection.commit()


def save_analysis(result: dict, database_path: str | Path | None = None) -> int:
	"""Save one completed analysis and return its database id."""
	timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
	text_statistics = (result.get("text_processing") or {}).get("statistics", {})
	connection = _connect(database_path)
	try:
		_initialize_database(connection)
		cursor = connection.execute(
			"""
			INSERT INTO analyses (timestamp, url, title, result_json, text_statistics_json)
			VALUES (?, ?, ?, ?, ?)
			""",
			(
				timestamp,
				result.get("url", ""),
				result.get("title", ""),
				json.dumps(result),
				json.dumps(text_statistics),
			),
		)
		connection.commit()
		return int(cursor.lastrowid)
	finally:
		connection.close()


def load_history(database_path: str | Path | None = None) -> list[dict]:
	"""Return saved analyses newest first."""
	connection = _connect(database_path)
	try:
		_initialize_database(connection)
		rows = connection.execute(
			"SELECT id, timestamp, url, title, result_json, text_statistics_json "
			"FROM analyses ORDER BY timestamp DESC, id DESC"
		).fetchall()
	finally:
		connection.close()
	history = []
	for row in rows:
		result = json.loads(row["result_json"])
		result["history_id"] = row["id"]
		result["timestamp"] = row["timestamp"]
		result["url"] = row["url"]
		result["title"] = row["title"]
		result["text_statistics"] = json.loads(row["text_statistics_json"])
		history.append(result)
	return history
