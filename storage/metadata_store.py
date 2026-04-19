"""
Model metadata storage using SQLite for O(1) performance and scaling.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from ml_platform.config import get_storage_paths
from ml_platform.exceptions import StorageError
from ml_platform.models import ModelMetadata

logger = logging.getLogger(__name__)

class MetadataStore:
    """Persist and query model metadata (SQLite)."""

    def __init__(self, metadata_path: Optional[Path] = None):
        if metadata_path is None:
            _, metadata_path, _ = get_storage_paths()
        self._root = Path(metadata_path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._db_path = self._root / "metadata.db"
        self._init_db()

    def _get_connection(self):
        # check_same_thread=False since registry lock handles multi-threading scope safely
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high-concurrency access (industry-grade SQLite optimization)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT,
                    version TEXT,
                    model_type TEXT,
                    features TEXT,
                    target TEXT,
                    training_date TEXT,
                    performance_metrics TEXT,
                    status TEXT,
                    artifact_path TEXT
                )
            ''')
            # Indexes for extremely fast metadata queries even with 10k+ versions
            conn.execute('CREATE INDEX IF NOT EXISTS idx_model_name ON model_metadata(model_name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON model_metadata(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_type ON model_metadata(model_type)')
            conn.commit()
            
        # Seamlessly intercept any lingering JSON files from previous system version
        self._migrate_existing_json()

    def _migrate_existing_json(self):
        """Automatically ingests legacy unstructured JSON flat files into the SQLite database."""
        moved_dir = self._root / "migrated_json"
        
        for path in self._root.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                
                training_date = d.get("training_date")
                if isinstance(training_date, str):
                    try:
                        training_date = datetime.fromisoformat(training_date.replace("Z", "+00:00"))
                    except ValueError:
                        training_date = None
                
                meta = ModelMetadata(
                    model_id=d["model_id"],
                    model_name=d["model_name"],
                    version=d["version"],
                    model_type=d["model_type"],
                    features=d.get("features", []),
                    target=d.get("target", ""),
                    training_date=training_date,
                    performance_metrics=d.get("performance_metrics", {}),
                    status=d.get("status", "candidate"),
                    artifact_path=d.get("artifact_path", ""),
                )
                self.save_metadata(meta)
                
                moved_dir.mkdir(exist_ok=True)
                path.rename(moved_dir / path.name)
                logger.info(f"Migrated legacy {path.name} to highly-scalable SQLite.")
            except Exception as e:
                logger.warning(f"Failed migrating {path}: {e}")

    def _row_to_metadata(self, row) -> ModelMetadata:
        try:
            feats = json.loads(row["features"]) if row["features"] else []
        except Exception:
            feats = []
            
        try:
            metrics = json.loads(row["performance_metrics"]) if row["performance_metrics"] else {}
        except Exception:
            metrics = {}

        t_date = row["training_date"]
        if t_date:
            try:
                t_date = datetime.fromisoformat(t_date)
            except Exception:
                t_date = None
                
        return ModelMetadata(
            model_id=row["model_id"],
            model_name=row["model_name"],
            version=row["version"],
            model_type=row["model_type"],
            features=feats,
            target=row["target"],
            training_date=t_date,
            performance_metrics=metrics,
            status=row["status"],
            artifact_path=row["artifact_path"]
        )

    def save_metadata(self, metadata: ModelMetadata) -> None:
        """Saves model metadata to the ultra-fast SQLite store (O(1) insertion space)."""
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO model_metadata (
                        model_id, model_name, version, model_type, features, target, 
                        training_date, performance_metrics, status, artifact_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(model_id) DO UPDATE SET
                        model_name=excluded.model_name,
                        version=excluded.version,
                        model_type=excluded.model_type,
                        features=excluded.features,
                        target=excluded.target,
                        training_date=excluded.training_date,
                        performance_metrics=excluded.performance_metrics,
                        status=excluded.status,
                        artifact_path=excluded.artifact_path
                ''', (
                    metadata.model_id,
                    metadata.model_name,
                    metadata.version,
                    metadata.model_type,
                    json.dumps(metadata.features),
                    metadata.target,
                    metadata.training_date.isoformat() if metadata.training_date else None,
                    json.dumps(metadata.performance_metrics),
                    metadata.status,
                    metadata.artifact_path
                ))
                conn.commit()
            logger.debug("Metadata strictly persisted to DB", extra={"model_id": metadata.model_id})
        except Exception as e:
            logger.error("Failed to insert db metadata %s: %s", metadata.model_id, e, exc_info=True)
            raise StorageError(f"Failed DB ingestion: {e}") from e

    def load_metadata(self, model_id: str) -> ModelMetadata:
        """Loads specific metadata instantaneously via primary key idx."""
        logger.debug("Loading DB metadata for model_id=%s", model_id)
        try:
            with self._get_connection() as conn:
                cur = conn.execute('SELECT * FROM model_metadata WHERE model_id = ?', (model_id,))
                row = cur.fetchone()
                if not row:
                    raise StorageError(f"Metadata not found: {model_id}")
                return self._row_to_metadata(row)
        except StorageError:
            raise
        except Exception as e:
            logger.error("Failed DB fetch %s: %s", model_id, e, exc_info=True)
            raise StorageError(f"DB load failure: {e}") from e

    def query_metadata(self, filters: dict) -> List[ModelMetadata]:
        """Runs an indexed SQL query directly resolving the JSON-scavenge bottleneck."""
        query = "SELECT * FROM model_metadata"
        params = []
        conditions = []
        
        if filters.get("model_name"):
            conditions.append("model_name = ?")
            params.append(filters["model_name"])
        if filters.get("status"):
            conditions.append("status = ?")
            params.append(filters["status"])
        if filters.get("model_type"):
            conditions.append("model_type = ?")
            params.append(filters["model_type"])
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        results = []
        try:
            with self._get_connection() as conn:
                cur = conn.execute(query, params)
                for row in cur:
                    results.append(self._row_to_metadata(row))
            return results
        except Exception as e:
            logger.warning("Failed querying metadata with %s: %s", filters, e)
            return []

    def list_all_model_ids(self) -> List[str]:
        try:
            with self._get_connection() as conn:
                cur = conn.execute('SELECT model_id FROM model_metadata')
                return [row["model_id"] for row in cur]
        except Exception:
            return []

    def delete_metadata(self, model_id: str) -> None:
        try:
            with self._get_connection() as conn:
                conn.execute('DELETE FROM model_metadata WHERE model_id = ?', (model_id,))
                conn.commit()
            logger.info("Metadata successfully deleted from backend DB", extra={"model_id": model_id})
        except Exception as e:
            logger.error("Failed root deletion %s: %s", model_id, e)
