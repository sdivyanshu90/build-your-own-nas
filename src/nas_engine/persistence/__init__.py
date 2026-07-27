"""Persistence: database connection, versioned schema, and the repository seam.

Nothing outside this package constructs SQL or holds a SQLAlchemy session. The public
surface is :class:`~nas_engine.persistence.repository.SearchRepository` plus the detached
read models it returns.
"""

from nas_engine.persistence.database import Database
from nas_engine.persistence.migrations import (
    MIGRATIONS,
    TARGET_SCHEMA_VERSION,
    apply_migrations,
    current_version,
    ensure_schema,
)
from nas_engine.persistence.models import (
    ArtifactRecord,
    Base,
    CandidateRecord,
    CheckpointRecord,
    MetricRecord,
    SearchEventRecord,
    SearchRecord,
    SearchStatus,
    TrialRecord,
)
from nas_engine.persistence.repository import (
    CandidateSummary,
    RecoveryReport,
    SearchRepository,
    SearchSummary,
)

__all__ = [
    "MIGRATIONS",
    "TARGET_SCHEMA_VERSION",
    "ArtifactRecord",
    "Base",
    "CandidateRecord",
    "CandidateSummary",
    "CheckpointRecord",
    "Database",
    "MetricRecord",
    "RecoveryReport",
    "SearchEventRecord",
    "SearchRecord",
    "SearchRepository",
    "SearchStatus",
    "SearchSummary",
    "TrialRecord",
    "apply_migrations",
    "current_version",
    "ensure_schema",
]
