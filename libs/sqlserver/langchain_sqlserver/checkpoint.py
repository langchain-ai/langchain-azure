"""LangGraph checkpoint saver backed by SQL Server / Azure SQL.

Provides :class:`SQLServerSaver`, an implementation of
``langgraph.checkpoint.base.BaseCheckpointSaver`` that persists graph state
in SQL Server. Modeled after ``langgraph.checkpoint.sqlite.SqliteSaver`` so
behavior is consistent with the other first-party savers, but adapted for
SQL Server: SQLAlchemy + ``pyodbc`` (sync), MERGE for upsert semantics, and
the same Entra-ID-vs-uid/pwd connection conventions as
:class:`langchain_sqlserver.SQLServer_VectorStore`.

Example:
    ```python
    from langchain_sqlserver import SQLServerSaver

    saver = SQLServerSaver(
        connection_string=(
            "Driver={ODBC Driver 18 for SQL Server};Server=tcp:host,1433;"
            "Database=mydb;Uid=user;Pwd=pwd;TrustServerCertificate=yes;"
        ),
    )

    graph = builder.compile(checkpointer=saver)
    ```
"""

from __future__ import annotations

import json
import logging
import re
import struct
import threading
from contextlib import contextmanager
from typing import (
    Any,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from sqlalchemy import (
    Column,
    Dialect,
    Integer,
    LargeBinary,
    PrimaryKeyConstraint,
    create_engine,
    event,
    text,
)
from sqlalchemy.dialects.mssql import NVARCHAR
from sqlalchemy.engine import URL, Connection, Engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy.pool import ConnectionPoolEntry

AZURE_TOKEN_URL = "https://database.windows.net/.default"
EXTRA_PARAMS = ";Trusted_Connection=Yes"
SQL_COPT_SS_ACCESS_TOKEN = 1256

DEFAULT_CHECKPOINTS_TABLE = "checkpoints"
DEFAULT_WRITES_TABLE = "checkpoint_writes"


class SQLServerSaver(BaseCheckpointSaver[str]):
    """A LangGraph checkpoint saver that persists state to SQL Server.

    The saver stores two tables: ``checkpoints`` for serialized
    ``Checkpoint`` blobs (plus their parent linkage and metadata), and
    ``checkpoint_writes`` for the intermediate per-task writes attached to
    a checkpoint. Tables are created lazily on first use.

    The class supports the same connection-string conventions as
    :class:`langchain_sqlserver.SQLServer_VectorStore`. Pass either a
    pre-built ``Connection`` or a ``connection_string``; if the connection
    string carries no credentials and no ``Trusted_Connection=yes`` the
    saver authenticates with Entra ID.

    For shared-process use, the saver guards every cursor with an internal
    lock — mirroring the safety semantics of the first-party
    ``SqliteSaver`` — so concurrent graph runs in the same process cannot
    interleave statements on the underlying engine.
    """

    def __init__(
        self,
        *,
        connection: Optional[Connection] = None,
        connection_string: str = "",
        checkpoints_table: str = DEFAULT_CHECKPOINTS_TABLE,
        writes_table: str = DEFAULT_WRITES_TABLE,
        db_schema: Optional[str] = None,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        """Initialize the checkpoint saver.

        Args:
            connection: Optional pre-existing SQLAlchemy ``Connection``.
                When provided, ``connection_string`` is ignored for engine
                creation.
            connection_string: SQL Server ODBC connection string. Required
                if ``connection`` is not provided. If the string carries no
                username/password and no ``Trusted_Connection=yes``, Entra
                ID authentication is used.
            checkpoints_table: Name of the table that stores serialized
                checkpoints. Defaults to ``checkpoints``.
            writes_table: Name of the table that stores intermediate
                writes. Defaults to ``checkpoint_writes``.
            db_schema: Optional schema in which the tables live. The
                schema must already exist.
            serde: Optional serializer override. Defaults to
                ``JsonPlusSerializer``.
        """
        super().__init__(serde=serde)
        self._checkpoints_table = checkpoints_table
        self._writes_table = writes_table
        self._schema = db_schema
        self._cps_t = self._qualified(checkpoints_table, db_schema)
        self._wrt_t = self._qualified(writes_table, db_schema)
        self.connection_string = (
            self._get_connection_url(connection_string) if connection_string else ""
        )
        self._bind: Union[Connection, Engine] = (
            connection if connection else self._create_engine()
        )
        self._checkpoints_model = self._build_checkpoints_model(
            checkpoints_table, db_schema
        )
        self._writes_model = self._build_writes_model(writes_table, db_schema)
        self.lock = threading.Lock()
        self.is_setup = False

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _build_checkpoints_model(self, name: str, schema: Optional[str]) -> Any:
        Base = declarative_base(class_registry=dict())  # type: Any

        class Checkpoints(Base):
            __tablename__ = name
            __table_args__ = (
                PrimaryKeyConstraint(
                    "thread_id",
                    "checkpoint_ns",
                    "checkpoint_id",
                    mssql_clustered=True,
                ),
                {"schema": schema},
            )
            thread_id = Column(NVARCHAR(255), nullable=False)
            checkpoint_ns = Column(NVARCHAR(255), nullable=False, default="")
            checkpoint_id = Column(NVARCHAR(255), nullable=False)
            parent_checkpoint_id = Column(NVARCHAR(255), nullable=True)
            type = Column(NVARCHAR(255), nullable=True)
            checkpoint = Column(LargeBinary, nullable=True)
            # Stored as JSON text (NVARCHAR(MAX)) so metadata filtering via
            # JSON_VALUE works reliably; casting binary to NVARCHAR does not
            # round-trip UTF-8 JSON.
            metadata_ = Column("metadata", NVARCHAR(None), nullable=True)

        return Checkpoints

    def _build_writes_model(self, name: str, schema: Optional[str]) -> Any:
        Base = declarative_base(class_registry=dict())  # type: Any

        class Writes(Base):
            __tablename__ = name
            __table_args__ = (
                PrimaryKeyConstraint(
                    "thread_id",
                    "checkpoint_ns",
                    "checkpoint_id",
                    "task_id",
                    "idx",
                    mssql_clustered=True,
                ),
                {"schema": schema},
            )
            thread_id = Column(NVARCHAR(255), nullable=False)
            checkpoint_ns = Column(NVARCHAR(255), nullable=False, default="")
            checkpoint_id = Column(NVARCHAR(255), nullable=False)
            task_id = Column(NVARCHAR(255), nullable=False)
            # Integer so ORDER BY idx sorts numerically; some writes use
            # negative indices (see WRITES_IDX_MAP) and text ordering would
            # place e.g. '-1'/'10' before '2'.
            idx = Column(Integer, nullable=False)
            channel = Column(NVARCHAR(255), nullable=False)
            type = Column(NVARCHAR(255), nullable=True)
            blob = Column(LargeBinary, nullable=True)

        return Writes

    def setup(self) -> None:
        """Create the checkpoint tables if they do not already exist.

        Called automatically on first use. Safe to call multiple times.
        """
        if self.is_setup:
            return
        try:
            with Session(self._bind) as session:
                bind = session.get_bind()
                self._checkpoints_model.__table__.create(bind, checkfirst=True)
                self._writes_model.__table__.create(bind, checkfirst=True)
                session.commit()
            self.is_setup = True
        except ProgrammingError as e:
            logging.error(f"Create checkpoint tables failed: {e.__cause__}")
            raise Exception(e.__cause__) from None

    @contextmanager
    def _session(self, transaction: bool = True) -> Iterator[Session]:
        """Context manager that hands out a guarded SQLAlchemy ``Session``.

        The lock mirrors the per-instance safety semantics of
        ``SqliteSaver`` — useful when the same saver instance is shared
        across worker threads.
        """
        with self.lock:
            self.setup()
            session = Session(self._bind)
            try:
                yield session
                if transaction:
                    session.commit()
            finally:
                session.close()

    # ------------------------------------------------------------------
    # BaseCheckpointSaver implementation
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Return the ``CheckpointTuple`` matching ``config``.

        If ``config`` carries a ``checkpoint_id``, the matching checkpoint
        is returned. Otherwise, the most recent checkpoint for the
        ``(thread_id, checkpoint_ns)`` pair is returned. Returns ``None``
        if no checkpoint matches.
        """
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        explicit_checkpoint_id = get_checkpoint_id(config)

        with self._session(transaction=False) as session:
            if explicit_checkpoint_id:
                row = session.execute(
                    text(
                        f"SELECT thread_id, checkpoint_id, parent_checkpoint_id, "
                        f"type, checkpoint, metadata FROM {self._cps_t} "
                        "WHERE thread_id = :thread_id AND checkpoint_ns = :ns "
                        "AND checkpoint_id = :cid"
                    ),
                    {
                        "thread_id": thread_id,
                        "ns": checkpoint_ns,
                        "cid": explicit_checkpoint_id,
                    },
                ).fetchone()
            else:
                row = session.execute(
                    text(
                        f"SELECT TOP 1 thread_id, checkpoint_id, parent_checkpoint_id, "
                        f"type, checkpoint, metadata FROM {self._cps_t} "
                        "WHERE thread_id = :thread_id AND checkpoint_ns = :ns "
                        "ORDER BY checkpoint_id DESC"
                    ),
                    {"thread_id": thread_id, "ns": checkpoint_ns},
                ).fetchone()
            if row is None:
                return None

            (
                thread_id_row,
                checkpoint_id_row,
                parent_checkpoint_id,
                type_,
                checkpoint_blob,
                metadata_blob,
            ) = row

            resolved_config: RunnableConfig
            if explicit_checkpoint_id:
                resolved_config = config
            else:
                resolved_config = {
                    "configurable": {
                        "thread_id": thread_id_row,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id_row,
                    }
                }

            writes_rows = session.execute(
                text(
                    f"SELECT task_id, channel, type, blob FROM {self._wrt_t} "
                    "WHERE thread_id = :thread_id AND checkpoint_ns = :ns "
                    "AND checkpoint_id = :cid ORDER BY task_id, idx"
                ),
                {
                    "thread_id": thread_id_row,
                    "ns": checkpoint_ns,
                    "cid": checkpoint_id_row,
                },
            ).fetchall()

        return CheckpointTuple(
            resolved_config,
            self.serde.loads_typed((type_, bytes(checkpoint_blob))),
            cast(
                CheckpointMetadata,
                json.loads(metadata_blob) if metadata_blob is not None else {},
            ),
            (
                {
                    "configurable": {
                        "thread_id": thread_id_row,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            [
                (task_id, channel, self.serde.loads_typed((wtype, bytes(blob))))
                for task_id, channel, wtype, blob in writes_rows
            ],
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield checkpoint tuples matching ``config``, newest first.

        Args:
            config: Base filter; ``thread_id`` and ``checkpoint_ns`` (if
                present) narrow the result set.
            filter: Optional metadata equality filter. Each key/value pair
                is matched against the JSON-encoded ``metadata`` column.
            before: Optional config providing a ``checkpoint_id``; only
                checkpoints with a strictly lower id are yielded.
            limit: Optional cap on the number of rows yielded.
        """
        where_clauses: List[str] = []
        params: dict[str, Any] = {}
        if config is not None:
            configurable = config.get("configurable", {})
            if "thread_id" in configurable:
                where_clauses.append("thread_id = :thread_id")
                params["thread_id"] = str(configurable["thread_id"])
            if "checkpoint_ns" in configurable:
                where_clauses.append("checkpoint_ns = :ns")
                params["ns"] = configurable["checkpoint_ns"]
        if before is not None:
            before_id = before.get("configurable", {}).get("checkpoint_id")
            if before_id:
                where_clauses.append("checkpoint_id < :before_id")
                params["before_id"] = before_id
        if filter:
            for idx, (key, value) in enumerate(filter.items()):
                if not key.isidentifier():
                    raise ValueError(f"Invalid metadata filter key: {key!r}")
                bind_name = f"meta_{idx}"
                where_clauses.append(f"JSON_VALUE(metadata, '$.{key}') = :{bind_name}")
                params[bind_name] = str(value)
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        # Only emit a TOP clause for a positive limit; TOP with a negative
        # value is invalid T-SQL.
        limit_clause = ""
        if limit is not None and int(limit) > 0:
            limit_clause = f"TOP {int(limit)} "
        query = (
            f"SELECT {limit_clause}thread_id, checkpoint_ns, checkpoint_id, "
            f"parent_checkpoint_id, type, checkpoint, metadata "
            f"FROM {self._cps_t} "
            f"{where} ORDER BY checkpoint_id DESC"
        )

        with self._session(transaction=False) as session:
            rows = session.execute(text(query), params).fetchall()
            for (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                type_,
                checkpoint_blob,
                metadata_blob,
            ) in rows:
                writes_rows = session.execute(
                    text(
                        f"SELECT task_id, channel, type, blob FROM {self._wrt_t} "
                        "WHERE thread_id = :thread_id AND checkpoint_ns = :ns "
                        "AND checkpoint_id = :cid ORDER BY task_id, idx"
                    ),
                    {
                        "thread_id": thread_id,
                        "ns": checkpoint_ns,
                        "cid": checkpoint_id,
                    },
                ).fetchall()
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    self.serde.loads_typed((type_, bytes(checkpoint_blob))),
                    cast(
                        CheckpointMetadata,
                        json.loads(metadata_blob) if metadata_blob is not None else {},
                    ),
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        }
                        if parent_checkpoint_id
                        else None
                    ),
                    [
                        (task_id, channel, self.serde.loads_typed((wtype, bytes(blob))))
                        for task_id, channel, wtype, blob in writes_rows
                    ],
                )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist ``checkpoint`` and return an updated config.

        Performs an upsert (``MERGE``) keyed on
        ``(thread_id, checkpoint_ns, checkpoint_id)``.
        """
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        )
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        merge_sql = text(
            f"MERGE INTO {self._cps_t} AS target "
            "USING (SELECT :thread_id AS thread_id, :ns AS checkpoint_ns, "
            ":cid AS checkpoint_id) AS src "
            "ON target.thread_id = src.thread_id "
            "AND target.checkpoint_ns = src.checkpoint_ns "
            "AND target.checkpoint_id = src.checkpoint_id "
            "WHEN MATCHED THEN UPDATE SET "
            "parent_checkpoint_id = :parent_cid, type = :type, "
            "checkpoint = :checkpoint, metadata = :metadata "
            "WHEN NOT MATCHED THEN INSERT "
            "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
            "type, checkpoint, metadata) "
            "VALUES (:thread_id, :ns, :cid, :parent_cid, :type, "
            ":checkpoint, :metadata);"
        )

        with self._session() as session:
            session.execute(
                merge_sql,
                {
                    "thread_id": thread_id,
                    "ns": checkpoint_ns,
                    "cid": checkpoint["id"],
                    "parent_cid": parent_checkpoint_id,
                    "type": type_,
                    "checkpoint": serialized_checkpoint,
                    "metadata": serialized_metadata,
                },
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist intermediate writes for a task on a checkpoint.

        For channels listed in ``WRITES_IDX_MAP`` (a fixed index per
        channel name), an upsert is performed so reruns overwrite the
        previous write. Otherwise the row is left untouched if it already
        exists, matching the ``INSERT OR IGNORE`` semantics of the
        first-party savers.
        """
        if not writes:
            return

        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = str(config["configurable"]["checkpoint_id"])
        replace = all(channel in WRITES_IDX_MAP for channel, _ in writes)

        with self._session() as session:
            for idx, (channel, value) in enumerate(writes):
                wtype, serialized = self.serde.dumps_typed(value)
                effective_idx = WRITES_IDX_MAP.get(channel, idx)
                params = {
                    "thread_id": thread_id,
                    "ns": checkpoint_ns,
                    "cid": checkpoint_id,
                    "task_id": task_id,
                    "idx": effective_idx,
                    "channel": channel,
                    "type": wtype,
                    "blob": serialized,
                }
                if replace:
                    session.execute(
                        text(
                            f"MERGE INTO {self._wrt_t} AS target "
                            "USING (SELECT :thread_id AS thread_id, :ns AS "
                            "checkpoint_ns, :cid AS checkpoint_id, "
                            ":task_id AS task_id, :idx AS idx) AS src "
                            "ON target.thread_id = src.thread_id "
                            "AND target.checkpoint_ns = src.checkpoint_ns "
                            "AND target.checkpoint_id = src.checkpoint_id "
                            "AND target.task_id = src.task_id "
                            "AND target.idx = src.idx "
                            "WHEN MATCHED THEN UPDATE SET "
                            "channel = :channel, type = :type, blob = :blob "
                            "WHEN NOT MATCHED THEN INSERT "
                            "(thread_id, checkpoint_ns, checkpoint_id, task_id, "
                            "idx, channel, type, blob) "
                            "VALUES (:thread_id, :ns, :cid, :task_id, :idx, "
                            ":channel, :type, :blob);"
                        ),
                        params,
                    )
                else:
                    session.execute(
                        text(
                            f"IF NOT EXISTS (SELECT 1 FROM {self._wrt_t} "
                            "WHERE thread_id = :thread_id AND checkpoint_ns = :ns "
                            "AND checkpoint_id = :cid AND task_id = :task_id "
                            "AND idx = :idx) "
                            f"INSERT INTO {self._wrt_t} "
                            "(thread_id, checkpoint_ns, checkpoint_id, task_id, "
                            "idx, channel, type, blob) VALUES "
                            "(:thread_id, :ns, :cid, :task_id, :idx, :channel, "
                            ":type, :blob);"
                        ),
                        params,
                    )

    def delete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint and write row belonging to ``thread_id``."""
        with self._session() as session:
            session.execute(
                text(f"DELETE FROM {self._cps_t} WHERE thread_id = :thread_id"),
                {"thread_id": str(thread_id)},
            )
            session.execute(
                text(f"DELETE FROM {self._wrt_t} WHERE thread_id = :thread_id"),
                {"thread_id": str(thread_id)},
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _qualified(table: str, schema: Optional[str]) -> str:
        table_escaped = table.replace("]", "]]")
        if schema:
            schema_escaped = schema.replace("]", "]]")
            return f"[{schema_escaped}].[{table_escaped}]"
        return f"[{table_escaped}]"

    # ------------------------------------------------------------------
    # Connection helpers (mirrors SQLServer_VectorStore conventions).
    # ------------------------------------------------------------------

    def _get_connection_url(self, conn_string: str) -> str:
        if conn_string is None or len(conn_string) == 0:
            raise ValueError("Connection string value cannot be None.")
        if conn_string.startswith("mssql+pyodbc"):
            return conn_string

        try:
            args = conn_string.split(";")
            arg_dict = {}
            for arg in args:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    arg_dict[key.lower().strip()] = value.strip()
            database = arg_dict.pop("database")
            server = arg_dict.pop("server").split(",", 1)
            server_host = server[0]
            server_port: Optional[int] = None
            if ":" in server_host:
                server_host = server_host.split(":", 1)[1]
            if len(server) > 1 and server[1].isdigit():
                server_port = int(server[1])
            username = arg_dict.pop("uid", None)
            password = arg_dict.pop("pwd", None)
            if "driver" in arg_dict.keys():
                driver = re.search(r"\{([^}]*)\}", arg_dict["driver"])
                if driver is not None:
                    arg_dict["driver"] = driver.group(1)
            url = URL.create(
                "mssql+pyodbc",
                username=username,
                password=password,
                database=database,
                host=server_host,
                port=server_port,
                query=arg_dict,
            )
        except KeyError as k:
            raise Exception(
                f"Server, DB details should be provided in connection string. {k}"
            )
        return url.render_as_string(hide_password=False)

    def _can_connect_with_entra_id(self) -> bool:
        parsed_url = urlparse(self.connection_string)
        if parsed_url is None:
            return False
        invalid_keywords = [
            "trusted_connection=yes",
            "trustedconnection=yes",
            "authentication",
            "integrated security",
        ]
        if (
            parsed_url.username
            or parsed_url.password
            or any(keyword in parsed_url.query.lower() for keyword in invalid_keywords)
        ):
            return False
        return True

    def _create_engine(self) -> Engine:
        if not self.connection_string:
            raise ValueError(
                "Either `connection` or `connection_string` must be provided."
            )
        engine = create_engine(url=self.connection_string)
        if self._can_connect_with_entra_id():
            event.listen(engine, "do_connect", self._provide_token)
            logging.info("Using Entra ID Authentication.")
        return engine

    def _provide_token(
        self,
        dialect: Dialect,
        conn_rec: Optional[ConnectionPoolEntry],
        cargs: List[str],
        cparams: MutableMapping[str, Any],
    ) -> None:
        credential = DefaultAzureCredential()
        cargs[0] = cargs[0].replace(EXTRA_PARAMS, str())
        token_bytes = credential.get_token(AZURE_TOKEN_URL).token.encode("utf-16-le")
        token_struct = struct.pack(
            f"<I{len(token_bytes)}s", len(token_bytes), token_bytes
        )
        cparams["attrs_before"] = {SQL_COPT_SS_ACCESS_TOKEN: token_struct}
