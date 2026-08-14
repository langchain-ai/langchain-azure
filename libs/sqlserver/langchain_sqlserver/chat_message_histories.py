"""Chat message history backed by SQL Server / Azure SQL.

Provides :class:`SQLServerChatMessageHistory`, an implementation of
``langchain_core.chat_history.BaseChatMessageHistory`` that persists messages
to a SQL Server table.
"""

from __future__ import annotations

import json
import logging
import re
import struct
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Union
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from sqlalchemy import (
    BigInteger,
    Column,
    Dialect,
    Index,
    PrimaryKeyConstraint,
    create_engine,
    delete,
    event,
    insert,
    select,
)
from sqlalchemy.dialects.mssql import NVARCHAR
from sqlalchemy.engine import URL, Connection, Engine
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy.pool import ConnectionPoolEntry

AZURE_TOKEN_URL = "https://database.windows.net/.default"
EXTRA_PARAMS = ";Trusted_Connection=Yes"
SQL_COPT_SS_ACCESS_TOKEN = 1256

DEFAULT_TABLE_NAME = "sqlserver_chat_history"


class SQLServerChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a SQL Server table.

    Each instance is scoped to a single ``session_id``; multiple sessions can
    share the same underlying table.

    The table is created on first use if it does not already exist. Messages
    are stored serialized as JSON in an ``NVARCHAR(MAX)`` column and ordered
    by an auto-incrementing ``id`` column on read, so insertion order is
    preserved.

    The class supports both username/password connections and Entra ID
    authentication using the same connection-string conventions as
    :class:`langchain_sqlserver.SQLServer_VectorStore`.

    Example:
        ```python
        from langchain_sqlserver import SQLServerChatMessageHistory

        history = SQLServerChatMessageHistory(
            session_id="user-123",
            connection_string=(
                "Driver={ODBC Driver 18 for SQL Server};Server=tcp:host,1433;"
                "Database=mydb;Uid=user;Pwd=pwd;TrustServerCertificate=yes;"
            ),
        )
        history.add_user_message("hi!")
        history.add_ai_message("hello")
        print(history.messages)
        ```
    """

    def __init__(
        self,
        session_id: str,
        *,
        connection: Optional[Connection] = None,
        connection_string: str = "",
        table_name: str = DEFAULT_TABLE_NAME,
        db_schema: Optional[str] = None,
    ) -> None:
        """Initialize a chat message history.

        Args:
            session_id: Identifier for the chat session. All messages added
                through this instance are associated with this ``session_id``
                and are returned by ``messages``.
            connection: Optional pre-existing SQLAlchemy ``Connection``. When
                provided, ``connection_string`` is ignored for engine creation.
            connection_string: SQL Server ODBC connection string. Required if
                ``connection`` is not provided. If the string contains no
                username/password and no ``Trusted_Connection=yes``, Entra ID
                authentication is used.
            table_name: Name of the table that stores messages. Defaults to
                ``sqlserver_chat_history``.
            db_schema: Optional schema in which the table lives. The schema
                must already exist.
        """
        if not session_id:
            raise ValueError("session_id must be a non-empty string.")

        self.session_id = session_id
        self.table_name = table_name
        self.schema = db_schema
        self.connection_string = (
            self._get_connection_url(connection_string) if connection_string else ""
        )
        self._bind: Union[Connection, Engine] = (
            connection if connection else self._create_engine()
        )
        self._message_store = self._get_message_store(self.table_name, self.schema)
        self._create_table_if_not_exists()

    def _get_message_store(self, name: str, schema: Optional[str]) -> Any:
        DynamicBase = declarative_base(class_registry=dict())  # type: Any

        class MessageStore(DynamicBase):
            """Base model for chat message rows."""

            __tablename__ = name
            __table_args__ = (
                PrimaryKeyConstraint("id", mssql_clustered=True),
                Index(f"idx_{name}_session_id", "session_id"),
                {"schema": schema},
            )
            id = Column(BigInteger, primary_key=True, autoincrement=True)
            session_id = Column(NVARCHAR(255), nullable=False)
            message = Column(NVARCHAR, nullable=False)  # NVARCHAR(MAX)

        return MessageStore

    def _create_table_if_not_exists(self) -> None:
        logging.info(f"Creating chat history table {self.table_name}.")
        try:
            with Session(self._bind) as session:
                self._message_store.__table__.create(
                    session.get_bind(), checkfirst=True
                )
                session.commit()
        except ProgrammingError as e:
            logging.error(f"Create table {self.table_name} failed.")
            raise Exception(e.__cause__) from None

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        """Return all messages stored for this session, in insertion order."""
        try:
            with Session(self._bind) as session:
                rows = session.execute(
                    select(self._message_store.message)
                    .where(self._message_store.session_id == self.session_id)
                    .order_by(self._message_store.id.asc())
                ).fetchall()
        except DBAPIError as e:
            logging.error(f"Fetch messages failed:\n {e.__cause__}\n")
            raise Exception(e.__cause__) from None

        items: List[Dict[str, Any]] = [json.loads(row[0]) for row in rows]
        return messages_from_dict(items)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append a batch of messages to the store.

        Args:
            messages: Sequence of ``BaseMessage`` objects to persist. Each one
                is serialized via ``message_to_dict`` and stored as JSON.
        """
        if not messages:
            return

        rows = [
            {
                "session_id": self.session_id,
                "message": json.dumps(message_to_dict(message)),
            }
            for message in messages
        ]
        try:
            with Session(self._bind) as session:
                session.execute(insert(self._message_store).values(rows))
                session.commit()
        except DBAPIError as e:
            logging.error(f"Add messages failed:\n {e.__cause__}\n")
            raise Exception(e.__cause__) from None

    def clear(self) -> None:
        """Delete every message belonging to this session_id."""
        try:
            with Session(self._bind) as session:
                session.execute(
                    delete(self._message_store).where(
                        self._message_store.session_id == self.session_id
                    )
                )
                session.commit()
        except DBAPIError as e:
            logging.error(f"Clear messages failed:\n {e.__cause__}\n")
            raise Exception(e.__cause__) from None

    # ------------------------------------------------------------------
    # Connection helpers (mirrors the conventions used by
    # SQLServer_VectorStore so users can reuse the same connection strings).
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
