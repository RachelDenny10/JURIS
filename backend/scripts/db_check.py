"""
Database connection check. Prints which database JURIS is using (password
hidden), the server version, the tables present, and the document count.
It does not create or change anything.

Run from the backend folder:
    python scripts\\db_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, inspect, select, text  # noqa: E402

from app.core.database import get_engine, get_session  # noqa: E402
from app.models.document import DocumentRecord  # noqa: E402


def main() -> None:
    try:
        engine = get_engine()
        print("Database URL :", engine.url.render_as_string(hide_password=True))

        with engine.connect() as conn:
            if engine.dialect.name == "sqlite":
                version = conn.execute(text("select sqlite_version()")).scalar()
                print("Backend      : SQLite", version)
            else:
                print("Backend      :", conn.execute(text("select version()")).scalar())

        tables = inspect(engine).get_table_names()
        print("Tables       :", tables or "(none yet)")

        if "documents" in tables:
            with get_session() as session:
                count = session.scalar(select(func.count()).select_from(DocumentRecord))
            print("Documents    :", count)
        else:
            print("Documents    : table not created yet (it is created when the server starts)")
    except Exception as exc:
        print("\nCould not connect to the database.")
        print("Reason:", exc)
        print("Check DATABASE_URL in backend\\.env and that the PostgreSQL service is running.")
        sys.exit(1)


if __name__ == "__main__":
    main()