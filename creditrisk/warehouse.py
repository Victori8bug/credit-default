"""A small SQL layer (DuckDB) over the raw data.

DuckDB is an in-process analytical database: it runs inside Python with no
server, and speaks standard SQL. The .sql files in /sql are the source of truth
for the queries; this module only runs them.
"""
from pathlib import Path

import duckdb
import pandas as pd

from .data import DATA_PATH

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def read_sql(name: str) -> str:
    return (SQL_DIR / name).read_text()


def connect(csv_path=DATA_PATH) -> duckdb.DuckDBPyConnection:
    """In-memory database with the accounts table and the feature view built."""
    con = duckdb.connect(":memory:")
    con.execute(read_sql("01_load_accounts.sql").replace("{csv_path}", str(csv_path)))
    con.execute(read_sql("02_account_features.sql"))
    return con


def query(con, name: str) -> pd.DataFrame:
    """Run one of the saved portfolio queries and return a DataFrame."""
    return con.execute(read_sql(name)).df()
