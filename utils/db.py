"""
DuckDB wrapper — schema creation, insert, and query helpers.
"""

import duckdb
import pandas as pd
from pathlib import Path
from utils.config import DB_PATH


def get_connection(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist."""
    con = get_connection(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id       VARCHAR PRIMARY KEY,
            date            DATE,
            site            VARCHAR,
            department      VARCHAR,
            report_type     VARCHAR,
            narrative       TEXT,
            sif_potential   BOOLEAN,
            sif_score       DOUBLE,
            confidence      VARCHAR,
            life_saving_rules VARCHAR,
            top_signals     VARCHAR,
            precursor_pattern VARCHAR,
            processed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.close()


def upsert_reports(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    """Insert or replace reports from a DataFrame."""
    con = get_connection(db_path)
    con.execute("CREATE TABLE IF NOT EXISTS _tmp AS SELECT * FROM reports WHERE 1=0")
    con.register("df_view", df)
    con.execute("""
        INSERT OR REPLACE INTO reports
        SELECT
            report_id, date::DATE, site, department, report_type,
            narrative, sif_potential, sif_score, confidence,
            life_saving_rules, top_signals, precursor_pattern,
            CURRENT_TIMESTAMP
        FROM df_view
    """)
    con.close()


def query_reports(
    db_path: str = DB_PATH,
    site: str = None,
    sif_only: bool = False,
    lsr_filter: str = None,
    date_from: str = None,
    date_to: str = None,
) -> pd.DataFrame:
    """Flexible query with optional filters."""
    con = get_connection(db_path)
    clauses = ["1=1"]
    if site:
        clauses.append(f"site = '{site}'")
    if sif_only:
        clauses.append("sif_potential = TRUE")
    if lsr_filter:
        clauses.append(f"life_saving_rules LIKE '%{lsr_filter}%'")
    if date_from:
        clauses.append(f"date >= '{date_from}'")
    if date_to:
        clauses.append(f"date <= '{date_to}'")
    where = " AND ".join(clauses)
    df = con.execute(f"SELECT * FROM reports WHERE {where}").df()
    con.close()
    return df


def site_risk_summary(db_path: str = DB_PATH) -> pd.DataFrame:
    """Returns per-site SIF density summary."""
    con = get_connection(db_path)
    df = con.execute("""
        SELECT
            site,
            COUNT(*) AS total_reports,
            SUM(CASE WHEN sif_potential THEN 1 ELSE 0 END) AS sif_count,
            ROUND(AVG(sif_score), 3) AS avg_sif_score,
            ROUND(100.0 * SUM(CASE WHEN sif_potential THEN 1 ELSE 0 END) / COUNT(*), 1) AS sif_pct
        FROM reports
        GROUP BY site
        ORDER BY sif_pct DESC
    """).df()
    con.close()
    return df


def lsr_summary(db_path: str = DB_PATH) -> pd.DataFrame:
    """Returns count of SIF reports per Life-Saving Rule."""
    con = get_connection(db_path)
    df = con.execute("""
        SELECT life_saving_rules, COUNT(*) AS count
        FROM reports
        WHERE sif_potential = TRUE
        GROUP BY life_saving_rules
        ORDER BY count DESC
    """).df()
    con.close()
    return df
