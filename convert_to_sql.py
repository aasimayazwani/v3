#!/usr/bin/env python
"""
ingest_gtfs_to_sqlite.py  ────────────────────────────────────────────────
Load every .txt file inside a GTFS zip into SQLite, one table per file.

Usage:  python ingest_gtfs_to_sqlite.py <gtfs.zip> [sqlite_db]

If [sqlite_db] is omitted, the script creates/uses 'vehicles.db'
in the same directory as the zip.
"""
import sys
import sqlite3
import zipfile
import io
from pathlib import Path
import pandas as pd


def sql_dtype(pd_dtype):
    """Map pandas dtype → SQLite column type."""
    if pd_dtype.kind in ("i", "u"):       # int / uint
        return "INTEGER"
    if pd_dtype.kind == "f":              # float
        return "REAL"
    return "TEXT"                         # default


def load_gtfs_zip(zip_path: Path, db_path: Path):
    print(f"→ Opening GTFS archive: {zip_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    with zipfile.ZipFile(zip_path, "r") as zf:
        txt_files = [m for m in zf.namelist() if m.endswith(".txt")]
        if not txt_files:
            raise RuntimeError("No .txt files found in the zip!")

        for txt in txt_files:
            table = Path(txt).stem.lower()  # e.g. 'agency.txt' → 'agency'
            print(f"  • Loading {txt}  →  table `{table}`")

            # Read into DataFrame
            with zf.open(txt) as fp:
                df = pd.read_csv(
                    io.TextIOWrapper(fp, encoding="utf-8-sig"),
                    dtype=str,              # keep everything as str first
                    na_values="",           # treat empty strings as NaN
                    keep_default_na=False,
                )

            # Infer dtypes (simple heuristic)
            for col in df.columns:
                # Try int
                try:
                    df[col] = pd.to_numeric(df[col], downcast="integer")
                except ValueError:
                    pass
                # Try float
                try:
                    df[col] = pd.to_numeric(df[col], downcast="float")
                except ValueError:
                    pass

            # Build CREATE TABLE
            columns_sql = ",\n  ".join(
                f'"{c}" {sql_dtype(df[c].dtype)}' for c in df.columns
            )
            ddl = f'CREATE TABLE IF NOT EXISTS "{table}" (\n  {columns_sql}\n);'
            cur.execute(f'DROP TABLE IF EXISTS "{table}"')
            cur.execute(ddl)

            # Insert rows
            df.to_sql(table, conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    print(f"✓ All GTFS tables imported into {db_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python ingest_gtfs_to_sqlite.py <gtfs.zip> [sqlite_db]")
    zip_file = Path(sys.argv[1]).expanduser().resolve()
    db_file = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) > 2
        else zip_file.parent / "vehicles.db"
    )
    load_gtfs_zip(zip_file, db_file)
