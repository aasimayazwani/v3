import sqlite3
import json
import zipfile
import io
import pandas as pd
from pathlib import Path

# === Load vehicle data from JSON ===
print("📦 Loading real-time vehicle data from getvehicles.json...")
with open("getvehicles.json", "r") as file:
    data = json.load(file)

vehicles = data["bustime-response"]["vehicle"]

# === Connect to SQLite DB ===
db_path = Path("vehicles.db")
connection = sqlite3.connect(db_path)
cursor = connection.cursor()

# === Create or replace VEHICLE table ===
cursor.execute("DROP TABLE IF EXISTS VEHICLE")
cursor.execute("""
CREATE TABLE VEHICLE (
    vid TEXT,
    tmstmp TEXT,
    lat REAL,
    lon REAL,
    hdg INTEGER,
    rt TEXT,
    des TEXT,
    spd INTEGER,
    tablockid TEXT,
    tripid INTEGER,
    blk INTEGER
)
""")

# === Insert vehicle rows ===
for v in vehicles:
    cursor.execute("""
        INSERT INTO VEHICLE (vid, tmstmp, lat, lon, hdg, rt, des, spd, tablockid, tripid, blk)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        v.get("vid"),
        v.get("tmstmp"),
        float(v.get("lat", 0)),
        float(v.get("lon", 0)),
        int(v.get("hdg", 0)),
        v.get("rt"),
        v.get("des"),
        int(v.get("spd", 0)),
        v.get("tablockid"),
        int(v["tripid"]) if v.get("tripid") not in (None, "N/A") else None,
        int(v["blk"]) if v.get("blk") not in (None, "N/A") else None
    ))

print("✅ VEHICLE table created and populated.")

# === Load GTFS data from ZIP ===
gtfs_zip_path = Path("gtfs_gtrans.zip")
if gtfs_zip_path.exists():
    print(f"📦 Importing GTFS data from {gtfs_zip_path}...")
    with zipfile.ZipFile(gtfs_zip_path, "r") as zf:
        for filename in zf.namelist():
            if not filename.endswith(".txt"):
                continue
            table_name = Path(filename).stem.lower()
            print(f"  • Loading {filename} → `{table_name}`")

            with zf.open(filename) as file:
                df = pd.read_csv(
                    io.TextIOWrapper(file, encoding="utf-8-sig"),
                    dtype=str,
                    na_values="",
                    keep_default_na=False
                )

            # Try to infer column types
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], downcast="integer")
                except ValueError:
                    pass
                try:
                    df[col] = pd.to_numeric(df[col], downcast="float")
                except ValueError:
                    pass

            # Create table
            col_defs = ", ".join(f'"{c}" TEXT' for c in df.columns)  # safe default
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            cursor.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

            # Insert rows
            df.to_sql(table_name, connection, if_exists="append", index=False)
    print("✅ All GTFS tables imported.")
else:
    print("⚠️ GTFS zip file not found. Skipping GTFS import.")

# === Done ===
connection.commit()
connection.close()
print("🧠 vehicles.db is ready to use.")
