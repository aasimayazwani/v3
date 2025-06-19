import sqlite3
from pathlib import Path
import pandas as pd

# === File Paths ===
db_path = Path("vehicles.db")

# === Remove existing DB if it exists ===
if db_path.exists():
    print("🗑️ Deleting existing vehicles.db")
    db_path.unlink()

# === CSV file to table mapping ===
csv_files = {
    "getvehicles": "getvehicles.csv",
    "gtfs_block": "gtfs_block.csv",
    "gtfs_trip": "gtfs_trip.csv",
    "gtfs_shape": "gtfs_shape.csv",
    "gtfs_calendar_dates": "gtfs_calendar_dates.csv",
    "trip_event_bustime": "trip_event_bustime.csv",
    "trip_event_bustime_to_block": "trip_event_bustime_to_block.csv",
    "clever_pred": "clever_pred.csv",
    "bus_vid":"bus_vid.csv"
}

# === Create a new SQLite connection ===
connection = sqlite3.connect(db_path)

try:
    for table_name, file_name in csv_files.items():
        file_path = Path(file_name)

        if not file_path.exists():
            print(f"⚠️  Skipping missing file: {file_name}")
            continue

        print(f"📦 Importing {file_name} → `{table_name}`")
        df = pd.read_csv(file_path, dtype=str, na_values="", keep_default_na=False)

        # Try converting to numeric types when safe
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore", downcast="float")

        df.to_sql(table_name, connection, if_exists="replace", index=False)

    connection.commit()
    print("✅ All tables loaded successfully.")
finally:
    connection.close()
    print("🧠 Database connection closed.")

print("📁 vehicles.db is ready to use.")
