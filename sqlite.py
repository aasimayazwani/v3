import sqlite3
from pathlib import Path
import pandas as pd

# === Connect to SQLite DB ===
db_path = Path("vehicles.db")
connection = sqlite3.connect(db_path)

# === List of CSV files to import ===
csv_files = {
    "vehicle": "VEHICLE.csv",
    "getvehicles": "getvehicles.csv",
    "gtfs_block": "gtfs_block.csv",
    "gtfs_trip": "gtfs_trip.csv",
    "gtfs_shape": "gtfs_shape.csv",
    "gtfs_calendar_dates": "gtfs_calendar_dates.csv",
    "historical_trips": "historical_trips.csv",
    "trip_event_bustime_to_block": "trip_event_bustime_to_block.csv",
    "clever_pred": "clever_pred.csv"
}

# === Load and insert each CSV file ===
for table_name, file_name in csv_files.items():
    path = Path(file_name)
    if not path.exists():
        print(f"⚠️ File not found: {file_name}")
        continue

    print(f"📦 Loading {file_name} → table `{table_name}`")
    df = pd.read_csv(path, dtype=str, na_values="", keep_default_na=False)

    # Try to infer numeric types
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore", downcast="float")
        except Exception:
            pass

    # Insert into DB (automatically replaces any existing table)
    df.to_sql(table_name, connection, if_exists="replace", index=False)

print("✅ All CSV tables imported successfully.")
connection.commit()
connection.close()
print("🧠 vehicles.db is ready to use.")
