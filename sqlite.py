import sqlite3
import json

# === Load and parse the JSON file ===
with open("getvehicles.json", "r") as file:
    data = json.load(file)

vehicles = data["bustime-response"]["vehicle"]

# === Connect to SQLite and create table ===
connection = sqlite3.connect("vehicles.db")
cursor = connection.cursor()

# === Create the VEHICLE table ===
cursor.execute("DROP TABLE IF EXISTS VEHICLE")  # to avoid re-creating in reruns
create_table_query = """
CREATE TABLE VEHICLE(
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
"""
cursor.execute(create_table_query)

# === Insert data into VEHICLE ===
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
        int(v.get("tripid", 0)) if "tripid" in v and v["tripid"] != "N/A" else None,
        int(v.get("blk", 0)) if "blk" in v and v["blk"] != "N/A" else None
    ))

# === Display inserted records ===
print("Inserted vehicle records:")
for row in cursor.execute("SELECT * FROM VEHICLE"):  # Display first 5 for brevity
    print(row)

# === Commit and close ===
connection.commit()
connection.close()
