###############################################################################
# app.py – streamlined, hardened version for vehicles.db                      #
# (ChatGPT API + developer‑defined system instructions)                       #
###############################################################################
import unicodedata
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine
import sqlite3
import pandas as pd

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI

# Attempt to pull LangChain's default SQL prompt so we can append our own.
try:
    from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX as _LC_SQL_PREFIX
except Exception:  # Fallback if import path changes
    _LC_SQL_PREFIX = ""

###############################################################################
# ---------- Developer system instructions -----------------------------------
###############################################################################
SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS = """
You are a data-savvy transit-operations assistant.
• Answer in concise, plain English.
• When creating new tables, name them in lower_case_snake_case.
• Never modify or drop baseline tables that existed at the start of the chat.
• Always wrap executed SQL in triple-backtick blocks marked as sql.

========================
1. OVERVIEW
========================
The database merges three data domains:
1. Static GTFS planning data (blocks, trips, shapes, calendar dates).
2. Real-time AVL and battery predictions from Clever Devices.
3. Historical performance aggregates at trip and block granularity.

Every table is already created.  Your job is to read, join, and filter –
never alter schema or contents unless explicitly asked.

========================
2. STATIC GTFS TABLES
========================
gtfs_block  – static sequence of trips a vehicle performs in a service day  
  • Primary key: (block_id_gtfs, day, service_id)  
  • Important fields  
    block_id_gtfs / block_id_user      : identifiers (varchar 15)  
    day                                : weekday in upper case  
    service_id                         : link to gtfs_calendar_dates  
    route_id, route_id_2, route_id_3   : routes within the block  
    start_time / end_time              : yard-to-yard clock times HH:MM  
    inservice_start_time / _end_time   : revenue portion only  
    st / et / inservice_st / _et       : seconds past midnight versions  
    revenue_time / _length             : minutes and miles in service  
    deadhead_time / _length            : minutes and miles deadhead  
    break_time                         : layover minutes  
  • Refreshed quarterly.

gtfs_calendar_dates – maps service_id to specific dates  
  • Primary key: (date) YYYY-MM-DD  
  • Fields: date, service_id, day (weekday text).  
  • Use to decide which blocks/trips run on any given date.  
  • Refreshed quarterly.

gtfs_shape – ordered GPS points for each shape_id  
  • Primary key: (shape_id, sequence)  
  • route_id, route_index, latitude, longitude, distance (m).  
  • Used to draw paths and measure shape length.  
  • Refreshed quarterly.

gtfs_trip – every scheduled trip  
  • Primary key: (trip_id, block_id_gtfs, day, trip_index, service_id)  
  • Links to gtfs_block by block_id and service_id.  
  • Fields include start_time, end_time, shape_id, route_type, geographic
    coordinates, distance, elevation stats, trip_type (STANDARD | DEADHEAD | LAYOVER).  
  • Refreshed quarterly.

========================
3. REAL-TIME TABLES
========================
getvehicles – live AVL snapshot (last 5 min, sorted by timestamp desc)  
  • Primary key: (timestamp, vid)  
  • Core fields: lat, lon, hdg, spd, dly (delay flag), psgld (load),
    tablockid (user block), blk (gtfs block), tatripid (user trip),
    tripid (gtfs trip).  
  • Mode 1 == bus, 2 == ferry, 3 == rail, 4 == people_mover.  
  • Updates every minute.

clever_pred – live battery / range prediction for EVs 2401-2407, 707, 736, 768, 775, 777  
  • Primary key: (timestamp, vid)  
  • Fields: current_soc, pred_end_soc, pred_end_soc_trip, pred_end_soc_test,
    left_miles, pred_end_miles, avg_kwh_mile, energy_used, avg_speed,
    max_speed, current_weight, current_temp.  
  • Use to flag low (<40 %) or critical (<10 %) SOC.  
  • Updates every minute.

========================
4. HISTORICAL TABLES
========================
trip_event_bustime – observed trip stats for EVs  
  • Primary key: (vid, tatripid, start_timestamp)  
  • Metrics: time_driven, miles_driven, start_soc / end_soc, kwh_mile,
    elevation gain/loss, avg_speed, max_speed, acceleration stats,
    passenger load, traffic factor.  
  • Updates daily.

trip_event_bustime_to_block – observed block stats for EVs  
  • Primary key: (vid, tablockid, start_timestamp)  
  • Metrics: num_trip, time_driven, miles_driven, soc_used, kwh_mile,
    passenger load, traffic, elevation stats, driver count.  
  • Updates daily.

========================
5. BUSINESS RULES
========================
Electric_vehicle_ids = [2401, 2402, 2403, 2404, 2405, 2406, 2407,
                        707, 736, 768, 775, 777]

Vehicle_inservice_status  
  inservice       : getvehicles.tablockid is not null  
  not_in_service  : tablockid is null

Battery_alerts  
  critical  : soc < 10 → return to depot immediately  
  low       : soc < 40 → attention needed  
  normal    : soc ≥ 40 → continue operation

========================
6. HOW TO ANSWER USERS
========================
Step 1 – parse intent: is the rider asking about a route, stop, vehicle, date,
          SOC level, or performance metric?  
Step 2 – choose the base table(s):  
  • service or schedule questions → gtfs_block, gtfs_trip, gtfs_calendar_dates  
  • location of a bus now → getvehicles (join to static tables for context)  
  • battery / range warnings → clever_pred (apply Battery_alert thresholds)  
  • historical efficiency → trip_event_bustime or _to_block  
Step 3 – apply joins / filters:  
  • Resolve service_id for a date with gtfs_calendar_dates.  
  • Join static id keys exactly: block_id, trip_id, shape_id, etc.  
Step 4 – format the answer in plain English; if SQL is required, show the
          query in a sql block and give a short English summary.

Cardinality reminders:
one block → many trips  
one trip  → many shape points  
one vehicle id  → many AVL rows over time  
static tables never change inside a chat; real-time tables mutate continuously.

Follow these instructions exactly to produce reliable, auditable answers.
"""
###############################################################################
# ---------- Utility helpers --------------------------------------------------
###############################################################################
DB_FILE = Path(__file__).parent / "vehicles.db"


def ascii_sanitise(value: str) -> str:
    """Return a strictly‑ASCII version of `value`."""
    return (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )

###############################################################################
# ---------- Streamlit sidebar (API key only) ---------------------------------
###############################################################################
st.set_page_config(page_title="LangChain • Vehicles DB", page_icon="🚌")
st.title("🚌 Chat with Vehicles Database")

api_key_raw = st.sidebar.text_input("OpenAI API Key", type="password")
api_key = ascii_sanitise(api_key_raw or "")

if not api_key:
    st.info("Please enter your OpenAI API key ↑ to begin.", icon="🔐")
    st.stop()

###############################################################################
# ---------- Configure DB connection (cached, auto‑invalidated) --------------
###############################################################################
@st.cache_resource(ttl=0)
def get_db_connection(db_path: Path, api_key_ascii: str):
    """Return (SQLDatabase, ChatOpenAI LLM) tuple."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    st.session_state["_db_mtime"] = db_path.stat().st_mtime  # refresh on change

    creator = lambda: sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql_db = SQLDatabase(create_engine("sqlite:///", creator=creator))

    llm = ChatOpenAI(
        openai_api_key=api_key_ascii,
        model_name="gpt-4o-mini",
        streaming=True,
    )
    return sql_db, llm


db, llm = get_db_connection(DB_FILE, api_key)

###############################################################################
# ---------- Capture baseline tables -----------------------------------------
###############################################################################
if "base_tables" not in st.session_state:
    with sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True) as _conn:
        st.session_state["base_tables"] = {
            row[0] for row in _conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
        }

###############################################################################
# ---------- LangChain agent with custom prompt ------------------------------
###############################################################################

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

custom_prefix = SYSTEM_INSTRUCTIONS.strip() + "\n\n" + _LC_SQL_PREFIX

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prefix=custom_prefix,
    handle_parsing_errors=True,
)

###############################################################################
# ---------- Table Download UI (new tables only) -----------------------------
###############################################################################
st.sidebar.markdown("### 📥 Download *new* Table as CSV")

with sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True) as _conn:
    current_tables = {
        row[0] for row in _conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
    }

new_tables = sorted(current_tables - st.session_state["base_tables"])

if new_tables:
    selected_new_table = st.sidebar.selectbox("Select a new table", new_tables)

    if selected_new_table:
        new_df = pd.read_sql_query(
            f"SELECT * FROM {selected_new_table};",
            sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True),
        )
        csv_bytes = new_df.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            label=f"Download `{selected_new_table}.csv`",
            data=csv_bytes,
            file_name=f"{selected_new_table}.csv",
            mime="text/csv",
        )
else:
    st.sidebar.info("No new tables have been created in this chat session yet.")

###############################################################################
# ---------- Chat UI & session history ---------------------------------------
###############################################################################
if (
    "messages" not in st.session_state
    or st.sidebar.button("🗑️ Clear chat history", help="Start a fresh session")
):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi there – ask me anything about the **VEHICLE** table, or "
                "create new tables and download them from the sidebar!"
            ),
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input("Ask a question…")

if user_query:
    st.chat_message("user").write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query, callbacks=[cb])
        except UnicodeEncodeError:
            response = (
                "⚠️ I encountered a Unicode encoding issue while talking to the LLM. "
                "Please try rephrasing your question using plain ASCII characters."
            )
        except Exception as e:
            response = f"⚠️ Something went wrong:\n\n`{e}`"

        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})