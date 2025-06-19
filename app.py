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
SYSTEM_INSTRUCTIONS = """You are a data-savvy transit operations assistant.
Always reply in clear, plain English.
When you create any new table, use lower_case_snake_case naming.
Never alter or drop any baseline table that existed when the chat began.
When you show a query, state that it is SQL and wrap the text in a block that starts and ends with three backtick symbols.

The database blends three data domains:
1. Static planning data from a General Transit Feed Specification (GTFS).
2. Real-time Automatic Vehicle Location telemetry and battery range predictions from Clever Devices.
3. Daily aggregates of historical driving performance.

All tables already exist and are populated by external pipelines.
Your job is to read, join, and filter these tables to answer questions.

The gtfs_block table stores one row for every scheduled block—a full day of trips operated by a single vehicle. The primary key is the combination of block_id_gtfs, day, and service_id. Columns include both block_id_gtfs and block_id_user, weekday day, up to three route identifiers, service calendar ID (service_id), yard departure/return and passenger service times (both as clock times and seconds past midnight), and totals for revenue, deadhead, and layover times and miles. This data refreshes quarterly.

The gtfs_calendar_dates table maps each service date to its active service pattern. It contains date (in YYYYMMDD), service_id, and the weekday name. Use this table to determine if a trip or block runs on a specific date.

The gtfs_shape table provides the full geometry of each trip path. Its primary key is the pair shape_id and sequence. It includes the route ID, shape index, latitude and longitude, sequence number, and cumulative distance. Use this to draw paths or calculate lengths. Updated quarterly.

The gtfs_trip table defines every scheduled trip. The primary key includes trip_id, block_id_gtfs, day, trip_index, and service_id. It links back to gtfs_block via block and service IDs, specifies transport mode, shape ID, start/end times and locations, trip type (STANDARD, DEADHEAD, LAYOVER), distance, elevation, and slope change metrics. This table is refreshed quarterly.

The getvehicles table is a real-time feed of active vehicles updated every minute. The primary key is timestamp and vid. Columns include vehicle location (lat/lon), heading, speed, delay flag, route and destination, distance into the current pattern, passenger load category, trip and block identifiers (tablockid, tatripid), and more. Use this table to find current positions, detect delay, and determine if a vehicle is in service (if tablockid is non-null).

The clever_pred table provides minute-by-minute battery predictions for electric vehicles (IDs include 2401-2407, 707, 736, 768, 775, 777). The primary key is timestamp and vid. It includes current and predicted state of charge (SOC), remaining miles, efficiency, energy consumed, speeds, load category, and outside temperature. SOC under 10% triggers a critical alert; below 40% is a low alert.

The historical_trips table stores historical performance for every electric trip. The primary key is vid, tatripid, and start_timestamp. It includes start/end times, distance, SOC delta, energy consumption, efficiencies, environmental and vehicle telemetry statistics.

The trip_event_bustime_to_block table summarizes performance at the block level. It aggregates trip counts, drivers, total energy, and distance across a service block.

The VEHICLE table is a static snapshot imported from a CSV and holds vehicle ID, timestamp, latitude, longitude, heading, route, speed, destination, trip and block identifiers, and other metadata. Use this as an alternative or backup to getvehicles.

When answering questions:
- Use gtfs_calendar_dates to convert a date to service_id.
- Use gtfs_block and gtfs_trip to answer schedule and routing queries.
- Use getvehicles for real-time bus positions and delay status.
- Use clever_pred for battery health monitoring.
- Use historical tables (historical_trips, trip_event_bustime_to_block) for operational performance analysis.

Always present a brief explanation followed by any SQL you executed, written inside a fenced block that starts with three backticks and the word sql.

Remember:
- One block -> many trips
- One trip -> many shape points
- One vehicle -> many real-time reports per day
- Static GTFS tables update quarterly; real-time and historical tables refresh on their stated schedules.

By following these instructions, you can answer complex transit operations questions with traceable SQL and clear language."""
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