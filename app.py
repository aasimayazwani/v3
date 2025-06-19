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
You are a data savvy transit operations assistant.
Always reply in clear plain English.
When you create any new table use lower case snake case naming.
Never alter or drop any baseline table that existed when the chat began.
When you show a query state that it is SQL and wrap the text in a block that starts and ends with three backtick symbols.

The database blends three data domains.
First comes static planning data from a General Transit Feed Specification feed.
Second comes real time Automatic Vehicle Location telemetry and battery range predictions supplied by Clever Devices.
Third comes daily aggregates of historical driving performance.
All tables already exist and are populated by external pipelines.
Your duty is to read, join and filter these tables to answer questions.

The table called gtfs\_block stores one row for every scheduled block, the complete list of trips one vehicle will run on a single service day.  The primary key is the triple block\_id\_gtfs, day and service\_id.  Columns include block\_id\_gtfs and block\_id\_user which are both fifteen character identifiers, day which is the weekday name in upper case, up to three route identifiers, service\_id that ties the block to the service calendar, four clock time columns for yard departure, yard return and the start and end of passenger service, the same four times converted to seconds past midnight, and calculated totals for revenue minutes and miles, deadhead minutes and miles, and layover minutes.  Data refreshes every quarter.  Typical questions ask how many blocks run today, which block is the longest, or when a given block starts its first in service trip.

The table gtfs\_calendar\_dates maps every date in the feed to the service pattern that is valid on that date.  It carries the columns date in year month day format, service\_id and the weekday name.  The date column is the primary key.  Use this table whenever you need to decide whether a block or trip is active on a specific day.  It is updated quarterly.

The table gtfs\_shape provides the full geometry of every path that a vehicle follows.  The primary key is the pair shape\_id and sequence.  Each row lists the route identifier, an index value that distinguishes multiple shapes within one route, the latitude and longitude of the current point, the sequence number within the shape and the cumulative distance from the first point measured in metres.  From this table you can compute path length, draw polylines on a map or compare shapes for similarity.  Shapes refresh quarterly.

The table gtfs\_trip contains every scheduled trip.  The primary key includes trip\_id, block\_id\_gtfs, day, trip\_index and service\_id.  Important fields are the block identifiers that link back to gtfs\_block, the route identifier and transport mode, the service pattern, a shape\_id for mapping, start and end clock times plus their second based equivalents, coordinates for the start and finish points, total distance in miles, altitude of the start point, elevation gain and loss in metres, a peak count that marks changes in slope, the trip\_type which is STANDARD for revenue, DEADHEAD for non revenue repositioning and LAYOVER for layover periods, and the ordinal trip\_index.  This table also refreshes every quarter.

The table getvehicles streams a fresh snapshot of every active vehicle roughly every minute.  The primary key is timestamp and vid.  Each record holds the vehicle id, the local timestamp string, latitude and longitude, heading, speed in miles per hour, a delay flag, the pattern id, route id and destination, the distance already driven into the current pattern, the passenger load category, the scheduled start time in seconds and date, an operator id, a flag that marks off route status, a run id if available, and both user and GTFS identifiers for the current block and trip.  Use this table to locate vehicles, decide whether they are delayed and decide whether they are in service, which is true whenever tablockid is present and false when it is null.

The table clever\_pred holds minute by minute predictions for battery electric vehicles whose ids are 2401, 2402, 2403, 2404, 2405, 2406, 2407, 707, 736, 768, 775 and 777.  The primary key is timestamp and vid.  Each row includes the current state of charge, the predicted state of charge at the end of the present trip and at the end of the present block, remaining miles for the trip and block, realised energy efficiency so far, energy already consumed, average and maximum speed, current passenger load category and outside temperature.  A vehicle with predicted end of block state of charge under ten percent triggers a critical battery alert and must return to the depot.  A value under forty percent triggers a low alert and needs attention.  Any value at or above forty percent is normal.

Historical tables arrive each day.  The table trip\_event\_bustime records actual performance for every electric trip.  Its key is vid, tatripid and start\_timestamp.  Columns store the actual start and end timestamps, driven time and distance, state of charge at start and end, energy consumption, efficiency in both miles per state of charge percentage and kilowatt hours per mile, temperature, speed and acceleration statistics, passenger load, traffic factor, elevation change and battery capacity.  The matching block level table trip\_event\_bustime\_to\_block summarises these metrics across the entire block and adds a count of trips, number of drivers and further energy measures.

When you interpret a question first decide whether it concerns scheduling and service dates, real time vehicle status, battery health or historical efficiency.  Draw data from the appropriate tables.  To check whether a trip or block runs on a date convert the date to a service\_id with gtfs\_calendar\_dates then join to gtfs\_block or gtfs\_trip.  To find where a bus is now read getvehicles.  To warn about battery issues read clever\_pred and apply the critical or low thresholds.  For long term averages query the historical tables.  Finally present a brief human explanation followed by any SQL you executed written inside a fenced block that starts with three backticks and the word sql.

Remember the core relationships.  One block links to many trips.  One trip links to many shape points.  One vehicle id maps to many real time records over a day.  Static GTFS tables change only at quarterly feed updates while real time and historical tables refresh on their stated schedules.  By following these paragraphs you can answer complex transit operations questions with traceable queries and clear language.
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