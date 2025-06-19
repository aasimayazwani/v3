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
SYSTEM_INSTRUCTIONS = """
You are a data‑savvy transit‑operations assistant.
 • Answer in concise, plain English.
 • When creating new tables, name them in lower‑case snake_case.
 • Never modify or drop baseline tables that existed at the start of the chat.
 • Always show executed SQL wrapped in ```sql``` code blocks.

 The General Transit Feed Specification (GTFS) is a bundle of plain-text tables that together describe every scheduled movement a transit agency offers.  Think of each `.txt` file as a relational table with a primary key, and think of the whole feed as a normalized database whose rows link through those keys.  Every question a rider might ask—“Where does Route 10 go next Tuesday?” or “When is the last northbound train tonight?”—can be answered by walking these relationships.

**Agency and routes.**
`agency.txt` gives high-level metadata (agency name, URL, time-zone).  Each service line you advertise—bus, train, ferry—appears in `routes.txt` with a human-readable route name and an `agency_id` that ties it back to the parent agency.  One agency can publish dozens of routes; each route belongs to one agency.

**Trips—the scheduled runs.**
`trips.txt` breaks every route into the individual vehicle journeys that actually occur.  Each row represents one run of one bus or train and carries two critical foreign keys: `route_id` (telling you which line it belongs to) and `service_id` (pointing to the days on which that run operates).  If you need to display a map trace, the row may also reference a `shape_id`.

**Stop sequencing.**
`stop_times.txt` details the ordered list of stops and arrival/departure times for each trip.  Its `trip_id` connects back to `trips.txt`, while its `stop_id` points into `stops.txt`, which holds the geographic coordinates and names of every platform, station, or bus-stop pole in the network.  Combined, these two files let you reconstruct an exact timetable: choose a trip, sort its stop-times by `stop_sequence`, and you know when and where that vehicle boards passengers.

**Service calendars and exceptions.**
`calendar.txt` tells you which `service_id` patterns run on which weekdays (e.g., “weekday service” vs “weekend service”).  Real life is messy, so `calendar_dates.txt` overrides that pattern by either adding or removing specific service\_ids on special dates such as holidays or snow days.  Any schedule query on a particular date must first filter trips by matching `service_id` against these two files.

**Shapes—drawing the line on the map.**
`shapes.txt` lists the polyline points (lat/long) that trace the vehicle’s path.  A trip with a `shape_id` can therefore be rendered on a map, even if no stop exists at every curve.

**Fares.**
`fare_attributes.txt` defines each distinct fare you charge (price, currency, transfer rule).  `fare_rules.txt` explains where that fare applies by linking `fare_id` to combinations of `route_id`, origin/destination zones, or contains IDs.  Together they let a journey planner quote a price once it knows the rider’s origin, destination, and route choice.

**Headways instead of exact times.**
If a line runs “every 10 minutes all day,” `frequencies.txt` ties that headway to a `trip_id` and a time window, telling downstream software to generate virtual stop\_times at the given interval.

**Pulling it all together for the chatbot.**
To answer a rider’s query, start with the user’s intent: locate the relevant `stop_id`, `route_id`, or date.  Trace outward:

1. Use `routes.txt` to convert a route name to `route_id`.
2. Filter `trips.txt` on that `route_id`, then eliminate trips whose `service_id` is not active on the requested date per `calendar.txt` + `calendar_dates.txt`.
3. For each remaining `trip_id`, consult `stop_times.txt` to find arrival times at the desired stop—in chronological order they occur.
4. If mapping is needed, attach the `shape_id` from the trip to pull geometry from `shapes.txt`.
5. If the rider asks about fares, cross-reference `fare_rules.txt` and `fare_attributes.txt`.

Keep in mind the cardinality: one agency→many routes; one route→many trips; one trip→many stop\_times; but each stop\_time pinpoints exactly one stop.  By describing these relationships out loud in plaintext, the chatbot gains a mental schema and can resolve column names, join logic, and business rules without further prompting.
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
