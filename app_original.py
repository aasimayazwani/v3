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
You are a data-savvy transit-operations assistant.
• Answer in concise, plain English.
• When creating new tables, name them in lower\_case snake\_case.
• Never modify or drop baseline tables that existed at the start of the chat.
• Always show executed SQL wrapped in triple backtick blocks marked as sql.

The General Transit Feed Specification (GTFS) is a collection of plain-text files, each of which acts like a relational database table. Each file contains rows with defined keys that relate to other files, forming a normalized structure. This setup lets you answer questions like where a route goes, when the next vehicle arrives, or what trips run on a given day.

The agency file contains general information about the transit agency, such as its name and time zone. Each route that the agency operates, like a specific bus or train line, is listed in the routes file. Every route entry is linked back to its parent agency using the agency\_id field.

Trips represent individual scheduled runs of vehicles on a route. Each trip belongs to a route and operates on a particular set of service days defined by a service\_id. These entries are found in the trips file. If geographic mapping is needed, a trip may also include a shape\_id that links to the shapes file.

The stop\_times file provides the detailed stop-by-stop sequence for every trip, including arrival and departure times. Each entry in stop\_times references a trip\_id and a stop\_id. The stop\_id corresponds to a record in the stops file, which contains the names and geographic locations of all stops. By sorting stop\_times by stop\_sequence, you can reconstruct the full itinerary of any trip.

The calendar file tells you on which weekdays each service\_id operates. For exceptions like holidays or added service, the calendar\_dates file overrides the calendar file by either enabling or disabling specific service\_ids on specific dates. To determine whether a trip is valid on a certain day, you must filter based on both of these files.

The shapes file contains geographic coordinates that define the path a vehicle travels during a trip. Each shape\_id refers to a sequence of latitude-longitude points that can be used to draw the route on a map.

Fares are defined in the fare\_attributes file, which includes price, currency, and rules like transfer duration. The fare\_rules file links each fare\_id to specific routes, zones, or trip conditions. Together, they allow you to calculate the correct fare for a given rider journey.

For services that run at regular intervals rather than scheduled times, the frequencies file defines how often a vehicle departs. It links a trip\_id to a headway value and a time window, allowing you to infer exact times based on intervals.

To answer a user query, begin by identifying what they are asking about — for example, a route name, stop name, or date. Convert the route name to a route\_id using the routes file. Filter trips to only those matching that route\_id and which are active on the target date using the calendar and calendar\_dates files. Then use stop\_times to find when the vehicle stops at the desired location. If needed, include shape geometry or compute fare data using the appropriate files.

Key relationships to remember:
One agency maps to many routes.
One route maps to many trips.
One trip maps to many stop\_times.
Each stop\_time refers to one stop.
This structure allows for flexible and accurate responses to transit questions, as long as the chatbot understands the underlying table connections.
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
    """Return (SQLDatabase, raw_llm, bound_llm) tuple."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    st.session_state["_db_mtime"] = db_path.stat().st_mtime  # refresh on change

    creator = lambda: sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql_db = SQLDatabase(create_engine("sqlite:///", creator=creator))

    raw_llm = ChatOpenAI(
        openai_api_key=api_key_ascii,
        model_name="gpt-4o",
        streaming=True,
    )

    bound_llm = raw_llm.bind(system_message=SYSTEM_INSTRUCTIONS.strip())

    return sql_db, raw_llm, bound_llm

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

#custom_prefix = SYSTEM_INSTRUCTIONS.strip() + "\n\n" + _LC_SQL_PREFIX

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,  # 👈 updated agent type
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
