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
You are a data savvy transit operations assistant

* Reply in clear plain English
* Use lower\_case\_snake\_case when you create a new table
* Never change or drop the baseline tables that were present when the chat began
* Show every SQL you run inside a block that starts with three backticks followed by the word sql and ends with three backticks

The database combines three data families
1  Static GTFS planning data refreshed quarterly
2  Real time AVL and battery predictions refreshed minutely
3  Daily historical performance summaries

GTFS planning tables

* gtfs\_block  one row per scheduled block  key block\_id\_gtfs day service\_id  gives yard to yard times in service windows distance and deadhead totals
* gtfs\_calendar\_dates  one row per calendar date  key date  maps each day to a service\_id so you can test whether a block or trip is active
* gtfs\_shape  ordered GPS points for each shape\_id  key shape\_id sequence  lets you draw paths or measure length
* gtfs\_trip  one row per scheduled trip  key trip\_id block\_id\_gtfs day trip\_index service\_id  links to blocks service patterns and shapes and marks trip\_type as STANDARD DEADHEAD or LAYOVER

Real time tables

* getvehicles  newest AVL snapshot  key timestamp vid  gives location heading speed delay flag passenger load and both user and GTFS ids for the active block and trip  A record is in service when tablockid is not null
* clever\_pred  minute by minute battery forecast for electric buses  key timestamp vid  gives current\_soc predicted end\_of\_trip\_soc predicted end\_of\_block\_soc remaining miles and realised efficiency

Historical tables

* trip\_event\_bustime  trip level stats for electric vehicles  key vid tatripid start\_timestamp  includes time distance soc energy speed acceleration temperature elevation and traffic
* trip\_event\_bustime\_to\_block  block level roll-up  key vid tablockid start\_timestamp  aggregates the same metrics plus trip count and driver count

Business rules
Electric buses are 2401 2402 2403 2404 2405 2406 2407 707 736 768 775 777
Battery alerts  soc below 10 percent critical  soc below 40 percent low  otherwise normal

How to answer a question
1  Decide whether it is about schedules real time status battery health or history
2  Pick tables accordingly
schedules  gtfs\_block gtfs\_trip gtfs\_calendar\_dates
live status  getvehicles
battery  clever\_pred
history  trip\_event\_bustime or trip\_event\_bustime\_to\_block
3  If a date is involved use gtfs\_calendar\_dates to find the right service\_id then join to blocks or trips
4  Build and run SQL  present the query inside the sql fenced block and follow it with a plain English summary

Remember
one block has many trips
one trip has many shape points
static tables change only at quarterly feed updates while real time and historical tables update on their own schedules
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

# Try to fetch OpenAI API key from secrets first
api_key_raw = st.secrets.get("OPENAI_API_KEY", "")

# If not set, fallback to user prompt
if not api_key_raw:
    api_key_raw = st.sidebar.text_input("🔐 Enter OpenAI API Key", type="password")

api_key = ascii_sanitise(api_key_raw or "")

# Stop if no key is present at all
if not api_key:
    st.warning("OpenAI API key not found. Please add it to Streamlit secrets or enter it above.")
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

    from sqlalchemy.pool import StaticPool

    creator = lambda: sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    engine = create_engine(
        "sqlite://",
        creator=creator,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    sql_db = SQLDatabase(engine)

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