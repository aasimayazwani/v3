###############################################################################
# app.py – streamlined, hardened version for vehicles.db
# (ChatGPT API + developer-defined system instructions)
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

# Pull LangChain’s default SQL prefix so we can append our own instructions
try:
    from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX as _LC_SQL_PREFIX
except Exception:
    _LC_SQL_PREFIX = ""

###############################################################################
# ---------- Developer system instructions -----------------------------------
###############################################################################
SYSTEM_INSTRUCTIONS = """
You are a data-savvy transit-operations assistant.
• Answer in concise, plain English.
• When creating new tables, name them in lower_case snake_case.
• Never modify or drop baseline tables that existed at the start of the chat.
• Always show executed SQL wrapped in triple backtick blocks marked as sql.

The General Transit Feed Specification (GTFS) consists of plain-text tables
whose keys link like a relational database. Typical workflow:

• Convert a spoken route name to route_id via routes table.
• Filter trips on that route_id plus service_id active on target date
  (calendar + calendar_dates).
• Use stop_times to find stop sequence / times, joining stops for names & coords.
• Optionally join shapes for geometry and fare_rules / fare_attributes for price.

Key relationships:
• One agency → many routes
• One route  → many trips
• One trip   → many stop_times
• Each stop_time references one stop
"""

###############################################################################
# ---------- Helper utilities -------------------------------------------------
###############################################################################
DB_FILE = Path(__file__).parent / "vehicles.db"


def ascii_sanitise(val: str) -> str:
    return (
        unicodedata.normalize("NFKD", val)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )

###############################################################################
# ---------- Streamlit setup --------------------------------------------------
###############################################################################
st.set_page_config(page_title="LangChain • Vehicles DB", page_icon="🚌")
st.title("🚌 Chat with Vehicles Database")

api_key_raw = st.sidebar.text_input("OpenAI API Key", type="password")
api_key = ascii_sanitise(api_key_raw or "")

if not api_key:
    st.info("Please enter your OpenAI API key ↑ to begin.", icon="🔐")
    st.stop()

###############################################################################
# ---------- Cached DB + LLM loader ------------------------------------------
###############################################################################
@st.cache_resource(ttl=0)
def get_db_and_llm(db_path: Path, api_key_ascii: str):
    """Return (SQLDatabase, ChatOpenAI) tuple."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    st.session_state["_db_mtime"] = db_path.stat().st_mtime  # invalidate cache on change

    creator = lambda: sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql_db = SQLDatabase(create_engine("sqlite:///", creator=creator))

    llm = ChatOpenAI(
        openai_api_key=api_key_ascii,
        model_name="gpt-4o",
        streaming=True,
    )
    return sql_db, llm


db, llm = get_db_and_llm(DB_FILE, api_key)

###############################################################################
# ---------- Baseline tables snapshot ----------------------------------------
###############################################################################
if "base_tables" not in st.session_state:
    with sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True) as _conn:
        st.session_state["base_tables"] = {
            row[0]
            for row in _conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        }

###############################################################################
# ---------- LangChain agent --------------------------------------------------
###############################################################################
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Merge our system instructions with LangChain’s standard SQL prefix
custom_prefix = SYSTEM_INSTRUCTIONS.strip() + "\n\n" + _LC_SQL_PREFIX

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    prefix=custom_prefix,
)

###############################################################################
# ---------- Table-download sidebar ------------------------------------------
###############################################################################
st.sidebar.markdown("### 📥 Download *new* Table as CSV")

with sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True) as _conn:
    current_tables = {
        row[0]
        for row in _conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    }

new_tables = sorted(current_tables - st.session_state["base_tables"])

if new_tables:
    sel = st.sidebar.selectbox("Select a new table", new_tables)
    if sel:
        new_df = pd.read_sql_query(
            f"SELECT * FROM {sel};",
            sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True),
        )
        st.sidebar.download_button(
            label=f"Download `{sel}.csv`",
            data=new_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{sel}.csv",
            mime="text/csv",
        )
else:
    st.sidebar.info("No new tables have been created in this chat session yet.")

###############################################################################
# ---------- Chat UI ----------------------------------------------------------
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

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

user_query = st.chat_input("Ask a question…")

if user_query:
    st.chat_message("user").write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container())
        try:
            answer = agent.run(user_query, callbacks=[cb])
        except UnicodeEncodeError:
            answer = (
                "⚠️ Unicode encoding issue. Try re-phrasing using plain ASCII characters."
            )
        except Exception as exc:
            answer = f"⚠️ Something went wrong:\n\n`{exc}`"

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
