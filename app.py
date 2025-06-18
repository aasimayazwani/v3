###############################################################################
# app.py – streamlined, hardened version for vehicles.db                      #
# (ChatGPT API + on‑the‑fly table download for *new* tables only)              #
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
# ---------- Configure DB connection (cached, auto-invalidated) --------------
###############################################################################
@st.cache_resource(ttl=0)
def get_db_connection(db_path: Path, api_key_ascii: str):
    """Return (SQLDatabase, ChatOpenAI LLM) tuple."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    st.session_state["_db_mtime"] = db_path.stat().st_mtime  # auto-invalidate

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
# Store list of tables that existed *before* the current chat session.
if "base_tables" not in st.session_state:
    with sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True) as _conn:
        st.session_state["base_tables"] = {
            row[0] for row in _conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
        }

###############################################################################
# ---------- LangChain agent --------------------------------------------------
###############################################################################

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

###############################################################################
# ---------- Table Download UI (new tables only) -----------------------------
###############################################################################
st.sidebar.markdown("### 📥 Download *new* Table as CSV")

# Current tables minus baseline → newly created in this session
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
                "create new tables with SQL and download them on the left!"
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
