###############################################################################
# app.py – streamlined, hardened version for vehicles.db                      #
# (Now using OpenAI ChatGPT API instead of Groq)                               #
###############################################################################
import os
import sys
import unicodedata
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine
import sqlite3
import pandas as pd  # Needed for CSV export

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI  # Switched from langchain_groq.ChatGroq

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

api_key_raw = st.sidebar.text_input("OpenAI API Key", type="password")  # label updated
api_key = ascii_sanitise(api_key_raw or "")

if not api_key:
    st.info("Please enter your OpenAI API key ↑ to begin.", icon="🔐")
    st.stop()

###############################################################################
# ---------- Configure DB connection (cached, auto‑invalidated) --------------
###############################################################################
@st.cache_resource(ttl=0)  # no TTL; cache busts automatically on file mtime
def get_db_connection(db_path: Path, api_key_ascii: str):
    """Return (SQLDatabase, ChatOpenAI LLM) tuple."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    st.session_state["_db_mtime"] = db_path.stat().st_mtime  # refresh trigger

    # SQLite read‑only URI
    creator = lambda: sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql_db = SQLDatabase(create_engine("sqlite:///", creator=creator))

    # LLM (headers already ASCII‑safe)
    llm = ChatOpenAI(
        openai_api_key=api_key_ascii,
        model_name="gpt-4o-mini",  # choose any ChatGPT model you prefer
        streaming=True,
    )
    return sql_db, llm


try:
    db, llm = get_db_connection(DB_FILE, api_key)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

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
# ---------- Table Download UI ------------------------------------------------
###############################################################################
st.sidebar.markdown("### 📥 Download Table as CSV")

with sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True) as conn:
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    )]

selected_table = st.sidebar.selectbox("Select a table", tables)

if selected_table:
    df = pd.read_sql_query(
        f"SELECT * FROM {selected_table};",
        sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True),
    )

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button(
        label=f"Download `{selected_table}.csv`",
        data=csv_bytes,
        file_name=f"{selected_table}.csv",
        mime="text/csv",
    )

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
            "content": "Hi there – ask me anything about the **VEHICLE** table!",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input("Ask a question…")

if user_query:
    st.chat_message("user").write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())

        try:
            response = agent.run(user_query, callbacks=[streamlit_callback])
        except UnicodeEncodeError:
            response = (
                "⚠️ I encountered a Unicode encoding issue while talking to the LLM. "
                "Please try rephrasing your question using plain ASCII characters."
            )
        except Exception as e:
            response = f"⚠️ Something went wrong:\n\n`{e}`"

        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
