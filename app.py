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
from langchain.agents import create_sql_agent, AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
import re
# Attempt to pull LangChain's default SQL prompt so we can append our own.
try:
    from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX as _LC_SQL_PREFIX
except Exception:  # Fallback if import path changes
    _LC_SQL_PREFIX = ""

def extract_markdown_table(markdown_text: str) -> pd.DataFrame | None:
    """Try to extract a DataFrame from a markdown table in a string."""
    lines = markdown_text.strip().splitlines()
    table_lines = [line for line in lines if "|" in line]
    if len(table_lines) >= 2:
        csv_text = "\n".join(
            line.strip().strip("|").replace("|", ",")
            for line in table_lines
            if "---" not in line
        )
        try:
            return pd.read_csv(StringIO(csv_text))
        except Exception:
            return None
    return None

def display_response_with_downloads(response) -> str:
    """Display a response and return the assistant reply content (for chat history)."""
    response_df = None

    if isinstance(response, pd.DataFrame):
        response_df = response
    elif isinstance(response, str):
        response_df = extract_markdown_table(response)

    if response_df is not None:
        st.dataframe(response_df)

        format_choice = st.radio(
            "📁 Choose download format:",
            options=["CSV", "PDF"],
            horizontal=True,
            index=0,
        )

        if format_choice == "CSV":
            st.download_button(
                label="📥 Download as CSV",
                data=response_df.to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",
                mime="text/csv",
            )
        elif format_choice == "PDF":
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            data = [response_df.columns.tolist()] + response_df.values.tolist()
            table = Table(data)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            doc.build([table])
            pdf_bytes = pdf_buffer.getvalue()

            st.download_button(
                label="📥 Download as PDF",
                data=pdf_bytes,
                file_name="query_result.pdf",
                mime="application/pdf",
            )

        return "Here is the table you requested. Use the buttons above to download it as CSV or PDF."
    else:
        st.write(response)
        return response


###############################################################################
# ---------- Developer system instructions -----------------------------------
###############################################################################
FORMAT_INSTRUCTIONS = """
Use the following format in your response:

Thought: Do I need to use a tool? Yes or No.
Action: the action to take, should be one of [sql_db_list_tables, sql_db_schema, sql_db_query]
Action Input: the input to the action

OR

Thought: Do I need to use a tool? No.
Final Answer: the final answer to the original input question. You must always use the phrase 'Final Answer:' exactly, or the system will throw a parsing error.
"""
SYSTEM_INSTRUCTIONS = """ 
System Instruction for SQL Chatbot Agent: Enhancing Table Selection Accuracy

 Primary Goal
Your primary goal is to correctly identify and utilize the appropriate database tables based on user queries. This involves understanding the users intent, matching it to the correct table(s), and handling any ambiguities or errors effectively.

 Guidelines for Table Selection

- Context Analysis:  
  Analyze the context and intent behind user queries to determine what data is being requested. For example, if a user asks about the current location of a vehicle, recognize this as a query related to real-time vehicle data. If the query involves historical performance, identify it as historical data.

- Table Lookup:  
  Search the predefined database schema for tables relevant to the identified context. For instance, use getvehicles for real-time vehicle data or trip_event_bustime for historical trip statistics.

- Table Matching:  
  Prioritize tables that best correspond to key query terms or user-specified criteria. If the user mentions "block information," prioritize tables like gtfs_block. For predictions like "State of Charge," consider clever_pred.

- Error Handling:  
  If table identification is ambiguous or fails, use fallback mechanisms:
  - Ask the user for clarification (e.g., "Do you want real-time or historical data?").
  - Suggest possible tables based on partial matches (e.g., "Did you mean getvehicles for current status?").

 Detailed Steps for Table Selection

1. Extract Key Terms and Intent:  
   Identify key terms (e.g., "current location," "predicted SOC," "historical trips") and determine the intent (e.g., real-time status, prediction, historical analysis).

2. Map to Table Categories:  
   Map the terms and intent to relevant table categories:  
   - Real-time data: getvehicles, clever_pred  
   - Historical data: trip_event_bustime, trip_event_bustime_to_block  
   - Static data: gtfs_block, bus_vid

3. Search Database Schema:  
   Search the schema for tables matching the identified categories, using table descriptions and common query patterns as guides.

4. Narrow Down Selection:  
   If multiple tables match, use additional context or query terms to refine the choice. For example, "end-of-block SOC" points to clever_pred rather than getvehicles.

5. Handle Ambiguities:  
   If no tables match or ambiguity persists, prompt the user for more details or suggest possible tables based on partial matches.

 Key Variables and Relationships

To accurately select tables and formulate queries, its essential to understand the key variables in each table and how they connect across the database. Below is a detailed breakdown by table, including variable definitions and their relationships:

 getvehicles (Real-Time Vehicle Data)
- Key Variables:
  - vid (Vehicle ID): Unique identifier for a vehicle, linking to other tables.
  - lat (Latitude): Geographic latitude of the vehicles current position.
  - lon (Longitude): Geographic longitude of the vehicles current position.
  - rt (Route Tag): Indicates the route the vehicle is currently assigned to.
  - tablockid (Users Block Identifier): Identifies the block (sequence of trips) the vehicle is operating.
  - timestamp: Time of the latest vehicle update.
  - route_id: Identifier for the route, consistent with GTFS standards.
- Relationships:
  - vid links to trip_event_bustime, clever_pred, and bus_vid for vehicle-specific data.
  - tablockid connects to gtfs_block and trip_event_bustime_to_block for block details.
  - route_id aligns with gtfs_block and trip_event_bustime for route-specific queries.

 trip_event_bustime (Historical Trip Data)
- Key Variables:
  - vid (Vehicle ID): Identifies the vehicle that performed the trip.
  - start_timestamp: Start time of the trip.
  - end_timestamp: End time of the trip.
  - route_id: Route associated with the trip.
  - trip_id: Unique identifier for the trip.
  - shape_id: Identifier for the trips geographic path.
  - tablockid (Users Block Identifier): Links to the block this trip belongs to.
- Relationships:
  - vid connects to getvehicles, clever_pred, and bus_vid.
  - tablockid links to gtfs_block and trip_event_bustime_to_block.
  - route_id, trip_id, and shape_id relate to gtfs_block for static trip and route data.

 clever_pred (Prediction Data)
- Key Variables:
  - vid (Vehicle ID): Vehicle for which the prediction is made.
  - soc (State of Charge): Predicted battery charge level (e.g., for electric buses).
  - timestamp: Time of the prediction.
- Relationships:
  - vid links to getvehicles, trip_event_bustime, and bus_vid.
  - timestamp can be correlated with getvehicles.timestamp for real-time context.

 gtfs_block (Static Block Data)
- Key Variables:
  - tablockid (Users Block Identifier): Unique block identifier.
  - route_id: Route associated with the block.
  - trip_id: Trip within the block.
  - shape_id: Shape of the trips path.
- Relationships:
  - tablockid connects to getvehicles and trip_event_bustime_to_block.
  - route_id, trip_id, and shape_id align with trip_event_bustime.

 bus_vid (Static Vehicle Specs)
- Key Variables:
  - vid (Vehicle ID): Unique vehicle identifier.
  - (Other static attributes like model or capacity may exist but are not specified here.)
- Relationships:
  - vid links to getvehicles, trip_event_bustime, and clever_pred.

 trip_event_bustime_to_block (Historical Block Mapping)
- Key Variables:
  - tablockid (Users Block Identifier): Block identifier.
  - (Likely includes trip or event references, though not fully specified.)
- Relationships:
  - tablockid connects to getvehicles and gtfs_block.


Clarifying Efficiency vs. Performance in Transit Dispatch Operations:
In the context of a transit agency, particularly for dispatch operations involving electric or hybrid vehicles, it's important to distinguish between efficiency and performance:

Efficiency refers specifically to how effectively energy is used during vehicle operations. It is typically quantified in one of two ways:

Energy Efficiency — measured as kilowatt-hours per mile (kWh/mi).

This reflects how much energy a vehicle consumes to cover a given distance.

A lower kWh/mi value indicates greater energy efficiency.

End-of-Trip State of Charge (SOC) — the remaining battery charge at the end of a vehicles scheduled block or shift.

Higher end-of-trip SOC implies energy was used conservatively, leaving more residual range.

This is useful for planning block chaining and avoiding mid-shift charging.

Efficiency focuses on energy conservation and optimization for electric fleet planning.

By understanding these variables and their interconnections, you can select the appropriate tables and join them effectively for queries requiring data from multiple sources (e.g., joining getvehicles and clever_pred on vid for real-time location and predicted SOC).

 Mechanisms for Ongoing Learning and Adaptation

- Feedback Loop:  
  Learn from user feedback on table selection accuracy and adjust logic to improve future performance.

- Interaction History:  
  Analyze past interactions to identify patterns in user queries and successful table matches, refining the mapping of terms to tables.

- Periodic Updates:  
  Regularly update the table schema and selection logic based on database changes or evolving user query patterns.

 Example Application
- Query: "What is the current location of bus 2401?"  
  - Step 1: Key terms: "current location" → Intent: real-time data.  
  - Step 2: Category: real-time data → Tables: getvehicles.  
  - Variables: Use vid, lat, lon, timestamp.

- Query: "What is the predicted SOC for bus 2402?"  
  - Step 1: Key terms: "predicted SOC" → Intent: prediction.  
  - Step 2: Category: predictions → Tables: clever_pred.  
  - Variables: Use vid, soc, timestamp.

- Query: "Tell me about bus 2403s last trip."  
  - Step 1: Key terms: "last trip" → Intent: historical data.  
  - Step 2: Category: historical data → Tables: trip_event_bustime.  
  - Variables: Use vid, start_timestamp, end_timestamp, route_id.

If the result includes more than one row or multiple related statistics, always present the output in a tabular format using markdown. Example:

| Metric                  | Value   |
|-------------------------|---------|
| Average kWh/mile        | 2.037   |
| Minimum kWh/mile        | 1.967   |
| Maximum kWh/mile        | 2.107   |

This formatting helps users view and download data easily. Always default to this style whenever numeric results can be structured.
"""

def extract_raw_sql(text: str) -> str:
    """Extracts the raw SQL query from markdown-fenced output, or returns raw."""
    match = re.search(r"```sql\s+(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()



def is_transit_related(query: str, api_key: str) -> bool:
    """Check if the user's query is related to fleet/transit/dispatch (for OpenAI SDK v1.0+)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    prompt = (
        "Is the following question about public transportation, electric buses, "
        "transit dispatch, scheduling, or vehicle telematics? Reply with only Yes or No.\n\n"
        f"Question: {query.strip()}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception as e:
        st.warning(f"⚠️ Domain check failed: {e}")
        return True  # fallback to allow query
    
###############################################################################
# ---------- Utility helpers --------------------------------------------------
###############################################################################
DB_FILE = Path(__file__).parent / "vehicles.db"


def ascii_sanitise(value: str) -> str:
    """Return a strictly-ASCII version of `value`."""
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
# ---------- Configure DB connection (cached, auto-invalidated) --------------
###############################################################################
@st.cache_resource(ttl=0)
def get_db_connection(db_path: Path, api_key_ascii: str):
    """Return (SQLDatabase, ChatOpenAI LLM) tuple."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    st.session_state["_db_mtime"] = db_path.stat().st_mtime  # refresh on change

    from sqlalchemy.pool import StaticPool

    # ⬇️ Open in read-only mode so no accidental writes occur
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
        model_name="gpt-4o",
        streaming=True,
        temperature=0,
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
# custom_prefix = SYSTEM_INSTRUCTIONS.strip() + "\n\n" + _LC_SQL_PREFIX
custom_prefix = (
    SYSTEM_INSTRUCTIONS.strip()
    + "\n\n"
    + FORMAT_INSTRUCTIONS.strip()
    + "\n\n"
    + _LC_SQL_PREFIX.strip()
)

_base_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prefix=custom_prefix,
    verbose=True,
)

agent = AgentExecutor.from_agent_and_tools(
    agent=_base_agent.agent,
    tools=toolkit.get_tools(),
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=10,         # Adjust this as needed
    max_execution_time=30,    # In seconds
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
                "Hi there – ask me anything about the **vehicles.db** tables. "
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
            chat_reply = display_response_with_downloads(response)

        except UnicodeEncodeError:
            chat_reply = (
                "⚠️ I encountered a Unicode encoding issue while talking to the LLM. "
                "Please try rephrasing your question using plain ASCII characters."
            )
        except Exception as e:
            chat_reply = f"⚠️ Something went wrong:\n\n`{e}`"

        st.write(chat_reply)
        st.session_state.messages.append({"role": "assistant", "content": chat_reply})
