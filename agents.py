"""
Multi-Agent AI System for Automated Database Insights
Core agent logic — separated from UI layer for clean architecture.
"""

import sqlite3
import os
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langchain.agents import tool
from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

def init_db(path: str = "shop.db") -> sqlite3.Connection:
    """Initialize the SQLite database with sample data."""
    conn = sqlite3.connect(path, check_same_thread=False)

    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        signup_date DATE
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        amount REAL,
        status TEXT,
        order_date DATE,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    # Only insert if tables are empty
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        users = [
            (1, 'Alice',   'alice@example.com',   '2024-01-10'),
            (2, 'Bob',     'bob@example.com',     '2024-03-15'),
            (3, 'Charlie', 'charlie@example.com', '2024-03-22'),
            (4, 'Diana',   'diana@example.com',   '2024-04-05'),
            (5, 'Eve',     'eve@example.com',     '2024-04-25'),
            (6, 'Frank',   'frank@example.com',   '2024-05-15'),
            (7, 'Grace',   'grace@example.com',   '2024-05-18'),
            (8, 'Henry',   'henry@example.com',   '2024-05-20'),
            (9, 'Ivy',     'ivy@example.com',     '2024-06-01'),
            (10, 'Jack',   'jack@example.com',    '2024-06-05'),
            (11, 'Kiran',  'kiran@example.com',   '2024-06-03'),
            (12, 'Lata',   'lata@example.com',    '2024-06-04'),
            (13, 'Manoj',  'manoj@example.com',   '2024-06-06'),
            (14, 'Fatin',  'fatin@example.com',   '2024-12-04'),
            (15, 'Oswa',   'oswa@example.com',    '2024-08-06'),
        ]
        conn.executemany("INSERT INTO users VALUES (?,?,?,?)", users)

    if conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0:
        orders = [
            (1,  1,  250.00, 'completed', '2024-03-10'),
            (2,  2,  100.00, 'pending',   '2024-03-16'),
            (3,  3,  320.00, 'completed', '2024-03-24'),
            (4,  4,  180.00, 'completed', '2024-05-01'),
            (5,  5,  210.00, 'completed', '2024-05-02'),
            (6,  1,  180.00, 'completed', '2024-03-18'),
            (7,  2,  120.00, 'completed', '2024-03-20'),
            (8,  1,  300.00, 'completed', '2024-04-01'),
            (9,  3,   80.00, 'cancelled', '2024-04-10'),
            (10, 4,  250.00, 'pending',   '2024-05-05'),
            (11, 6,  400.00, 'completed', '2024-05-20'),
            (12, 7,  320.00, 'completed', '2024-05-22'),
            (13, 8,  150.00, 'pending',   '2024-05-25'),
            (14, 9,  220.00, 'completed', '2024-06-02'),
            (15, 10, 500.00, 'completed', '2024-06-06'),
        ]
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Tools (defined at module level; conn injected at runtime)
# ---------------------------------------------------------------------------

_conn: sqlite3.Connection = None


def set_connection(conn: sqlite3.Connection):
    global _conn
    _conn = conn


@tool
def get_schema() -> str:
    """Return the database schema as a string."""
    schema = ""
    for table in ["users", "orders"]:
        rows = _conn.execute(f"PRAGMA table_info({table})").fetchall()
        cols = ", ".join([f"{r[1]} {r[2]}" for r in rows])
        schema += f"{table}({cols})\n"
    return schema.strip()


@tool
def execute_sql(query: str) -> str:
    """Execute a read-only SQL query and return results."""
    try:
        result = _conn.execute(query).fetchall()
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def generate_pdf_report(text: str, filename: str = "report.pdf") -> str:
    """Generate a PDF report from the given text and return its file path."""
    from fpdf import FPDF  # fpdf2 package
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 10, line)
        file_path = f"/tmp/{filename}"
        pdf.output(file_path)
        return file_path
    except Exception as e:
        return f"PDF generation failed: {str(e)}"


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

class AnalystState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ExpertState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ReviewerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_node: Literal["analyst", "expert", "reviewer", "END"]


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def build_graph(llm, tools_list, system_prompt: str, state_cls):
    """Generic helper to build a single-agent subgraph."""
    bound_llm = llm.bind_tools(tools_list)
    sys_msg = [SystemMessage(content=system_prompt)]

    def agent_node(state):
        response = bound_llm.invoke(sys_msg + state["messages"])
        return {"messages": [response]}

    g = StateGraph(state_cls)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(tools_list))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


# ---------------------------------------------------------------------------
# Main pipeline factory
# ---------------------------------------------------------------------------

def build_pipeline(openai_api_key: str) -> tuple:
    """Build and return the final compiled LangGraph pipeline."""
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = ChatOpenAI(model="gpt-4o-mini")

    analyst_app = build_graph(
        llm,
        [get_schema],
        (
            "You are a data analyst. Start by understanding the database schema using tools. "
            "Then ask at least 10 insightful questions in a single response."
        ),
        AnalystState,
    )

    expert_app = build_graph(
        llm,
        [get_schema, execute_sql],
        "You are a data expert. Use tools to answer the analyst's questions by querying the database.",
        ExpertState,
    )

    reviewer_app = build_graph(
        llm,
        [generate_pdf_report],
        (
            "You are an expert reviewer summarizing database analysis reports. "
            "Produce a concise summary in exactly eight lines with 2 actionable insights at the end. "
            "Then generate a PDF report using the generate_pdf_report tool."
        ),
        ReviewerState,
    )

    # --- Supervisor ---
    class AgentSelector(BaseModel):
        next_node: Literal["analyst", "expert", "reviewer", "END"] = Field(
            description="Route to the next agent if needed, else END"
        )

    selector_llm = llm.with_structured_output(AgentSelector)
    supervisor_sys = [SystemMessage(content=(
        "You are a supervisor orchestrating [analyst, expert, reviewer]. "
        "Run analyst first, then expert, then reviewer, then END."
    ))]

    def supervisor(state: SupervisorState) -> SupervisorState:
        response = selector_llm.invoke(supervisor_sys + state["messages"])
        return {
            "messages": [AIMessage(content=f"Routing to: {response.next_node}")],
            "next_node": response.next_node,
        }

    def route(state: SupervisorState):
        return "__end__" if state.get("next_node") == "END" else state["next_node"]

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("analyst",   analyst_app)
    graph.add_node("expert",    expert_app)
    graph.add_node("reviewer",  reviewer_app)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route)
    graph.add_edge("analyst",  "supervisor")
    graph.add_edge("expert",   "supervisor")
    graph.add_edge("reviewer", "supervisor")

    return graph.compile()


def run_pipeline(pipeline, user_prompt: str):
    """Stream outputs from the pipeline and yield (node, messages) tuples."""
    inputs = {"messages": [HumanMessage(content=user_prompt)]}
    for output in pipeline.stream(inputs):
        for node, value in output.items():
            yield node, value.get("messages", [])
