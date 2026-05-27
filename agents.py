"""
Multi-Agent AI System for Automated Database Insights
Core agent logic — separated from UI layer for clean architecture.
"""

import sqlite3
import os
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

def init_db(path: str = "/tmp/shop.db") -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY, name TEXT, email TEXT, signup_date DATE)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL,
        status TEXT, order_date DATE, FOREIGN KEY(user_id) REFERENCES users(id))''')

    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        conn.executemany("INSERT INTO users VALUES (?,?,?,?)", [
            (1,'Alice','alice@example.com','2024-01-10'),
            (2,'Bob','bob@example.com','2024-03-15'),
            (3,'Charlie','charlie@example.com','2024-03-22'),
            (4,'Diana','diana@example.com','2024-04-05'),
            (5,'Eve','eve@example.com','2024-04-25'),
            (6,'Frank','frank@example.com','2024-05-15'),
            (7,'Grace','grace@example.com','2024-05-18'),
            (8,'Henry','henry@example.com','2024-05-20'),
            (9,'Ivy','ivy@example.com','2024-06-01'),
            (10,'Jack','jack@example.com','2024-06-05'),
            (11,'Kiran','kiran@example.com','2024-06-03'),
            (12,'Lata','lata@example.com','2024-06-04'),
            (13,'Manoj','manoj@example.com','2024-06-06'),
            (14,'Fatin','fatin@example.com','2024-12-04'),
            (15,'Oswa','oswa@example.com','2024-08-06'),
        ])

    if conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0:
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", [
            (1,1,250.00,'completed','2024-03-10'),
            (2,2,100.00,'pending','2024-03-16'),
            (3,3,320.00,'completed','2024-03-24'),
            (4,4,180.00,'completed','2024-05-01'),
            (5,5,210.00,'completed','2024-05-02'),
            (6,1,180.00,'completed','2024-03-18'),
            (7,2,120.00,'completed','2024-03-20'),
            (8,1,300.00,'completed','2024-04-01'),
            (9,3,80.00,'cancelled','2024-04-10'),
            (10,4,250.00,'pending','2024-05-05'),
            (11,6,400.00,'completed','2024-05-20'),
            (12,7,320.00,'completed','2024-05-22'),
            (13,8,150.00,'pending','2024-05-25'),
            (14,9,220.00,'completed','2024-06-02'),
            (15,10,500.00,'completed','2024-06-06'),
        ])

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Global connection (set at runtime)
# ---------------------------------------------------------------------------

_conn: sqlite3.Connection = None

def set_connection(conn: sqlite3.Connection):
    global _conn
    _conn = conn


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_schema() -> str:
    """Return the database schema."""
    schema = ""
    for table in ["users", "orders"]:
        rows = _conn.execute(f"PRAGMA table_info({table})").fetchall()
        cols = ", ".join([f"{r[1]} {r[2]}" for r in rows])
        schema += f"{table}({cols})\n"
    return schema.strip()


@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query and return results."""
    try:
        result = _conn.execute(query).fetchall()
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def generate_pdf_report(text: str, filename: str = "report.pdf") -> str:
    """Generate a PDF report from text and return its file path."""
    from fpdf import FPDF
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", size=12)
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

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_node: Literal["analyst", "expert", "reviewer", "END"]


# ---------------------------------------------------------------------------
# Generic subgraph builder
# ---------------------------------------------------------------------------

def build_subgraph(llm, tools_list, system_prompt: str):
    bound = llm.bind_tools(tools_list)
    sys_msgs = [SystemMessage(content=system_prompt)]

    def agent_node(state: AgentState) -> AgentState:
        response = bound.invoke(sys_msgs + state["messages"])
        return {"messages": [response]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(tools_list))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_pipeline(openai_api_key: str):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = ChatOpenAI(model="gpt-4o-mini")

    analyst_app = build_subgraph(llm, [get_schema],
        "You are a data analyst. Use get_schema to understand the database, "
        "then ask at least 10 insightful analytical questions in one response.")

    expert_app = build_subgraph(llm, [get_schema, execute_sql],
        "You are a data expert. Answer the analyst's questions by querying the database with execute_sql.")

    reviewer_app = build_subgraph(llm, [generate_pdf_report],
        "You are an expert reviewer. Summarize the database analysis in exactly 8 lines "
        "with 2 actionable insights at the end. Then call generate_pdf_report with your summary.")

    # Supervisor
    class AgentSelector(BaseModel):
        next_node: Literal["analyst", "expert", "reviewer", "END"] = Field(
            description="Which agent to route to next, or END when done.")

    selector_llm = llm.with_structured_output(AgentSelector)
    supervisor_sys = [SystemMessage(content=(
        "You are a supervisor orchestrating analyst, expert, and reviewer agents. "
        "Always run them in order: analyst first, then expert, then reviewer, then END."))]

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
    graph.add_node("analyst", analyst_app)
    graph.add_node("expert", expert_app)
    graph.add_node("reviewer", reviewer_app)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route)
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("expert", "supervisor")
    graph.add_edge("reviewer", "supervisor")
    return graph.compile()


def run_pipeline(pipeline, user_prompt: str):
    inputs = {"messages": [HumanMessage(content=user_prompt)]}
    for output in pipeline.stream(inputs):
        for node, value in output.items():
            yield node, value.get("messages", [])
