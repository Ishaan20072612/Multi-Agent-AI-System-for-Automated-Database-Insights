# 1. Install Dependencies

# pip install -q langchain_openai langgraph langchain fpdf

# 2. Import Libraries

import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
import sqlite3
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel, Field
from IPython.display import Image, display
from langchain.agents import tool
from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langchain_core.messages import HumanMessage

# 3. Database Setup

conn = sqlite3.connect("shop.db", check_same_thread=False)
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

conn.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com', '2024-01-10')")
conn.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@example.com', '2024-03-15')")
conn.execute("INSERT INTO users VALUES (3, 'Charlie', 'charlie@example.com', '2024-03-22')")
conn.execute("INSERT INTO users VALUES (4, 'Diana', 'diana@example.com', '2024-04-05')")
conn.execute("INSERT INTO users VALUES (5, 'Eve', 'eve@example.com', '2024-04-25')")
conn.execute("INSERT INTO users VALUES (6, 'Frank', 'frank@example.com', '2024-05-15')")
conn.execute("INSERT INTO users VALUES (7, 'Grace', 'grace@example.com', '2024-05-18')")
conn.execute("INSERT INTO users VALUES (8, 'Henry', 'henry@example.com', '2024-05-20')")
conn.execute("INSERT INTO users VALUES (9, 'Ivy', 'ivy@example.com', '2024-06-01')")
conn.execute("INSERT INTO users VALUES (10, 'Jack', 'jack@example.com', '2024-06-05')")
conn.execute("INSERT INTO users VALUES (11, 'Kiran', 'kiran@example.com', '2024-06-03')")
conn.execute("INSERT INTO users VALUES (12, 'Lata', 'lata@example.com', '2024-06-04')")
conn.execute("INSERT INTO users VALUES (13, 'Manoj', 'manoj@example.com', '2024-06-06')")
conn.execute("INSERT INTO users VALUES (14, 'Fatin', 'fatin@example.com', '2024-12-04')")
conn.execute("INSERT INTO users VALUES (15, 'Oswa', 'oswa@example.com', '2024-08-06')")

conn.execute("INSERT INTO orders VALUES (1, 1, 250.00, 'completed', '2024-03-10')")
conn.execute("INSERT INTO orders VALUES (2, 2, 100.00, 'pending', '2024-03-16')")
conn.execute("INSERT INTO orders VALUES (3, 3, 320.00, 'completed', '2024-03-24')")
conn.execute("INSERT INTO orders VALUES (4, 4, 180.00, 'completed', '2024-05-01')")
conn.execute("INSERT INTO orders VALUES (5, 5, 210.00, 'completed', '2024-05-02')")
conn.execute("INSERT INTO orders VALUES (6, 1, 180.00, 'completed', '2024-03-18')")
conn.execute("INSERT INTO orders VALUES (7, 2, 120.00, 'completed', '2024-03-20')")
conn.execute("INSERT INTO orders VALUES (8, 1, 300.00, 'completed', '2024-04-01')")
conn.execute("INSERT INTO orders VALUES (9, 3, 80.00, 'cancelled', '2024-04-10')")
conn.execute("INSERT INTO orders VALUES (10, 4, 250.00, 'pending', '2024-05-05')")
conn.execute("INSERT INTO orders VALUES (11, 6, 400.00, 'completed', '2024-05-20')")
conn.execute("INSERT INTO orders VALUES (12, 7, 320.00, 'completed', '2024-05-22')")
conn.execute("INSERT INTO orders VALUES (13, 8, 150.00, 'pending', '2024-05-25')")
conn.execute("INSERT INTO orders VALUES (14, 9, 220.00, 'completed', '2024-06-02')")
conn.execute("INSERT INTO orders VALUES (15, 10, 500.00, 'completed', '2024-06-06')")
conn.commit()

# 4. Tools

@tool
def get_schema() -> str:
    schema = ""
    for table in ["users", "orders"]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        cols = ",".join([f"{r[1]} {r[2]}" for r in rows])
        schema += f"{table}({cols})\n"
    return schema.strip()

@tool
def execute_sql(query: str) -> str:
    try:
        result = conn.execute(query).fetchall()
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def generate_pdf_report(text: str, filename: str = "my_report.pdf") -> str:
    from fpdf import FPDF
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 10, line)
        file_path = f"/content/sample_data/{filename}"
        pdf.output(file_path)
        return file_path
    except Exception as e:
        return f"PDF generation failed: {str(e)}"

# 5. LLM

llm = ChatOpenAI(model="gpt-4o-mini")

# 6. Analyst Agent

analyst_llm = llm.bind_tools([get_schema])
analyst_system_message = [SystemMessage(content="""You are a data analyst. Start by understanding the database schema using tools.
Then ask at least 10 insightful questions in a single response.""")]

class AnalystState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def analyst(state: AnalystState) -> AnalystState:
    response = analyst_llm.invoke(analyst_system_message + state["messages"])
    return {"messages": [response]}

analyst_graph = StateGraph(AnalystState)
analyst_graph.add_node("analyst", analyst)
analyst_graph.add_node("tools", ToolNode([get_schema]))
analyst_graph.add_edge(START, "analyst")
analyst_graph.add_conditional_edges("analyst", tools_condition)
analyst_graph.add_edge("tools", "analyst")
analyst_app = analyst_graph.compile()

# 7. Expert Agent

expert_llm = llm.bind_tools([get_schema, execute_sql])
expert_system_message = [SystemMessage(content="""You are a data expert. Use tools to answer the analyst's questions by querying the database.""")]

class ExpertState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def expert(state: ExpertState) -> ExpertState:
    response = expert_llm.invoke(expert_system_message + state["messages"])
    return {"messages": [response]}

expert_graph = StateGraph(ExpertState)
expert_graph.add_node("expert", expert)
expert_graph.add_node("tools", ToolNode([get_schema, execute_sql]))
expert_graph.add_edge(START, "expert")
expert_graph.add_conditional_edges("expert", tools_condition)
expert_graph.add_edge("tools", "expert")
expert_app = expert_graph.compile()

# 8. Reviewer Agent

reviewer_llm = llm.bind_tools([generate_pdf_report])
reviewer_system_message = [SystemMessage(content="""You are an expert reviewer tasked with summarizing detailed database analysis reports.
Produce a concise and clear summary in exactly eight lines with 2 actionable insights at the end.""")]

class ReviewerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def reviewer(state: ReviewerState) -> ReviewerState:
    response = reviewer_llm.invoke(reviewer_system_message + state["messages"])
    return {"messages": [response]}

reviewer_graph = StateGraph(ReviewerState)
reviewer_graph.add_node("reviewer", reviewer)
reviewer_graph.add_node("tools", ToolNode([generate_pdf_report]))
reviewer_graph.add_edge(START, "reviewer")
reviewer_graph.add_conditional_edges("reviewer", tools_condition)
reviewer_graph.add_edge("tools", "reviewer")
reviewer_app = reviewer_graph.compile()

# 9. Supervisor Agent

class AgentSelector(BaseModel):
    next_node: Literal["analyst", "expert", "reviewer", "END"] = Field(
        description="Route to the available agent if needed, else route to END"
    )

agent_selector_llm = llm.with_structured_output(AgentSelector)
supervisor_system_message = [SystemMessage(content="""You are a supervisor agent in charge of orchestrating [analyst, expert, reviewer].""")]

class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_node: Literal["analyst", "expert", "reviewer", "END"]

def supervisor(state: SupervisorState) -> SupervisorState:
    response = agent_selector_llm.invoke(supervisor_system_message + state["messages"])
    return {"messages": [AIMessage(content=f"Routing to: {response.next_node}")], "next_node": response.next_node}

def route_from_supervisor(state: SupervisorState) -> Literal["analyst", "expert", "reviewer", "__end__"]:
    next_node = state.get("next_node")
    if next_node == "END":
        return "__end__"
    else:
        return next_node

graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("analyst", analyst_app)
graph.add_node("expert", expert_app)
graph.add_node("reviewer", reviewer_app)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor)
graph.add_edge("analyst", "supervisor")
graph.add_edge("expert", "supervisor")
graph.add_edge("reviewer", "supervisor")
final_app = graph.compile()

# 10. Run Final App

inputs = {"messages": [HumanMessage(content="Generate a summary report based on tables in my database.")]}
for output in final_app.stream(inputs):
    for key, value in output.items():
        print("=" * 50)
        print(f"📍 Node: {key}")
        print("-" * 50)
        for msg in value["messages"]:
            print(f"{msg.type.upper()}: {msg.content}\n")
        print("=" * 50)