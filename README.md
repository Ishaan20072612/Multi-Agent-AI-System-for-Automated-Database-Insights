# 🤖 Multi-Agent AI System for Automated Database Insights

A production-ready multi-agent pipeline built with **LangGraph** and **GPT-4o-mini** that autonomously analyzes a SQLite database, answers data questions, and generates a downloadable PDF report — all orchestrated by a Supervisor agent.

---

## 🏗️ Architecture

```
User Prompt
     │
     ▼
┌─────────────┐
│  Supervisor  │  ← Routes between agents
└──────┬──────┘
       │
  ┌────┴────────────────────┐
  │          │              │
  ▼          ▼              ▼
Analyst   Expert        Reviewer
  │          │              │
Asks 10+  Queries DB    Summarizes +
questions  via SQL      generates PDF
```

| Agent | Role | Tools |
|-------|------|-------|
| **Supervisor** | Orchestrates the pipeline | Structured routing |
| **Analyst** | Understands schema, asks insightful questions | `get_schema` |
| **Expert** | Answers questions by running SQL queries | `get_schema`, `execute_sql` |
| **Reviewer** | Summarizes findings, creates PDF report | `generate_pdf_report` |

---

## ✨ Features

- 🔄 **Fully autonomous** multi-agent loop with LangGraph
- 🗄️ **SQLite** database with users & orders tables
- 📊 **Analyst agent** generates 10+ data-driven questions
- 🔍 **Expert agent** answers via live SQL queries
- 📝 **Reviewer agent** produces an 8-line summary + 2 actionable insights
- 📥 **PDF report** download via Streamlit UI
- 🔒 API key entered at runtime — never stored

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multi-agent-db-insights.streamlit.app)

> Replace the link above after deploying to Streamlit Cloud.

---

## 🛠️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/Ishaan20072612/Multi-Agent-AI-System-for-Automated-Database-Insights.git
cd Multi-Agent-AI-System-for-Automated-Database-Insights
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

### 5. Open in browser
Go to `http://localhost:8501` and enter your OpenAI API key in the sidebar.

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. **Fork or push** this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set **Main file path** to `app.py`
4. Click **Deploy** — done in ~2 minutes!
5. Share the public URL with anyone

> ⚠️ Users will need to enter their own OpenAI API key in the sidebar. Your key is never embedded in the code.

---

## 📁 Project Structure

```
├── app.py              # Streamlit UI
├── agents.py           # Core LangGraph pipeline & agent logic
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🧪 Sample Database

**users** (15 rows): id, name, email, signup_date  
**orders** (15 rows): id, user_id, amount, status, order_date

Sample questions the Analyst asks:
- Which users have placed the most orders?
- What is the average order value by month?
- What percentage of orders are pending vs completed vs cancelled?
- Which users signed up but never placed an order?

---

## 🔧 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `langchain` | 0.3.x | LLM framework |
| `langchain-openai` | 0.3.x | OpenAI integration |
| `langgraph` | 0.4.x | Agent orchestration |
| `openai` | 1.x | GPT-4o-mini |
| `fpdf` | 1.7.x | PDF generation |
| `streamlit` | 1.x | Web UI |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Ishaan Chowdhury** · [@Ishaan20072612](https://github.com/Ishaan20072612)
