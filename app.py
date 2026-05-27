"""
Streamlit UI for Multi-Agent AI System for Automated Database Insights
"""

import os
import streamlit as st
from agents import init_db, set_connection, build_pipeline, run_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent DB Insights",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00E5FF;
        letter-spacing: -1px;
    }
    .subtitle {
        color: #90A4AE;
        font-size: 0.95rem;
        margin-top: -10px;
        margin-bottom: 24px;
    }
    .node-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'Space Mono', monospace;
        margin-bottom: 6px;
    }
    .badge-supervisor { background: #1A237E; color: #82B1FF; }
    .badge-analyst    { background: #1B5E20; color: #69F0AE; }
    .badge-expert     { background: #4A148C; color: #EA80FC; }
    .badge-reviewer   { background: #BF360C; color: #FFAB40; }
    .stTextArea textarea {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🤖 Multi-Agent DB Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by LangGraph · GPT-4o-mini · SQLite</div>', unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — API key + DB info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored. It's used only for this session.",
    )
    st.caption("🔒 Key is only held in memory for this session.")

    st.divider()
    st.header("🗄️ Database")
    st.markdown("""
**Tables:**
- `users` — 15 sample users
- `orders` — 15 sample orders

**Schema:**
```
users(id, name, email, signup_date)
orders(id, user_id, amount, status, order_date)
```
    """)

    st.divider()
    st.header("🧠 Agent Pipeline")
    st.markdown("""
```
Supervisor
  ├─ Analyst    (asks questions)
  ├─ Expert     (queries DB)
  └─ Reviewer   (writes report + PDF)
```
    """)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "What would you like to know?",
        value="Generate a summary report based on the tables in my database.",
        height=100,
    )

with col2:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")
    clear_btn = st.button("🗑️ Clear Output", use_container_width=True)

if clear_btn:
    st.session_state.pop("results", None)
    st.rerun()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
BADGE = {
    "supervisor": ("badge-supervisor", "🧭 SUPERVISOR"),
    "analyst":    ("badge-analyst",    "📊 ANALYST"),
    "expert":     ("badge-expert",     "🔬 EXPERT"),
    "reviewer":   ("badge-reviewer",   "📝 REVIEWER"),
}

if run_btn:
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not prompt.strip():
        st.error("⚠️ Please enter a prompt.")
        st.stop()

    # Init DB
    conn = init_db()
    set_connection(conn)

    pipeline = build_pipeline(api_key)

    st.divider()
    st.subheader("📡 Live Agent Output")

    results = []
    pdf_path = None
    output_container = st.container()

    with output_container:
        with st.spinner("Agents are working..."):
            for node, messages in run_pipeline(pipeline, prompt):
                cls, label = BADGE.get(node, ("badge-supervisor", node.upper()))
                st.markdown(f'<span class="node-badge {cls}">{label}</span>', unsafe_allow_html=True)

                for msg in messages:
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    if not content:
                        continue

                    msg_type = getattr(msg, "type", "ai")

                    if msg_type == "human":
                        with st.chat_message("user"):
                            st.write(content)
                    else:
                        with st.chat_message("assistant"):
                            st.write(content)

                    results.append((node, msg_type, content))

                    # Check if a PDF was generated
                    if "/tmp/" in content and content.endswith(".pdf"):
                        pdf_path = content.strip()

                st.divider()

    # PDF download
    if pdf_path and os.path.exists(pdf_path):
        st.success("✅ PDF report generated!")
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
                use_container_width=True,
            )

    st.session_state["results"] = results

elif "results" in st.session_state:
    st.divider()
    st.subheader("📡 Previous Output")
    for node, msg_type, content in st.session_state["results"]:
        cls, label = BADGE.get(node, ("badge-supervisor", node.upper()))
        st.markdown(f'<span class="node-badge {cls}">{label}</span>', unsafe_allow_html=True)
        role = "user" if msg_type == "human" else "assistant"
        with st.chat_message(role):
            st.write(content)
