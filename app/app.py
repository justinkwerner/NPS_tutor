"""Streamlit UI for the NPS SS3861 LLM Tutor.

Run from the project root:
    streamlit run app/app.py
"""

import os
import re
import sys
import tempfile
from pathlib import Path

# Make project root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from SQL_db.database import (
    delete_conversation,
    get_conversation_history,
    get_or_create_user,
    get_user_conversations,
    init_db,
    save_message,
    start_conversation,
)

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = os.getenv("SQLITE_DB_PATH", str(Path(__file__).parent.parent / "tutor.db"))

# ── Eval model registry (temporary — remove after model selection) ─────────────

_GROQ_URL  = os.getenv("GROQ_BASE_URL", "https://api.groq.com")
_GROQ_KEY  = os.getenv("GROQ_API_KEY", "")
_NIM_URL  = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
_NIM_KEY  = os.getenv("NIM_API_KEY", "")
_NIM_MODEL = os.getenv("NIM_MODEL", "nvidia/nemotron-3-super")

EVAL_MODELS = {
    f"NVIDIA — {_NIM_MODEL}": {
        "url": _NIM_URL, "key": _NIM_KEY, "model": _NIM_MODEL,
    },
    "Qwen3-32B (Groq)": {
        "url": _GROQ_URL, "key": _GROQ_KEY, "model": "qwen/qwen3-32b",
    },
    "Llama-3.3-70B (Groq)": {
        "url": _GROQ_URL, "key": _GROQ_KEY, "model": "llama-3.3-70b-versatile",
    },
}

st.set_page_config(
    page_title="NPS SS3861 Tutor",
    page_icon="🛰️",
    layout="wide",
)

# ── DB init ───────────────────────────────────────────────────────────────────

init_db(DB_PATH)

# ── Vectorstore startup check ─────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent
_CHROMA_PATH = Path(os.getenv("CHROMA_DB_PATH", str(_PROJECT_ROOT / "databases" / "chroma_unified")))
_SATOPS_CHROMA_PATH = Path(os.getenv("SATOPS_CHROMA_DB_PATH", str(_PROJECT_ROOT / "databases" / "chroma_satops")))

_VECTORSTORE_OPTIONS = {
    "Course materials": {"output_dir": str(_CHROMA_PATH), "collection_name": "unified_course"},
    "SATOPS": {"output_dir": str(_SATOPS_CHROMA_PATH), "collection_name": "satops_course"},
}

_DOC_TYPE_OPTIONS = {
    "Course materials": ["code", "upload", "lecture", "lab", "manual"],
    "SATOPS": ["satops", "satops_commands", "satops_manual"],
}


@st.cache_resource(show_spinner=False)
def _ensure_vectorstore():
    """Build the vectorstore on first launch if it is empty or missing.

    Cached so the check runs once per process, not on every Streamlit rerun.
    """
    sqlite_file = _CHROMA_PATH / "chroma.sqlite3"
    if sqlite_file.exists() and sqlite_file.stat().st_size > 0:
        return  # already built

    with st.spinner("First launch: building vectorstore from course materials — this may take a few minutes..."):
        from scripts.build_vectorstore import build_vectorstore  # noqa: PLC0415
        build_vectorstore(data_dir=str(_PROJECT_ROOT), output_dir=str(_CHROMA_PATH))


_ensure_vectorstore()

# ── Session state defaults ────────────────────────────────────────────────────

for key, default in {
    "user": None,           # dict from DB
    "conversation_id": None,
    "messages": [],         # list of {role, content} for display
    "selected_model": list(EVAL_MODELS.keys())[0],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Markdown / LaTeX rendering ───────────────────────────────────────────────

def _md(text: str):
    """Render text with LaTeX and HTML support.

    Converts \[...\] → $$...$$ and \(...\) → $...$ so Streamlit's KaTeX
    renderer picks them up, then renders with HTML enabled for <sub>/<sup>.
    """
    text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'【[^】]*】', '', text)
    st.markdown(text, unsafe_allow_html=True)


# ── Login page ────────────────────────────────────────────────────────────────

def show_login():
    st.title("NPS SS3861 Tutor")
    st.markdown("Spacecraft Payload Communications & Data Handling")
    st.divider()

    with st.form("login_form"):
        name = st.text_input("Name", placeholder="Enter your name")
        role = st.selectbox("Role", ["student", "instructor"])
        submitted = st.form_submit_button("Enter")

    if submitted:
        if not name.strip():
            st.error("Please enter your name.")
            return
        user = get_or_create_user(DB_PATH, name.strip(), role)
        st.session_state.user = user
        # Start a fresh conversation for this session.
        conv_id = start_conversation(DB_PATH, user["id"])
        st.session_state.conversation_id = conv_id
        st.session_state.messages = []
        st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────

def show_sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"**{user['name']}** · {user['role'].capitalize()}")
        st.divider()

        if st.button("New conversation", use_container_width=True):
            conv_id = start_conversation(DB_PATH, user["id"])
            st.session_state.conversation_id = conv_id
            st.session_state.messages = []
            st.rerun()

        st.markdown("#### Past conversations")
        conversations = get_user_conversations(DB_PATH, user["id"])
        for conv in conversations:
            first_msgs = get_conversation_history(DB_PATH, conv["id"], limit=50)
            user_msgs = [m for m in first_msgs if m["role"] == "user"]
            label = user_msgs[0]["content"][:60] + "…" if user_msgs else conv["started_at"]

            is_active = conv["id"] == st.session_state.conversation_id
            btn_type = "primary" if is_active else "secondary"
            col_label, col_del = st.columns([5, 1])
            with col_label:
                if st.button(label, key=f"conv_{conv['id']}", use_container_width=True, type=btn_type):
                    st.session_state.conversation_id = conv["id"]
                    st.session_state.messages = get_conversation_history(DB_PATH, conv["id"])
                    st.rerun()
            with col_del:
                if st.button("×", key=f"del_{conv['id']}", use_container_width=True):
                    delete_conversation(DB_PATH, conv["id"])
                    if is_active:
                        new_id = start_conversation(DB_PATH, user["id"])
                        st.session_state.conversation_id = new_id
                        st.session_state.messages = []
                    st.rerun()

        # ── Eval model selector (instructor only) ────────────────────────────
        if user["role"] == "instructor":
            st.divider()
            st.markdown("#### Model selector")
            selected = st.selectbox(
                "Active model",
                list(EVAL_MODELS.keys()),
                index=list(EVAL_MODELS.keys()).index(st.session_state.selected_model),
                label_visibility="collapsed",
            )
            if selected != st.session_state.selected_model:
                st.session_state.selected_model = selected
                st.rerun()

        # ── Instructor upload panel ───────────────────────────────────────────
        if user["role"] == "instructor":
            st.divider()
            st.markdown("#### Upload course material")
            vs_choice = st.selectbox(
                "Vectorstore",
                list(_VECTORSTORE_OPTIONS.keys()),
            )
            uploaded_file = st.file_uploader(
                "PDF, PPTX, DOCX, MD, or PY",
                type=["pdf", "pptx", "docx", "md", "py"],
                label_visibility="collapsed",
            )
            _type_opts = _DOC_TYPE_OPTIONS[vs_choice]
            _default_type = (
                "code"
                if uploaded_file and Path(uploaded_file.name).suffix.lower() == ".py"
                else _type_opts[0]
            )
            doc_type = st.selectbox(
                "Document type",
                _type_opts,
                index=_type_opts.index(_default_type),
            )
            if st.button("Ingest file", use_container_width=True, disabled=uploaded_file is None):
                vs_cfg = _VECTORSTORE_OPTIONS[vs_choice]
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                try:
                    with st.spinner(f"Ingesting {uploaded_file.name} → {vs_choice}…"):
                        from scripts.build_vectorstore import ingest_file_to_vectorstore  # noqa: PLC0415
                        n_added = ingest_file_to_vectorstore(
                            file_path=tmp_path,
                            doc_type=doc_type,
                            output_dir=vs_cfg["output_dir"],
                            collection_name=vs_cfg["collection_name"],
                            original_filename=uploaded_file.name,
                        )
                    if n_added:
                        st.success(f"Added {n_added} chunks from {uploaded_file.name} to {vs_choice}")
                    else:
                        st.info("File already in vectorstore — no new chunks added.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

        st.divider()
        if st.button("Log out", use_container_width=True):
            for key in ["user", "conversation_id", "messages"]:
                st.session_state[key] = None if key != "messages" else []
            st.rerun()

# ── Chat page ─────────────────────────────────────────────────────────────────

def show_chat():
    show_sidebar()

    st.title("Space Systems Tutor")
    active_cfg = EVAL_MODELS[st.session_state.selected_model]
    st.caption(f"Model: `{active_cfg['model']}` — {active_cfg['url']}")

    # Load history from DB if messages list is empty (e.g. after page refresh).
    if not st.session_state.messages and st.session_state.conversation_id:
        st.session_state.messages = get_conversation_history(
            DB_PATH, st.session_state.conversation_id
        )

    # Render conversation history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            _md(msg["content"])

    # Chat input.
    user_input = st.chat_input("Ask a question about the course...")
    if not user_input:
        return

    # Display and persist user message immediately.
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message(DB_PATH, st.session_state.conversation_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the agent and stream the response.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from agent.agent import run_agent  # lazy import to avoid cold-start cost
                response = run_agent(
                    query=user_input,
                    user_context={
                        "name": st.session_state.user["name"],
                        "role": st.session_state.user["role"],
                    },
                    conversation_history=st.session_state.messages[:-1],  # exclude current msg
                    model_config=EVAL_MODELS[st.session_state.selected_model],
                )
            except Exception as e:
                response = f"⚠️ Agent error: {e}"
        _md(response)

    # Persist and record assistant response.
    save_message(DB_PATH, st.session_state.conversation_id, "assistant", response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# ── Router ────────────────────────────────────────────────────────────────────

if st.session_state.user is None:
    show_login()
else:
    show_chat()
