"""LangGraph ReAct agent for NPS SS3861 course tutoring.

Phase 1 checkpoint — run end-to-end test:
    python agent/agent.py
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path so db.database is importable regardless
# of the working directory the script is launched from.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports for various embedding providers and LLMs. The agent can run with just the local HuggingFaceEmbeddings and a compatible NIM endpoint, but will use NVIDIA-hosted embeddings if credentials are provided.
from SQL_db.database import get_conversation_history, save_message
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

# Optional NVIDIA-hosted embeddings support
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings  # type: ignore
except Exception:
    NVIDIAEmbeddings = None

# Optional cross-encoder re-ranker support
try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None
from langgraph.prebuilt import create_react_agent

# Optional OpenAI-compatible (NIM) support
try:
    from openai import OpenAI as OpenAIClient  # type: ignore
except Exception:
    OpenAIClient = None

try:
    from langchain_openai import ChatOpenAI as LangOpenAI  # type: ignore
except Exception:
    LangOpenAI = None

try:
    from langchain_groq import ChatGroq  # type: ignore
except Exception:
    ChatGroq = None

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Embeddings ────────────────────────────────────────────────────────────────
# Priority: 1) NVIDIA hosted embeddings (nvapi- key), 2) HF endpoint server,
#           3) local all-MiniLM-L6-v2 fallback.

EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embed-v1")
EMBED_SERVER_URL = os.getenv(
    "EMBEDDING_SERVER_URL",
    "http://trac-malenia.ern.nps.edu:8080/gpu5/embed",
)
EMBED_FALLBACK = "all-MiniLM-L6-v2"

# ALTERNATE_EMBEDDING_MODEL_KEY controls embeddings separately from NIM_API_KEY (which is for the LLM).
# Leave ALTERNATE_EMBEDDING_MODEL_KEY unset to skip NVIDIA embeddings and use the local fallback instead.
_nvidia_embed_key = os.getenv("ALTERNATE_EMBEDDING_MODEL_KEY")
_nvidia_url = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

if NVIDIAEmbeddings is not None and _nvidia_embed_key:
    try:
        hfe = NVIDIAEmbeddings(
            model=EMBED_MODEL,
            api_key=_nvidia_embed_key,
            base_url=_nvidia_url,
        )
        hfe.embed_query("test")  # probe
        logger.info("Embedding function set from NVIDIA API: %s", EMBED_MODEL)
    except Exception as e:
        logger.warning("NVIDIA embeddings failed: %s — falling back to local", e)
        hfe = None
else:
    hfe = None

if hfe is None:
    try:
        _candidate = HuggingFaceEndpointEmbeddings(model=EMBED_SERVER_URL)
        _candidate.embed_query("test")  # probe
        hfe = _candidate
        logger.info("Embedding function set from HF endpoint server")
    except Exception as e:
        logger.info("HF endpoint unreachable: %s — using local embeddings", e)
        hfe = HuggingFaceEmbeddings(
            model_name=EMBED_FALLBACK,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
        logger.info("Embedding function set locally with %s", EMBED_FALLBACK)

# ── Vectorstore ─────────────────────────────────

UNIFIED_CHROMA_DIR = os.getenv("CHROMA_DB_PATH", "./databases/chroma_unified")
SATOPS_CHROMA_DIR = os.getenv("SATOPS_CHROMA_DB_PATH", "./databases/chroma_satops")

# Vectorstore that holds all course materials (lectures, labs, manual) with metadata for filtering.
unified_vectorstore = Chroma(
    collection_name="unified_course",
    embedding_function=hfe,
    persist_directory=UNIFIED_CHROMA_DIR,
)

# Vectorstore for satellite operations (MAX GDS commands, SATOPS manuals).
satops_vectorstore = Chroma(
    collection_name="satops_course",
    embedding_function=hfe,
    persist_directory=SATOPS_CHROMA_DIR,
)


# ── Re-ranker ────────────────────────────────────────────────────────────────
# Set RERANKER_MODEL in .env to enable cross-encoder re-ranking, e.g.:
#   RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# Leave unset (or set to empty string) to disable re-ranking.

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "")

_reranker: "CrossEncoder | None" = None
if RERANKER_MODEL and CrossEncoder is not None:
    try:
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Re-ranker loaded: %s", RERANKER_MODEL)
    except Exception as e:
        logger.warning("Re-ranker failed to load (%s): %s — skipping re-ranking", RERANKER_MODEL, e)
elif RERANKER_MODEL and CrossEncoder is None:
    logger.warning("RERANKER_MODEL set but sentence_transformers not installed — skipping re-ranking")


def _rerank(query: str, candidates: list, top_n: int) -> list:
    """Re-rank (doc, score) candidates with the cross-encoder and return top_n.

    Falls back to the original order if the re-ranker is unavailable.
    """
    if _reranker is None or not candidates:
        return candidates[:top_n]
    pairs = [[query, doc.page_content] for doc, _ in candidates]
    ce_scores = _reranker.predict(pairs)
    reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    logger.info("  Re-ranker applied — top CE score: %.4f", reranked[0][1] if reranked else 0)
    return [candidate for candidate, _ in reranked[:top_n]]


def _strip_repetition(text: str, max_repeats: int = 3) -> str:
    """Truncate text if a sentence repeats more than max_repeats times.

    Guards against LLM repetition loops where a model gets stuck cycling
    the same sentence indefinitely (common at temperature=0).
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen: dict[str, int] = {}
    output: list[str] = []
    for sentence in sentences:
        key = sentence.strip().lower()
        if len(key) > 20:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > max_repeats:
                logger.warning("Repetition loop detected — truncating response.")
                break
        output.append(sentence)
    return " ".join(output)


def satops_search(
    query: str,
    doc_type: str | None = None,
    k: int = 5,
    score_threshold: float = 0.15,
):
    """Search the SATOPS vectorstore.

    Args:
        query:           Search string.
        doc_type:        Optional doc_type filter (e.g. "satops_commands").
        k:               Number of results to request.
        score_threshold: Minimum relevance score.

    Returns:
        List of (Document, score) tuples above threshold.
    """
    filter_dict = {"doc_type": doc_type} if doc_type else None
    results = satops_vectorstore.similarity_search_with_relevance_scores(
        query, k=k, filter=filter_dict
    )
    return [(doc, score) for doc, score in results if score >= score_threshold]


def filtered_search(
    query: str,
    doc_type: str | None = None,
    lecture_number: int | None = None,
    k: int = 5,
    score_threshold: float = 0.15,
):
    """Search the unified vectorstore with optional doc_type and lecture_number filters.

    Args:
        query:            Search string.
        doc_type:         "lab", "lecture", "manual", or None for all.
        lecture_number:   Integer lecture number to restrict results to a specific
                          lecture's pptx slides. Only applies when doc_type="lecture".
        k:                Number of results to request from the store.
        score_threshold:  Minimum relevance score (0-1). Filters noisy results.

    Returns:
        List of (Document, score) tuples above threshold.
    """
    if lecture_number is not None and doc_type == "lecture":
        filter_dict = {"$and": [{"doc_type": "lecture"}, {"lecture_number": lecture_number}]}
    elif doc_type:
        filter_dict = {"doc_type": doc_type}
    else:
        filter_dict = None
    results = unified_vectorstore.similarity_search_with_relevance_scores(
        query, k=k, filter=filter_dict
    )
    return [(doc, score) for doc, score in results if score >= score_threshold]


# ── Query rewriting ─────────────────────────────────

_REWRITE_SYSTEM = (
    "You are a helpful AI assistant that specializes in rewriting complex user "
    "queries. Rewrite the following query by expanding it into multiple simpler, "
    "more concise queries. Start each new query with '[START]'. Do not provide "
    "any additional responses besides the rewritten queries."
)

_rewrite_chain = None  # built lazily on first use


def _get_rewrite_chain():
    """Build the rewrite chain on first call using the configured NIM endpoint."""
    global _rewrite_chain
    if _rewrite_chain is not None:
        return _rewrite_chain

    if LangOpenAI is None:
        raise RuntimeError("langchain_openai is required for query rewriting.")

    nim_url = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nim_key = os.getenv("NIM_API_KEY")
    nim_model = os.getenv("NIM_MODEL", "nvidia/nemotron-3-super")

    llm = LangOpenAI(
        model=nim_model,
        base_url=nim_url,
        api_key=nim_key,
        temperature=0,
        max_tokens=512,
    )
    rewrite_prompt = ChatPromptTemplate([
        ("system", _REWRITE_SYSTEM),
        ("human", "{prompt}"),
    ])
    _rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    return _rewrite_chain


def _rewrite_query(original: str) -> list[str]:
    """Expand one query into multiple sub-queries via the LLM (cell 31)."""
    raw = _get_rewrite_chain().invoke({"prompt": original})
    sub_queries = [q.strip() for q in raw.split("[START]") if q.strip()]
    return sub_queries if sub_queries else [original]


# ── RAG retriever runnables ─────────────────────


def semantic_retrieve_lecture_w_scores(state: dict) -> dict:
    logger.info("NODE: Semantic Retriever - Lectures")
    query = state["question"]
    k = state.get("k", 5)
    lecture_number = state.get("lecture_number", None)
    docs_with_scores = filtered_search(query, doc_type="lecture", lecture_number=lecture_number, k=k)
    logger.info(f"  Retrieved {len(docs_with_scores)} documents")
    return {"documents": docs_with_scores}


def semantic_retrieve_lab_w_scores(state: dict) -> dict:
    logger.info("NODE: Semantic Retriever - Labs")
    query = state["question"]
    k = state.get("k", 5)
    docs_with_scores = filtered_search(query, doc_type="lab", k=k)
    logger.info(f"  Retrieved {len(docs_with_scores)} documents")
    return {"documents": docs_with_scores}


def semantic_retrieve_manual_w_scores(state: dict) -> dict:
    logger.info("NODE: Semantic Retriever - Manual")
    query = state["question"]
    k = state.get("k", 5)
    docs_with_scores = filtered_search(query, doc_type="manual", k=k)
    logger.info(f"  Retrieved {len(docs_with_scores)} documents")
    return {"documents": docs_with_scores}


semantic_retriever_lecture = RunnableLambda(
    semantic_retrieve_lecture_w_scores, name="semantic_retriever_lecture"
)
semantic_retriever_lab = RunnableLambda(
    semantic_retrieve_lab_w_scores, name="semantic_retriever_lab"
)
semantic_retriever_manual = RunnableLambda(
    semantic_retrieve_manual_w_scores, name="semantic_retriever_manual"
)

# ── Agent tools ─────────────────────────────────


@tool(response_format="content_and_artifact")
def retrieve_context_lectures(query: str, lecture_number: int | None = None):
    """Retrieve information from the lecture slides to help answer a query.

    Use this tool for conceptual questions about electronics, circuits,
    active/passive components, spacecraft payloads, solar cells, signal theory,
    or any course topic not specific to labs or oscilloscope operation.

    When the student asks about a specific lecture (e.g. "lecture 2"), pass the
    lecture number as an integer so results are restricted to that lecture's slides.
    Leave lecture_number as None for general topic searches across all lectures.
    """
    results = semantic_retriever_lecture.invoke({"question": query, "lecture_number": lecture_number})
    docs_with_scores = results.get("documents", [])
    if not docs_with_scores:
        return "No relevant documents found for the query.", []
    docs_with_scores = _rerank(query, docs_with_scores, top_n=5)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')} | "
        f"Topic: {doc.metadata.get('topic', 'N/A')} | "
        f"Lecture: {doc.metadata.get('lecture_number', 'N/A')}\n"
        f"Relevance: {score:.2f}\n"
        f"Content: {doc.page_content}"
        for doc, score in docs_with_scores
    )
    return serialized, [doc for doc, _ in docs_with_scores]


@tool(response_format="content_and_artifact")
def retrieve_context_labs(query: str):
    """Retrieve information from the lab documents to help answer a query.

    Use this tool when you need to find information about course labs.
    """
    results = semantic_retriever_lab.invoke({"question": query})
    docs_with_scores = results.get("documents", [])
    if not docs_with_scores:
        return "No relevant documents found for the query.", []
    docs_with_scores = _rerank(query, docs_with_scores, top_n=5)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')} | "
        f"Topic: {doc.metadata.get('topic', 'N/A')} | "
        f"Lab: {doc.metadata.get('lab_number', 'N/A')}\n"
        f"Relevance: {score:.2f}\n"
        f"Content: {doc.page_content}"
        for doc, score in docs_with_scores
    )
    return serialized, [doc for doc, _ in docs_with_scores]


@tool(response_format="content_and_artifact")
def retrieve_context_manuals(query: str):
    """Retrieve information from the oscilloscope manual to help answer a query.

    Use this tool ONLY for questions about how to operate or use the oscilloscope
    (buttons, display settings, probes, measurements). Do NOT use for general
    electronics or circuit theory questions.
    """
    results = semantic_retriever_manual.invoke({"question": query})
    docs_with_scores = results.get("documents", [])
    if not docs_with_scores:
        return "No relevant documents found for the query.", []
    docs_with_scores = _rerank(query, docs_with_scores, top_n=5)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')} | "
        f"Topic: {doc.metadata.get('topic', 'N/A')}\n"
        f"Relevance: {score:.2f}\n"
        f"Content: {doc.page_content}"
        for doc, score in docs_with_scores
    )
    return serialized, [doc for doc, _ in docs_with_scores]


@tool(response_format="content_and_artifact")
def rewrite_and_retrieve(query: str):
    """Rewrite a query into sub-queries and retrieve across all course materials.

    Use this tool when a question is broad, ambiguous, or spans multiple topics.
    It expands the query into focused sub-queries, searches lectures, labs, and
    the manual for each, deduplicates results, and returns combined context.
    Prefer the focused tools (retrieve_context_lectures, retrieve_context_labs,
    retrieve_context_manuals) for narrow, single-topic questions.
    """
    logger.info("NODE: Query Rewriter")
    sub_queries = _rewrite_query(query)
    logger.info("  Sub-queries: %s", sub_queries)

    seen: set[str] = set()
    docs_with_scores: list = []

    for q in sub_queries:
        for doc, score in filtered_search(q, doc_type=None):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs_with_scores.append((doc, score))

    if not docs_with_scores:
        return "No relevant documents found for the query.", []

    docs_with_scores = _rerank(query, docs_with_scores, top_n=5)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')} | "
        f"Type: {doc.metadata.get('doc_type', 'N/A')} | "
        f"Topic: {doc.metadata.get('topic', 'N/A')}\n"
        f"Relevance: {score:.2f}\n"
        f"Content: {doc.page_content}"
        for doc, score in docs_with_scores
    )
    return serialized, [doc for doc, _ in docs_with_scores]


@tool(response_format="content_and_artifact")
def retrieve_context_satops(query: str, doc_type: str | None = None):
    """Retrieve information from the SATOPS / MAX GDS knowledge base.

    Use this tool for questions about satellite operations, MAX GDS commands,
    telemetry windows, file upload/download procedures, pass scheduling,
    or any SATOPS-specific topic.

    Optionally pass doc_type (e.g. "satops_commands") to restrict results to
    a specific category of SATOPS content.
    """
    results = satops_search(query, doc_type=doc_type)
    if not results:
        return "No relevant SATOPS documents found for the query.", []
    results = _rerank(query, results, top_n=5)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('filename', 'Unknown')} | "
        f"Type: {doc.metadata.get('doc_type', 'N/A')}\n"
        f"Relevance: {score:.2f}\n"
        f"Content: {doc.page_content}"
        for doc, score in results
    )
    return serialized, [doc for doc, _ in results]


@tool(response_format="content_and_artifact")
def retrieve_context_code(query: str):
    """Retrieve uploaded Python code from the course vectorstore.

    ALWAYS call this tool first when a student asks about a Python file, editing
    code, how to implement something in a script, or references a file by name
    (e.g. main.py, payload_class.py). Instructors upload lab starter code here.
    Do NOT skip this tool and rely on general knowledge — the actual code content
    may be available and should be used to give accurate, specific guidance.
    """
    results = filtered_search(query, doc_type="code", k=5, score_threshold=0.0)
    if not results:
        return "No relevant code examples found for the query.", []
    results = _rerank(query, results, top_n=5)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('filename', 'Unknown')}\n"
        f"Relevance: {score:.2f}\n"
        f"Content: {doc.page_content}"
        for doc, score in results
    )
    return serialized, [doc for doc, _ in results]


@tool
def python_calculator(code: str) -> str:
    """Execute a Python expression to perform calculations.

    Use this for any numeric computation, unit conversion, or binary/hex math.
    Example: 'bin(31)', '(128/255)*5', 'int("00011111", 2)'
    """
    try:
        result = eval(
            code,
            {"__builtins__": {}},
            {"bin": bin, "hex": hex, "int": int, "round": round, "abs": abs, "pow": pow},
        )
        return str(result)
    except Exception as e:
        return f"Error: {e}"


_websearch = DuckDuckGoSearchResults(output_format="json", num_results=3)


@tool
def web_search(query: str) -> str:
    """Search the web for additional information about the query.

    Use this tool when there are no relevant documents found for the query.
    """
    results = _websearch.invoke(query)
    try:
        results_list = json.loads(results)
        return "\n\n".join(
            f"Title: {r['title']}\nSnippet: {r['snippet']}\nLink: {r['link']}"
            for r in results_list
        )
    except Exception:
        return results


rag_tools = [
    rewrite_and_retrieve,
    retrieve_context_code,
    retrieve_context_lectures,
    retrieve_context_labs,
    retrieve_context_manuals,
    retrieve_context_satops,
    web_search,
    python_calculator,
]

# ── System prompt ───────────────────────────────────

_BASE_SYSTEM_PROMPT = (
    "You are a teaching assistant and tutor for students in the Spacecraft Payload "
    "Communications & Data Handling course (SS3861) at Naval Postgraduate School.\n"
    "{user_context_block}"
    "Use prior conversation context (provided in message history) to build on what "
    "the student already knows and avoid repeating explanations they have already received.\n"
    "Use the tools provided to assist students. Try not to directly provide an answer "
    "to their questions; help them learn and build a better understanding of the material.\n"
    "Provide a citation for where you found the correct information. Give document name "
    "and page or slide numbers as needed.\n"
    "You may need to use more than one tool. Use rewrite_and_retrieve for broad or "
    "multi-topic questions — it expands the query and searches all course materials. "
    "Use retrieve_context_lectures for focused conceptual questions. "
    "When the student references a specific lecture by number (e.g. 'lecture 2', 'from lecture 5'), "
    "you MUST pass that integer as the lecture_number argument to retrieve_context_lectures "
    "so results are restricted to that lecture's slides. Do NOT omit lecture_number when a "
    "specific lecture is mentioned.\n"
    "Use retrieve_context_labs for lab assignments. "
    "Use retrieve_context_manuals for oscilloscope operation only. "
    "Use retrieve_context_satops for satellite operations questions, MAX GDS commands, "
    "telemetry windows, file transfers, pass scheduling, or any SATOPS procedure. "
    "Use retrieve_context_code when a student asks about code, a Python script, or editing/modifying "
    "a file (e.g. 'how do I edit main.py', 'what does this function do', 'help me with the code'). "
    "ALWAYS call retrieve_context_code before giving any code-specific guidance — do not rely on "
    "general knowledge alone when code may have been uploaded. "
    "Use web_search if no relevant information is found in the course materials.\n"
    "Retrieved documents include metadata fields (topic, lab_number, lecture_number) — "
    "use these to give precise citations such as 'Lecture 7, power_systems' or "
    "'Lab 3, comm_protocols'.\n"
    "Do NOT use bracketed citation markers like 【0†L1-L4】. Cite sources inline "
    "using plain text: document name, topic, and page or slide number only."
)

# Allows agent to know if they're speaking to a student or instructor, and personalize the prompt with their name.

def _build_system_prompt(user_context: dict | None) -> str:
    if not user_context:
        return _BASE_SYSTEM_PROMPT.format(user_context_block="")
    name = user_context.get("name", "the student")
    role = user_context.get("role", "student")
    if role == "instructor":
        user_block = (
            f"You are speaking with {name}, an instructor in this course. "
            "Engage at a peer-level technical discussion.\n"
        )
    else:
        user_block = f"You are speaking with {name}, a student in the course.\n"
    return _BASE_SYSTEM_PROMPT.format(user_context_block=user_block)


# ── Agent factory (NPS_tutor.ipynb cell 50) ───────────────────────────────────


def build_agent(user_context: dict | None = None, model_config: dict | None = None):
    """Create and return a LangGraph ReAct agent.

    Args:
        user_context:  Optional dict with 'name' and 'role' keys to personalize
                       the system prompt. Role must be 'student' or 'instructor'.
        model_config:  Optional dict with 'url', 'key', 'model' keys to override
                       the NIM_BASE_URL / NIM_API_KEY / NIM_MODEL env vars.
    Returns:
        Compiled LangGraph agent.
    """
    if model_config:
        nim_url   = model_config["url"]
        nim_key   = model_config["key"]
        nim_model = model_config["model"]
    else:
        nim_url   = os.getenv("NIM_BASE_URL")
        nim_key   = os.getenv("NIM_API_KEY")
        nim_model = os.getenv("NIM_MODEL")

    if not (nim_url and nim_key):
        raise RuntimeError(
            "NIM_BASE_URL and NIM_API_KEY must be set to use an OpenAI-compatible NIM model.\n"
            "Set them in your environment or a .env file (example: NIM_BASE_URL=http://localhost:8000/v1)."
        )

    if "groq.com" in nim_url:
        if ChatGroq is None:
            raise RuntimeError("langchain_groq is required for Groq models. Run: pip install langchain-groq")
        llm = ChatGroq(
            model=nim_model,
            api_key=nim_key,
            temperature=0.1,
            max_tokens=2048,
        )
    else:
        if LangOpenAI is None:
            raise RuntimeError(
                "langchain_openai is not available. Run: pip install langchain-openai"
            )
        llm = LangOpenAI(
            model=nim_model,
            base_url=nim_url,
            api_key=nim_key,
            temperature=0.1,
            max_tokens=2048,
        )
    logger.info("Using NIM/OpenAI-compatible endpoint for LLM: %s", nim_model)
    system_prompt = _build_system_prompt(user_context)
    return create_react_agent(
        llm,
        rag_tools,
        prompt=SystemMessage(content=system_prompt),
    )


def run_agent(
    query: str,
    user_context: dict | None = None,
    conversation_history: list | None = None,
    db_path: str | None = None,
    conversation_id: int | None = None,
    model_config: dict | None = None,
) -> str:
    """Run the agent on a single query.

    Args:
        query:                The student's question.
        user_context:         Optional dict with 'name' and 'role' for system
                              prompt personalization.
        conversation_history: Optional list of prior message dicts
                              [{"role": "user"|"assistant", "content": str}, ...]
                              used when *not* loading history from the DB.
                              Ignored if db_path and conversation_id are given.
        db_path:              Path to the SQLite3 database file. When provided
                              together with conversation_id, history is loaded
                              from the DB and the new exchange is persisted.
        conversation_id:      Active conversation id in the DB.
        model_config:         Optional dict with 'url', 'key', 'model' to
                              override the default NIM endpoint for this call.

    Returns:
        The agent's final response as a string.
    """
    agent = build_agent(user_context, model_config=model_config)

    # Prefer DB history over the caller-supplied list when both are available.
    if db_path and conversation_id:
        history = get_conversation_history(db_path, conversation_id)
    else:
        history = conversation_history or []

    messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=query))

    result = agent.invoke({"messages": messages})
    response = _strip_repetition(result["messages"][-1].content)

    # Persist the new exchange to the DB when connected.
    if db_path and conversation_id:
        save_message(db_path, conversation_id, "user", query)
        save_message(db_path, conversation_id, "assistant", response)

    return response


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_query = "What is the purpose of encoding in digital communications?"
    user_ctx = {"name": "Test Student", "role": "student"}

    print(f"Query: {test_query}\n")
    response = run_agent(test_query, user_context=user_ctx)
    print(f"Response:\n{response}")
