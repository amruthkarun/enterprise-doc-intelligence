import os
import streamlit as st
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from user_auth import get_user_access

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CHROMA_DIRECTORY = os.path.join(BASE_DIR, "..", "chroma_db")
DATA_DIRECTORY = os.path.join(BASE_DIR, "..", "data")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
    )


@st.cache_resource(show_spinner="Connecting to vector store…")
def get_document_store(_embedding_model):
    return Chroma(
        persist_directory=CHROMA_DIRECTORY,
        embedding_function=_embedding_model,
    )


def _indexed_companies(store) -> set:
    try:
        result = store._collection.get(include=["metadatas"])
        return {m.get("company", "") for m in result["metadatas"] if m.get("company")}
    except Exception:
        return set()


def ensure_indexed(store):
    already = _indexed_companies(store)
    new_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for filename in os.listdir(DATA_DIRECTORY):
        if not filename.lower().endswith(".pdf"):
            continue
        company_name = os.path.splitext(filename)[0].strip()
        if company_name in already:
            continue
        file_path = os.path.join(DATA_DIRECTORY, filename)
        loader = PyPDFLoader(file_path)
        for page_num, page in enumerate(loader.load(), start=1):
            for chunk in splitter.split_documents([page]):
                new_docs.append(Document(
                    page_content=chunk.page_content,
                    metadata={
                        "source": file_path,
                        "company": company_name,
                        "page": page_num,
                    },
                ))

    if new_docs:
        store.add_documents(new_docs)


def search(store, query: str, authorized_companies: list, k: int = 5) -> list:
    hits = store.similarity_search_with_relevance_scores(query, k=k * 4)
    results = []
    for doc, score in hits:
        company = doc.metadata.get("company", "")
        if company in authorized_companies:
            results.append({
                "company": company,
                "content": doc.page_content,
                "page": doc.metadata.get("page", "?"),
                "score": round(score, 3),
            })
        if len(results) >= k:
            break
    return results


def init_session():
    for key, val in {
        "chat_histories": {},
        "last_results": [],
        "last_query": "",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val


st.set_page_config(
    page_title="Enterprise Document Intelligence",
    page_icon="🏢",
    layout="wide",
)

init_session()
embedding_model = get_embedding_model()
document_store = get_document_store(embedding_model)
ensure_indexed(document_store)

st.title("🏢 Enterprise Document Intelligence Platform")

st.sidebar.header("User Login")
st.sidebar.write("Sign in with your corporate email to access authorized transcripts.")
user_email = st.sidebar.text_input("Email", placeholder="name@example.com").strip().lower()

if not user_email:
    st.info("Enter your corporate email in the sidebar to begin.")
    st.stop()

authorized_companies = get_user_access(user_email)

if not authorized_companies:
    st.sidebar.error("❌ Access Denied. No documents available for this account.")
    st.error("Your account has no authorized documents. Contact your administrator.")
    st.stop()

st.sidebar.success("✅ Access Granted")
st.sidebar.markdown("**Authorized Documents**")
for company in authorized_companies:
    st.sidebar.write(f"• {company}")
st.sidebar.markdown("---")

chat_history = st.session_state.chat_histories.setdefault(user_email, [])

if st.sidebar.button("Reset Session"):
    st.session_state.chat_histories[user_email] = []
    st.session_state.last_results = []
    st.session_state.last_query = ""
    st.rerun()

st.sidebar.info("Ask about earnings, strategy, risks, or topics in your authorized documents.")

search_tab, history_tab = st.tabs(["🔎 Search", "🗂 History"])

with search_tab:
    st.subheader(f"Query: {', '.join(authorized_companies)}")

    with st.form("query_form"):
        query = st.text_input(
            "Enter your question",
            value=st.session_state.last_query,
            placeholder="What are the latest earnings highlights?",
        )
        col_k, col_btn = st.columns([1, 4])
        with col_k:
            k = st.slider("Results", min_value=1, max_value=10, value=5)
        with col_btn:
            submitted = st.form_submit_button("Search", use_container_width=True)

    if submitted:
        if query.strip():
            st.session_state.last_query = query
            with st.spinner("Searching…"):
                results = search(document_store, query, authorized_companies, k=k)
            st.session_state.last_results = results

            chat_history.append({"role": "user", "content": query})
            if results:
                for r in results:
                    chat_history.append({"role": "assistant", "content": r["content"]})
            else:
                chat_history.append({"role": "assistant", "content": "No relevant information found."})
        else:
            st.warning("Please enter a question before searching.")

    st.markdown("---")
    st.subheader("📄 Search Results")
    if st.session_state.last_results:
        for i, r in enumerate(st.session_state.last_results, 1):
            label = f"{i}. **{r['company']}** — page {r['page']} — relevance {r['score']:.0%}"
            with st.expander(label, expanded=(i == 1)):
                st.write(r["content"])
    elif st.session_state.last_query:
        st.warning("No relevant excerpts found in your authorized documents.")
    else:
        st.info("Search results will appear here after you submit a query.")

    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("Authorized Documents", len(authorized_companies))
    m2.metric("History Entries", len(chat_history))
    if st.session_state.last_query:
        st.caption(f"Last query: {st.session_state.last_query}")

with history_tab:
    st.subheader("Conversation History")
    if not chat_history:
        st.info("Your chat history is empty. Ask a question in the Search tab to begin.")
    else:
        st.caption(f"{len(chat_history)} entries")
        for entry in chat_history:
            if entry["role"] == "user":
                st.chat_message("user").write(entry["content"])
            else:
                st.chat_message("assistant").write(entry["content"])
