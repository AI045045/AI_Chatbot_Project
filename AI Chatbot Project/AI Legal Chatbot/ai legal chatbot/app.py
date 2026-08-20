"""
AI Legal Chatbot - Streamlit Application Entrypoint

This script renders the main Streamlit web application. It handles routing/views,
user login and sign-up (admin/user), document uploads, and lists RAG session details.
"""

import os
import shutil
from typing import Any, Dict, List, Optional

import streamlit as st

from database import MIN_PASSWORD_LENGTH, authenticate_user, create_user, init_db
from engine import (
    answer_question,
    build_rag_session_from_files,
    check_ollama_reachable,
    ollama_setup_hint,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "users.db")
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, "tmp_uploads")


DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "llama3:8b"


def _ensure_directories() -> None:
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


def _delete_temp_uploads() -> None:
    # Delete all files in the temporary upload folder.
    if os.path.exists(TEMP_UPLOAD_DIR):
        shutil.rmtree(TEMP_UPLOAD_DIR, ignore_errors=True)
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


def _cleanup_rag_session() -> None:
    rag = st.session_state.get("rag_session")
    if rag is not None:
        try:
            rag.cleanup()
        except Exception:
            pass

    # Drop references to encourage GC.
    st.session_state.pop("rag_session", None)
    st.session_state.pop("active_doc_name", None)


def clear_session(*, logout: bool) -> None:
    """
    Confidentiality wipe:
    - deletes all temp uploads
    - flushes in-memory vector index (FAISS)
    - resets chat history
    - optionally logs user out
    """
    _delete_temp_uploads()
    _cleanup_rag_session()
    st.session_state["chat_history"] = []

    if logout:
        st.session_state["authenticated"] = False
        st.session_state["user"] = None

    st.rerun()


def _init_session_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("rag_session", None)
    st.session_state.setdefault("active_doc_name", None)
    st.session_state.setdefault("chat_history", [])


def _require_auth() -> None:
    if not st.session_state.get("authenticated", False):
        st.warning("Please log in to upload documents and chat.")
        st.stop()


def _save_upload(uploaded_file) -> str:
    filename = uploaded_file.name
    dest_path = os.path.join(TEMP_UPLOAD_DIR, filename)

    # Ensure we don't accidentally keep old files with the same name.
    if os.path.exists(dest_path):
        os.remove(dest_path)

    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return dest_path


def _is_ollama_connection_error(exc: Exception) -> bool:
    s = str(exc).lower()
    t = str(exc)
    return (
        "10061" in t
        or "connection refused" in s
        or "actively refused" in s
        or "failed to establish" in s
        or ("max retries exceeded" in s and "11434" in t)
    )


def _is_ollama_model_missing_error(exc: Exception) -> bool:
    """404 / 'model X not found, try pulling' from Ollama API."""
    s = str(exc).lower()
    t = str(exc)
    return (
        "try pulling" in s
        or ("not found" in s and "model" in s)
        or ("404" in t and "model" in s)
    )


def _render_ollama_connection_help() -> None:
    st.error("Ollama is not running or not reachable (needed for embeddings and chat).")
    st.markdown(ollama_setup_hint())


def _render_ollama_model_help(embedding_model: str, llm_model: str) -> None:
    st.error("The embedding or chat model is not downloaded in Ollama yet.")
    st.markdown(
        "Open a terminal where `ollama` works and run:\n\n"
        f"`ollama pull {embedding_model}`  ← required for document indexing\n\n"
        f"`ollama pull {llm_model}`  ← required for answers\n\n"
        "Then try again. You can change names under **Model settings** if you use other models."
    )


def _render_chat() -> None:
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                # Keep citations readable: one line per source chunk.
                citations = msg["citations"]
                st.caption("Sources:")
                for c in citations:
                    chunk_id = c.get("chunk_id") or ""
                    src = c.get("source") or ""
                    page = c.get("page")
                    page_part = f"p{page}" if page else "page?"
                    st.write(f"- [{chunk_id}] {src} ({page_part})")


def main() -> None:
    st.set_page_config(page_title="AI LegalChatBot (Local-Only)", layout="wide")

    _ensure_directories()
    _init_session_state()
    init_db(DB_PATH)

    st.sidebar.title("Session")

    if st.session_state.get("authenticated"):
        user = st.session_state.get("user") or {}
        st.sidebar.markdown(f"**Signed in as:** `{user.get('username')}`")
        st.sidebar.markdown(f"**Role:** `{user.get('role')}`")

        if st.sidebar.button("Clear Session (Wipe Doc + Index)", use_container_width=True):
            clear_session(logout=False)

        if st.sidebar.button("Logout (Clear Session)", use_container_width=True):
            clear_session(logout=True)
    else:
        st.sidebar.info("Log in to upload documents and chat locally.")

    if not st.session_state.get("authenticated"):
        st.title("AI LegalChatBot (Local-Only)")
        st.markdown(
            "Upload a **PDF/DOCX** and ask questions. "
            "All text extraction, embeddings, and LLM inference run **locally** via Ollama. "
            "No document content is sent to external APIs."
        )

        admin_invite_code_required = os.environ.get("ADMIN_INVITE_CODE") is not None
        admin_invite_code_hint = (
            "Set `ADMIN_INVITE_CODE` environment variable to allow admin registration."
            if admin_invite_code_required
            else "To enable admin registration, set `ADMIN_INVITE_CODE`."
        )

        # Use radio (not tabs): `st.form` inside inactive `st.tabs` panels often fails to
        # submit or show messages reliably in Streamlit.
        auth_mode = st.radio(
            "Account",
            ["Login", "Register"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_mode",
        )

        if auth_mode == "Login":
            st.subheader("Login")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login", type="primary")

                if submitted:
                    if not username or not password:
                        st.error("Username and password are required.")
                    else:
                        user = authenticate_user(DB_PATH, username=username, password=password)
                        if not user:
                            st.error("Invalid credentials.")
                        else:
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = {
                                "id": user.id,
                                "username": user.username,
                                "role": user.role,
                            }
                            st.session_state["chat_history"] = []
                            st.success("Logged in successfully.")
                            st.rerun()
        else:
            st.subheader("Register")
            with st.form("register_form", clear_on_submit=False):
                username = st.text_input("Username", key="reg_username")
                password = st.text_input(
                    "Password",
                    type="password",
                    key="reg_password",
                    help=f"At least {MIN_PASSWORD_LENGTH} characters. Long passwords are supported.",
                )
                password2 = st.text_input("Confirm password", type="password", key="reg_password2")

                role_choice = st.selectbox(
                    "Role",
                    options=["user", "admin"],
                    index=0,
                    help="Admins can be created only if ADMIN_INVITE_CODE is set.",
                    key="reg_role",
                )

                admin_code = st.text_input(
                    "Admin invite code (only if Role is admin)",
                    type="password",
                    key="reg_admin_code",
                )

                submitted = st.form_submit_button("Create account", type="primary")

                if submitted:
                    if not username or not password:
                        st.error("Username and password are required.")
                    elif len(password) < MIN_PASSWORD_LENGTH:
                        st.error(
                            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
                        )
                    elif password != password2:
                        st.error("Passwords do not match.")
                    elif role_choice == "admin":
                        expected = os.environ.get("ADMIN_INVITE_CODE")
                        if not expected:
                            st.error(
                                "Admin registration is disabled. Set environment variable "
                                "`ADMIN_INVITE_CODE`, or choose Role: user."
                            )
                        elif admin_code != expected:
                            st.error("Invalid admin invite code.")
                        else:
                            try:
                                create_user(
                                    DB_PATH, username=username, password=password, role="admin"
                                )
                                st.success("Admin account created. Please log in.")
                            except Exception as e:
                                st.error(str(e))
                    else:
                        try:
                            create_user(
                                DB_PATH, username=username, password=password, role="user"
                            )
                            st.success("Account created. Please log in.")
                        except Exception as e:
                            st.error(str(e))

        st.caption(admin_invite_code_hint)
        return

    # ---- Authenticated UI ----
    _require_auth()

    st.title("AI LegalChatBot (Local-Only)")
    st.markdown(
        "Upload a **PDF/DOCX**, then ask questions. Answers are generated using ONLY the retrieved sections from your document."
    )

    with st.expander("Model settings (local only)", expanded=False):
        ok_ollama, ollama_err = check_ollama_reachable()
        if ok_ollama:
            st.caption("Ollama API is reachable.")
        else:
            st.warning(f"Cannot reach Ollama: {ollama_err}")
            st.markdown(ollama_setup_hint())

        embedding_model = st.text_input(
            "Embedding model (Ollama)",
            value=DEFAULT_EMBEDDING_MODEL,
            help="Must be available in Ollama. Example: `nomic-embed-text`.",
        )
        llm_model = st.text_input(
            "LLM model (Ollama)",
            value=DEFAULT_LLM_MODEL,
            help="Must be available in Ollama. Examples: `llama3:8b`, `mistral`.",
        )
        top_k = st.slider("Retriever top_k", min_value=2, max_value=8, value=4, step=1)

    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Document Upload")
        uploaded = st.file_uploader(
            "Upload a legal PDF or DOCX",
            type=["pdf", "docx"],
            accept_multiple_files=False,
        )

        if uploaded is not None:
            st.write(f"Selected: `{uploaded.name}`")
            if st.button("Process Document (Build Local Index)", use_container_width=True):
                clear_existing = st.session_state.get("rag_session") is not None
                if clear_existing:
                    _cleanup_rag_session()
                    _delete_temp_uploads()
                    st.session_state["chat_history"] = []

                try:
                    # Ensure old uploads from prior sessions/documents are removed.
                    _delete_temp_uploads()
                    dest_path = _save_upload(uploaded)
                    with st.spinner("Extracting text, chunking, embedding, and indexing..."):
                        rag_session = build_rag_session_from_files(
                            file_path=dest_path,
                            filename=uploaded.name,
                            embedding_model=embedding_model,
                            llm_model=llm_model,
                        )

                    st.session_state["rag_session"] = rag_session
                    st.session_state["active_doc_name"] = uploaded.name
                    st.session_state["chat_history"] = [
                        {
                            "role": "assistant",
                            "content": (
                                "Document indexed locally. Ask a question and I will answer using only retrieved sections."
                            ),
                            "citations": [],
                        }
                    ]
                    st.success("Document processed successfully.")
                    st.rerun()
                except Exception as e:
                    if _is_ollama_model_missing_error(e):
                        _render_ollama_model_help(embedding_model, llm_model)
                    elif _is_ollama_connection_error(e):
                        _render_ollama_connection_help()
                    else:
                        st.error(f"Failed to process document: {e}")

        else:
            st.info("Upload a PDF/DOCX to enable chat.")

    with col2:
        st.subheader("Chat")
        if not st.session_state.get("rag_session"):
            st.info("Process a document first to enable retrieval-based chat.")
        else:
            _render_chat()

            prompt = st.chat_input("Ask a legal question about the uploaded document...")
            if prompt:
                st.session_state["chat_history"].append(
                    {"role": "user", "content": prompt, "citations": []}
                )

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving relevant sections and generating an answer..."):
                        try:
                            result = answer_question(
                                st.session_state["rag_session"],
                                prompt,
                                top_k=top_k,
                            )
                            answer_text = result["answer"]
                            citations = result["citations"]

                            st.markdown(answer_text)
                            if citations:
                                st.caption("Sources:")
                                for c in citations:
                                    chunk_id = c.get("chunk_id") or ""
                                    src = c.get("source") or ""
                                    page = c.get("page")
                                    page_part = f"p{page}" if page else "page?"
                                    st.write(f"- [{chunk_id}] {src} ({page_part})")

                            st.session_state["chat_history"].append(
                                {
                                    "role": "assistant",
                                    "content": answer_text,
                                    "citations": citations,
                                }
                            )
                        except Exception as e:
                            if _is_ollama_model_missing_error(e):
                                _render_ollama_model_help(embedding_model, llm_model)
                                err = (
                                    "Ollama model missing. Pull the models shown above, then ask again."
                                )
                            elif _is_ollama_connection_error(e):
                                _render_ollama_connection_help()
                                err = (
                                    "Ollama is not reachable. Start Ollama, pull your models, "
                                    "then ask again."
                                )
                            else:
                                err = f"Error generating answer: {e}"
                                st.error(err)
                            st.session_state["chat_history"].append(
                                {"role": "assistant", "content": err, "citations": []}
                            )


if __name__ == "__main__":
    main()

