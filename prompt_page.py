"""Streamlit UI for role-based prompt testing."""

import re
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import oracledb
import streamlit as st

from cancellation import CancellationToken, PromptCancelled
from ollama_client import stream_guest_answer
from package_metadata import read_package_metadata
from prompt_engine import interpret_prompt
from query_executor import execute_generated_query
from report_downloads import dataframe_to_csv, dataframe_to_excel
from session_manager import (
    create_session,
    delete_session,
    delete_user_history,
    get_user_sessions,
    init_history_db,
    load_messages,
    save_message,
)
from users_authentication import GUEST_ROLE, authenticate, can_view_sql


GREETING_WORDS = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
SQL_REQUEST_PATTERN = re.compile(
    r"\b(show|display|view|see|include|print|write|provide|give|explain|what is|what's)"
    r"\b.{0,40}\b(sql|query)\b"
    r"|\b(sql|query)\b.{0,40}\b(show|display|view|see|include|print|write|provide|give|explain)"
    r"\b",
    re.IGNORECASE,
)
PROMPT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="prompt-job")


def _is_greeting(prompt: str) -> bool:
    return prompt.strip().lower().strip("!.? ") in GREETING_WORDS


def _asks_to_see_sql(prompt: str) -> bool:
    return bool(SQL_REQUEST_PATTERN.search(prompt))


def _metadata_label_for_prompt(prompt: str, field: dict) -> str:
    compact_prompt = "".join(re.findall(r"[A-Z0-9_$#]+", prompt, re.IGNORECASE)).lower()
    matches = []
    for label in field["labels"]:
        compact_label = "".join(
            re.findall(r"[A-Z0-9_$#]+", label, re.IGNORECASE)
        ).lower()
        if compact_label and compact_label in compact_prompt:
            matches.append((len(compact_label), label))
    if matches:
        return max(matches, key=lambda item: item[0])[1]
    return str(field["name"]).replace("_", " ").title()


def _hide_internal_field_names(message: str, prompt: str, metadata, result: dict) -> str:
    field_catalog = metadata.field_catalog()
    selected_fields = [
        result.get("measure_field"),
        *(result.get("group_by") or []),
    ]
    for selected in selected_fields:
        if not isinstance(selected, str):
            continue
        field = field_catalog.get(selected.upper())
        if field is None:
            continue
        display_label = _metadata_label_for_prompt(prompt, field)
        hidden_terms = {
            selected,
            *(label for label in field["labels"] if " " in label.strip()),
        }
        hidden_terms.discard(display_label)
        for term in sorted(hidden_terms, key=len, reverse=True):
            message = re.sub(
                rf"\b{re.escape(term)}\b",
                display_label,
                message,
                flags=re.IGNORECASE,
            )
    return message


def _help_text(role: str) -> str:
    return (
        "Ask for a business statistic supported by the current MBS package, such "
        "as total region-wise business for a month. The app returns matching "
        "business results, and Admins and Developers can also review the SQL."
    )


def _respond(
    prompt: str,
    role: str,
    conversation: list[dict[str, str]] | None = None,
    cancellation: CancellationToken | None = None,
) -> tuple[str, dict]:
    if role == GUEST_ROLE:
        guest_messages = conversation or [{"role": "user", "content": prompt}]
        chunks = (
            stream_guest_answer(guest_messages)
            if cancellation is None
            else stream_guest_answer(guest_messages, cancellation)
        )
        return "".join(chunks).strip(), {"kind": "guest"}
    if _is_greeting(prompt):
        return _help_text(role), {"kind": "help"}

    if cancellation is None:
        metadata = read_package_metadata("PKG_BIEL_MBS", owner="PQTEPF")
        result = interpret_prompt(prompt, metadata)
    else:
        metadata = read_package_metadata(
            "PKG_BIEL_MBS",
            owner="PQTEPF",
            cancellation=cancellation,
        )
        result = interpret_prompt(
            prompt,
            metadata,
            cancellation=cancellation,
        )
    result["show_sql"] = can_view_sql(role) and _asks_to_see_sql(prompt)
    if result.get("status") == "ready":
        response = "The business query is ready and will run as a read-only query."
    else:
        response = result.get("message") or "The prompt needs clarification."
        response = _hide_internal_field_names(response, prompt, metadata, result)
    return response, result


def _process_prompt(
    prompt: str,
    role: str,
    conversation: list[dict[str, str]],
    cancellation: CancellationToken,
) -> tuple[str, dict]:
    response, result = _respond(prompt, role, conversation, cancellation)
    cancellation.check()
    if role == GUEST_ROLE or result.get("status") != "ready":
        return response, result

    dataframe = execute_generated_query(result, cancellation=cancellation)
    cancellation.check()
    result["dataframe"] = dataframe
    if dataframe.empty and result.get("empty_result_message"):
        response = result["empty_result_message"]
    else:
        response = f"Results for: {prompt}"
    if result.get("operation") == "direct_count" and not dataframe.empty:
        heading = result.get("measure_heading")
        if heading not in dataframe.columns:
            raise ValueError(
                "The direct count result did not contain its expected heading."
            )
        response = f"{heading}: {int(dataframe.iloc[0][heading]):,}."
    elif not dataframe.empty and result.get("operation") == "distinct_values":
        distinct_count = result.get("distinct_count", len(dataframe))
        response = (
            f"{result.get('result_heading', 'Total Number of Values')}: "
            f"{distinct_count:,}."
        )
        if distinct_count <= 100:
            response += " The matching values are listed below."
    elif (
        not dataframe.empty
        and result.get("aggregation") == "SUM"
        and result.get("measure_heading") in dataframe.columns
    ):
        total = dataframe[result["measure_heading"]].sum()
        response += (
            f"\n\nTotal across {len(dataframe)} groups: {total:,.2f}"
        )
    return response, result


def _login() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1100px;
            padding-top: 4rem;
            padding-bottom: 3rem;
        }
        .login-brand {
            text-align: center;
            margin-bottom: 2rem;
        }
        .login-brand h1 {
            margin-bottom: 0.35rem;
            letter-spacing: 0.04em;
        }
        .login-brand p {
            color: #667085;
            font-size: 1.05rem;
        }
        </style>
        <div class="login-brand">
            <h1>PAK QATAR</h1>
            <p>AI Data and SQL Prompt</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 1.15, 1])
    with center:
        with st.container(border=True):
            st.subheader("Welcome")
            st.caption("Sign in to continue to your role-based workspace.")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                )
                submitted = st.form_submit_button(
                    "Sign in",
                    use_container_width=True,
                    type="primary",
                )
            if submitted:
                identity = authenticate(username, password)
                if identity is None:
                    st.error("Sign-in failed. Check your username and password.")
                else:
                    st.session_state.identity = identity
                    st.session_state.session_id = None
                    st.rerun()
            st.markdown(
                "<p style='text-align:center;color:#667085;margin:0.75rem 0'>"
                "or continue with limited access</p>",
                unsafe_allow_html=True,
            )
            if st.button("Continue as Guest", use_container_width=True):
                st.session_state.identity = {"username": "guest", "role": GUEST_ROLE}
                st.session_state.session_id = None
                st.rerun()
        st.caption("Guest access is limited to general questions through local Ollama.")


def _sidebar(identity: dict) -> None:
    username = identity["username"]
    role = identity["role"]
    with st.sidebar:
        st.markdown(f"### {username.title()}")
        st.caption(f"Role: {role.replace('_', ' ').title()}")
        if role == GUEST_ROLE:
            with st.popover("Clear chat history", use_container_width=True):
                st.warning("This clears the current guest conversation.")
                if st.button("Confirm clear", key="clear_guest_history"):
                    job = st.session_state.get("prompt_job")
                    if job is not None:
                        job["cancellation"].cancel()
                        job["future"].cancel()
                        st.session_state.prompt_job = None
                    st.session_state.guest_messages = []
                    st.session_state.last_result = None
                    st.rerun()
        else:
            if st.button("＋ New chat", use_container_width=True):
                st.session_state.session_id = None
                st.session_state.last_result = None
                st.rerun()
            st.markdown("#### Conversation history")
            sessions = get_user_sessions(username)
            for session_id, title in sessions[:10]:
                select_column, delete_column = st.columns([0.82, 0.18])
                label = ("● " if session_id == st.session_state.session_id else "") + title
                if select_column.button(
                    label,
                    key=f"session_{session_id}",
                    use_container_width=True,
                ):
                    st.session_state.session_id = session_id
                    st.session_state.last_result = None
                    st.rerun()
                if delete_column.button(
                    "×",
                    key=f"delete_session_{session_id}",
                    help="Delete this conversation",
                    use_container_width=True,
                ):
                    job = st.session_state.get("prompt_job")
                    if job is not None and job["session_id"] == session_id:
                        job["cancellation"].cancel()
                        job["future"].cancel()
                        st.session_state.prompt_job = None
                    delete_session(session_id, username)
                    if st.session_state.session_id == session_id:
                        st.session_state.session_id = None
                        st.session_state.last_result = None
                    st.rerun()
            if sessions:
                with st.popover("Delete all history", use_container_width=True):
                    st.warning(
                        "This permanently deletes all your saved conversations."
                    )
                    if st.button("Confirm delete all", key="delete_all_history"):
                        job = st.session_state.get("prompt_job")
                        if job is not None:
                            job["cancellation"].cancel()
                            job["future"].cancel()
                            st.session_state.prompt_job = None
                        delete_user_history(username)
                        st.session_state.session_id = None
                        st.session_state.last_result = None
                        st.rerun()
        if st.button("Log out", use_container_width=True):
            job = st.session_state.get("prompt_job")
            if job is not None:
                job["cancellation"].cancel()
                job["future"].cancel()
            st.session_state.clear()
            st.rerun()


def _render_result(result: dict, role: str) -> None:
    if result.get("kind") == "help" or result.get("kind") == "guest":
        return
    if result.get("status") != "ready":
        return

    if result.get("operation") == "distinct_values":
        st.caption(result.get("result_heading", "Total Number of Values"))
    if result.get("show_sql") and can_view_sql(role):
        with st.expander("Review generated SQL", expanded=True):
            st.code(result["sql"], language="sql")
    dataframe = result.get("dataframe")
    if isinstance(dataframe, pd.DataFrame):
        st.success(f"Returned {len(dataframe)} result row(s).")
        st.dataframe(dataframe, use_container_width=True)
        excel, csv = st.columns(2)
        with excel:
            st.download_button(
                "Download Excel",
                dataframe_to_excel(dataframe),
                file_name="business_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with csv:
            st.download_button(
                "Download CSV",
                dataframe_to_csv(dataframe),
                file_name="business_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )


def _save_cancelled_response(job: dict) -> None:
    message = "Prompt cancelled."
    if job["role"] == GUEST_ROLE:
        st.session_state.guest_messages.append(
            {"role": "assistant", "content": message}
        )
    else:
        save_message(job["session_id"], job["username"], "assistant", message)


@st.fragment(run_every="1s")
def _render_active_prompt() -> None:
    job = st.session_state.get("prompt_job")
    if job is None:
        return

    future = job["future"]
    if future.done():
        if job["cancellation"].cancelled:
            _save_cancelled_response(job)
        else:
            try:
                response, result = future.result()
            except PromptCancelled:
                _save_cancelled_response(job)
            except (RuntimeError, ValueError, LookupError, oracledb.Error) as error:
                st.session_state.prompt_error = str(error)
            else:
                if job["role"] == GUEST_ROLE:
                    st.session_state.guest_messages.append(
                        {"role": "assistant", "content": response}
                    )
                else:
                    save_message(
                        job["session_id"],
                        job["username"],
                        "assistant",
                        response,
                    )
                st.session_state.last_result = {
                    "result": result,
                    "role": job["role"],
                    "session_id": job["session_id"],
                }
        st.session_state.prompt_job = None
        st.rerun()

    st.info(
        "Thinking with local Ollama..."
        if job["role"] == GUEST_ROLE
        else "Preparing a response..."
    )
    if st.button("Stop prompt", key=f"stop_prompt_{job['id']}", type="secondary"):
        job["cancellation"].cancel()
        future.cancel()
        _save_cancelled_response(job)
        st.session_state.prompt_job = None
        st.rerun()


def render_prompt_page() -> None:
    init_history_db()
    if "identity" not in st.session_state:
        st.session_state.identity = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if st.session_state.identity is None:
        _login()
        return

    identity = st.session_state.identity
    username = identity["username"]
    role = identity["role"]
    _sidebar(identity)
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 900px;
            padding-top: 1.5rem;
            padding-bottom: 7rem;
        }
        [data-testid="stAppViewContainer"] .main h1 {
            font-size: 1.7rem;
            font-weight: 600;
            letter-spacing: -0.035em;
            margin-bottom: 0.25rem;
        }
        [data-testid="stAppViewContainer"] .main [data-testid="stCaptionContainer"] {
            color: #737373;
            font-size: 0.9rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #e7e7e5;
        }
        [data-testid="stSidebar"] button {
            border-radius: 10px;
            text-align: left;
        }
        [data-testid="stSidebar"] button p {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        [data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin: 0.65rem 0;
        }
        [data-testid="stChatMessage"] p {
            line-height: 1.65;
        }
        [data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid #e7e7e5;
        }
        [data-testid="stChatInput"] {
            max-width: 900px;
            margin: 0 auto 1rem;
            border: 1px solid #d9d9d6;
            border-radius: 20px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
            background: #fff;
        }
        [data-testid="stChatInput"] textarea {
            min-height: 52px;
            padding-top: 0.85rem;
        }
        [data-testid="stDownloadButton"] button {
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AI Data and SQL Prompt")
    st.caption("Local Ollama | Live Oracle package metadata | Read-only business queries")

    if role == GUEST_ROLE:
        messages = st.session_state.setdefault("guest_messages", [])
    else:
        messages = (
            load_messages(st.session_state.session_id, username)
            if st.session_state.session_id
            else []
        )
        if not messages:
            st.info(_help_text(role))
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    last_result = st.session_state.get("last_result")
    if (
        last_result is not None
        and last_result["session_id"] == st.session_state.session_id
    ):
        _render_result(last_result["result"], last_result["role"])
    if st.session_state.get("prompt_error"):
        st.error(f"Unable to process this prompt: {st.session_state.prompt_error}")
        st.session_state.prompt_error = None

    active_job = st.session_state.get("prompt_job")
    prompt = st.chat_input(
        "Ask a question about the available business data...",
        disabled=active_job is not None,
    )
    if not prompt:
        _render_active_prompt()
        return

    if role != GUEST_ROLE:
        if st.session_state.session_id is None:
            st.session_state.session_id = create_session(username)
        save_message(st.session_state.session_id, username, "user", prompt)
    if role == GUEST_ROLE:
        messages.append({"role": "user", "content": prompt})

    cancellation = CancellationToken()
    session_id = (
        None if role == GUEST_ROLE else st.session_state.session_id
    )
    conversation = list(messages)
    st.session_state.last_result = None
    st.session_state.prompt_job = {
        "id": uuid.uuid4().hex,
        "prompt": prompt,
        "role": role,
        "username": username,
        "session_id": session_id,
        "cancellation": cancellation,
        "future": PROMPT_EXECUTOR.submit(
            _process_prompt,
            prompt,
            role,
            conversation,
            cancellation,
        ),
    }
    st.rerun()
