import streamlit as st
import pandas as pd
from auth_manager import verify_user_credentials
from history_manager import init_history_db, create_new_session, get_user_sessions, save_chat_message, load_chat_messages
from download_manager import convert_dataframe_to_excel, convert_dataframe_to_csv

# Force wide enterprise screen mapping with customized titles
st.set_page_config(page_title="Pak Qatar | AI Platform", layout="wide", page_icon="🤖")
init_history_db()

# Secure state tracking node initializations
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.current_session_id = None
    st.session_state.active_df = None  # NEW: Locks the active query results to prevent empty screen resets
    st.session_state.active_sql = None # NEW: Locks the generated query block on screen

# ==========================================
# 🔐 AUTHENTICATION LAYER
# ==========================================
if not st.session_state.authenticated:
    col1, login_center, col3 = st.columns([1, 1.2, 1])
    with login_center:
        st.write("") 
        st.write("") 
        with st.container(border=True):
            st.title("PAK QATAR")
            st.caption("Enterprise Intelligence AI Engine Portal")
            st.markdown("---")
            
            user_input = st.text_input("Corporate Username", placeholder="e.g., admin")
            pass_input = st.text_input("Security Password", type="password", placeholder="••••••••")
            
            if st.button("Authenticate Identity →", use_container_width=True, type="primary"):
                if user_input and pass_input:
                    auth_res = verify_user_credentials(user_input, pass_input)
                    if auth_res["authenticated"]:
                        st.session_state.authenticated = True
                        st.session_state.username = user_input
                        st.session_state.role = auth_res["role"]
                        st.success("Identity Verified. Loading System Components...")
                        st.rerun()
                    else:
                        st.error("Authentication Failure: Unrecognized credentials.")
                else:
                    st.warning("Please fill out both fields.")
    st.stop()

# ==========================================
# 📊 CHAT INTERFACE LAYER (CHATGPT DEPLOYMENT)
# ==========================================

with st.sidebar:
    st.markdown(f"### 👤 **{st.session_state.username.upper()}**")
    st.caption(f"Access Role: {st.session_state.role}")
    st.markdown("---")
    
    st.markdown("#### 💬 Conversation Logs")
    if st.button("➕ Start New Session", use_container_width=True):
        new_id = create_new_session(st.session_state.username)
        st.session_state.current_session_id = new_id
        st.session_state.active_df = None  # Clear past calculations
        st.session_state.active_sql = None
        st.rerun()
        
    st.write("")
    sessions = get_user_sessions(st.session_state.username)
    
    seen_titles = set()
    if sessions:
        for sess_id, title in sessions:
            if title not in seen_titles:
                seen_titles.add(title)
                is_active = "🔹 " if sess_id == st.session_state.current_session_id else "📄 "
                if st.button(f"{is_active}{title}", key=sess_id, use_container_width=True):
                    st.session_state.current_session_id = sess_id
                    st.session_state.active_df = None
                    st.session_state.active_sql = None
                    st.rerun()
            if len(seen_titles) >= 5:
                break
    else:
        st.info("No prior chats found.")
        
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.current_session_id = None
        st.session_state.active_df = None
        st.session_state.active_sql = None
        st.rerun()

if not st.session_state.current_session_id:
    st.session_state.current_session_id = create_new_session(st.session_state.username)

st.title("🤖 Core Intelligence Engine")
st.caption(f"Active Session Context Tracking Node: {st.session_state.current_session_id}")
st.markdown("---")

# Render active chat history messages on screen refresh
messages = load_chat_messages(st.session_state.current_session_id)
for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ===================================================
# 3. INTERACTIVE CHAT INPUT ENGINE FOR INFINTIE PROMPTS
# ===================================================
user_prompt = st.chat_input("Ask anything about PQTEPF data summaries...")

if user_prompt:
    # Save the user query to the database instantly
    save_chat_message(st.session_state.current_session_id, "user", user_prompt)
    with st.chat_message("user"):
        st.write(user_prompt)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.write("🔄 **Dynamic Brain Processing:** Compiling layout parameters...")
        
    try:
            from query_builder import generate_genius_sql
            from databases import execute_pqtepf_query
            
            # Step 1: Attempt to compile the first query layout
            compiled_sql = generate_genius_sql(user_prompt, database_name="PQTEPF")
            
            try:
                # Step 2: Attempt execution against live Oracle tables
                live_df = execute_pqtepf_query(compiled_sql)
            except Exception as first_db_error:
                # AUTONOMOUS BRAIN LOOP: Catch the query syntax error code natively!
                error_msg = str(first_db_error)
                
                # Feed the error back to the engine to compile a clean, corrected query instantly
                compiled_sql = generate_genius_sql(user_prompt, database_name="PQTEPF", error_feedback=error_msg)
                
                # Execute the corrected query string seamlessly
                live_df = execute_pqtepf_query(compiled_sql)
                
            st.session_state.active_sql = compiled_sql
            st.session_state.active_df = live_df
            
            save_chat_message(st.session_state.current_session_id, "assistant", "Dynamic Matrix Rendered Successfully.")
            st.rerun()
            
    except Exception as permanent_fault:
      st.error(f"❌ Structural Execution Failure: {str(permanent_fault)}")


# PERSISTENT RENDERING ANCHOR: Keeps your table and files on screen securely!
if st.session_state.active_df is not  None:
    with st.expander("🔍 Review Dynamically Compiled Structural SQL", expanded=False):
        st.code(st.session_state.active_sql, language="sql")
        
    st.success("Business Intelligence Matrix Compiled Successfully:")
    # Render the interactive data grid container
    st.dataframe(st.session_state.active_df, use_container_width=True)
    
    # Render high-control download buttons side-by-side
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Export Report to Excel",
            data=convert_dataframe_to_excel(st.session_state.active_df),
            file_name="Dynamic_Business_Analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            label="📥 Export Report to CSV",
            data=convert_dataframe_to_csv(st.session_state.active_df),
            file_name="Dynamic_Business_Analysis.csv",
            mime="text/csv",
            use_container_width=True
        )
