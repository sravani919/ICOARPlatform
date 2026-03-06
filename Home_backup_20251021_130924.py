# Home.py (FULL UPDATED — Improved bubbles + icons + tinted sidebar background + collapsible sidebar + icon rail)
# ✅ Sidebar order: New chat / Search / Continue analysis / Your chats
# ✅ “Your chats” = REAL chat threads (NOT every question). One thread can have many messages.
# ✅ Right pane chat: USER on RIGHT, ASSISTANT on LEFT (clear differentiation)
# ✅ Dataset/Download/Actions appear INSIDE the same assistant reply bubble (chat-like attachment)
# ✅ “Thinking…” appears as a LEFT assistant bubble while agent runs
# ✅ No session_state mutation-after-widget errors: actions are queued then executed
# ✅ Chat is fixed-height + scrollable; messages sit at bottom (ChatGPT-like)
# ✅ NEW: Sidebar full-height background (#A9A9A9) that truly wraps Streamlit widgets (CSS :has marker trick)
# ✅ NEW: Collapsible sidebar with ChatGPT-like icon rail when collapsed
# ✅ FIXED: No duplicate toggle buttons / no duplicate close buttons

import os
from datetime import datetime
import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from tabs.login import login_error
from icoar_agent import run_agent_response


# -----------------------------
# Session defaults
# -----------------------------
def _ss_default(k, v):
    if k not in st.session_state:
        st.session_state[k] = v


_ss_default("authenticator", None)
_ss_default("user_login", True)
_ss_default("user_registration", False)
_ss_default("user_registration_complete", False)
_ss_default("authentication_status", False)

# Threaded chat state
_ss_default("assistant_threads", [])            # list[thread]
_ss_default("assistant_active_thread", None)   # str thread_id
_ss_default("assistant_notice", "")
_ss_default("assistant_pending_action", None)  # queued action to run safely (session-level queue)

# ✅ Collapsible sidebar state
_ss_default("ai_sidebar_open", True)


# -----------------------------
# Page config + base CSS
# -----------------------------
icon_path = "./headerui_dev/apps/icoar_logo.png"
fallback_icon_path = "./header_tab/src/icoar_logo.png"
page_icon = icon_path if os.path.exists(icon_path) else fallback_icon_path

st.set_page_config(
    page_title="ICOAR",
    page_icon=page_icon,
    layout="wide",
)

st.markdown(
    """
<style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header[data-testid="stHeader"] { display: none; }

  .block-container { padding: 0rem; }
  div[data-testid="stVerticalBlock"] { gap: 0.6rem; }

  /* Slightly tighter + balanced margins */
  .ai-wrap{ margin-left: 4.5%; margin-right: 4.5%; margin-top: 0%; }

  /* ---------------------------
     ✅ REAL Sidebar Background (wraps Streamlit widgets)
     We mark the left column with a hidden element (#ai-sidebar-marker)
     and style the actual Streamlit column via :has()
     --------------------------- */
  #ai-sidebar-marker { display:none; }

  /* Apply to the column that contains the marker */
  div[data-testid="column"]:has(#ai-sidebar-marker) > div{
    background: #F1F1F1 !important;
    border-radius: 16px !important;
    padding: 10px !important;
    min-height: 92vh !important;
  }

  /* Sidebar card (floating glass on top of grey) */
  


  .ai-section-title{
    font-weight: 700;
    margin-top: 0.9rem;
    margin-bottom: 0.35rem;
    display:block;
  }

  /* Welcome */
  .ai-welcome{
    display:flex;
    align-items:center;
    justify-content:center;
    height: 52vh;
    text-align:center;
    color:#333;
  }
  .ai-welcome h2{ font-weight: 750; margin-bottom: 0.35rem; }
  .ai-welcome p{ color:#666; margin-top: 0.2rem; }

  @media (min-width: 1100px){
    .ai-sticky{ position: sticky; top: 12px; }
  }

  /* ---------------------------
     CUSTOM CHAT BUBBLES (IMPROVED)
     --------------------------- */
  .msg-row{ display:flex; width:100%; gap:10px; }
  .msg-left{ justify-content:flex-start; }
  .msg-right{ justify-content:flex-end; }

  .bubble{
    max-width: 78%;
    padding: 10px 12px;
    border-radius: 16px;
    border: 1px solid #ececec;
    line-height: 1.45;
    font-size: 0.98rem;
    background: #fff;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .bubble-user{
    background: #f1f7ff;
    border-color: #d7e7ff;
  }
  .bubble-assistant{
    background: #ffffff;
  }

  /* Avatar/name row */
  .bubble-head{
    display:flex;
    align-items:center;
    gap:8px;
    margin-bottom: 6px;
  }
  .avatar{
    width: 22px;
    height: 22px;
    border-radius: 999px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size: 0.85rem;
    border: 1px solid #e6e6e6;
    background: #fff;
  }
  .name-tag{
    font-size: 0.85rem;
    font-weight: 700;
    color: #444;
  }

  .meta{
    font-size: 0.75rem;
    color:#888;
    margin-top: 6px;
  }

  /* Attachment area INSIDE assistant bubble */
  .bubble-attachment{
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #ededed;
  }

  /* Inline action bar */
  .inline-actions{
    display:flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
  }
  .inline-actions .stButton>button{
    padding: 0.35rem 0.6rem !important;
    border-radius: 999px !important;
  }

  /* Chat list container */
  .chat-wrap{
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 10px;
    padding-right: 4px;
    padding-bottom: 10px;
  }

  /* --- Collapsed icon rail (ChatGPT-like) --- */
  /* ✅ Collapsed rail: remove the extra top space */
  .icon-rail{ padding-top: 0px !important; }



  .icon-btn .stButton>button{
    width: 42px !important;
    height: 42px !important;
    border-radius: 12px !important;
    padding: 0 !important;
    border: 1px solid #e5e7eb !important;
    background: #fff !important;
    font-size: 18px !important;
  }

  .icon-btn .stButton>button:hover{
    border-color:#d1d5db !important;
  }
</style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# React header component (header_tab/build)
# -----------------------------
production = True
root_dir = os.path.dirname(os.path.abspath(__file__))

if production:
    build_dir = os.path.join(root_dir, "header_tab/build")
    _discrete_slider = components.declare_component("discrete_slider", path=build_dir)
else:
    _discrete_slider = components.declare_component("discrete_slider", url="http://localhost:3000")


def discrete_slider():
    return _discrete_slider(default=0, logged_in=False)


def selection_bar_1():
    build_dir2 = os.path.join(root_dir, "header_tab2/build")
    _selection_bar = components.declare_component("discrete_slider", path=build_dir2)
    return _selection_bar()


def selection_bar_2():
    build_dir3 = os.path.join(root_dir, "header_tab3/build")
    _selection_bar = components.declare_component("discrete_slider", path=build_dir3)
    return _selection_bar()


# =========================================================
# AI ASSISTANT HELPERS (THREADS)
# =========================================================
def _now_ts():
    return datetime.now().strftime("%H:%M:%S")


def _thread_id():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


def _get_active_thread():
    tid = st.session_state.get("assistant_active_thread")
    threads = st.session_state.get("assistant_threads") or []
    for t in threads:
        if t.get("id") == tid:
            return t
    return None


def _set_active_thread(tid: str):
    st.session_state.assistant_active_thread = tid


def _create_new_thread():
    tid = _thread_id()
    thread = {
        "id": tid,
        "title": "New chat",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],           # list[{role,text,timestamp,kind,payload}]
        "last_result": None,      # dict from agent
        "last_user_question": "",
        "pending_action": None,   # stash action here so reruns don't lose it
    }
    st.session_state.assistant_threads.append(thread)
    _set_active_thread(tid)
    return thread


def _ensure_active_thread():
    t = _get_active_thread()
    if t is None:
        t = _create_new_thread()
    return t


def _append_msg(thread: dict, role: str, text: str, kind: str = "text", payload=None):
    thread["messages"].append(
        {"role": role, "text": text, "timestamp": _now_ts(), "kind": kind, "payload": payload}
    )
    if len(thread["messages"]) > 200:
        thread["messages"] = thread["messages"][-200:]


def _update_thread_title_if_needed(thread: dict):
    if thread.get("title") in (None, "", "New chat"):
        for m in thread.get("messages", []):
            if m.get("role") == "user" and (m.get("text") or "").strip():
                q = m["text"].strip()
                thread["title"] = q if len(q) <= 34 else q[:34] + "…"
                break


def _set_last_result(thread: dict, result: dict, last_question: str):
    thread["last_result"] = result
    thread["last_user_question"] = last_question


def _preview_csv(path: str, max_rows: int = 5, max_cols: int = 6):
    try:
        df = pd.read_csv(path)
        cols = list(df.columns)[:max_cols]
        return df[cols].head(max_rows), cols, len(df)
    except Exception:
        return None, None, None


def _run_action(action_input: str, spinner_label: str):
    with st.spinner(spinner_label):
        try:
            return run_agent_response(action_input)
        except Exception as e:
            return {"text": f"Error calling agent: {e}", "file": None, "actions": [], "plot_png": None}


def _queue_action(action_str: str, label: str):
    st.session_state.assistant_pending_action = {
        "action": action_str,
        "label": label,
        "queued_at": datetime.now().isoformat(timespec="seconds"),
    }


def _clear_pending_action():
    st.session_state.assistant_pending_action = None


def _extract_actions(result: dict):
    actions = (result or {}).get("actions") or []
    clean_actions = [a for a in actions if a.get("type") == "clean"]
    viz_actions = [a for a in actions if a.get("type") == "visualize"]
    export_actions = [a for a in actions if a.get("type") == "export"]
    analyze_actions = [a for a in actions if a.get("type") == "analyze"]
    return clean_actions, viz_actions, export_actions, analyze_actions


def _render_dataset_and_buttons_inline(thread: dict, result: dict, key_prefix: str = ""):
    if not isinstance(result, dict):
        return

    fpath = result.get("file")
    plot_png = result.get("plot_png")
    clean_actions, viz_actions, export_actions, analyze_actions = _extract_actions(result)

    if plot_png:
        st.image(plot_png, caption="Preview", use_container_width=True)

    if fpath and os.path.exists(fpath):
        with open(fpath, "rb") as f:
            st.download_button(
                "⬇️ Download CSV",
                f,
                file_name=os.path.basename(fpath),
                mime="text/csv",
                use_container_width=True,
                key=f"{key_prefix}_dl_csv",
            )

        df_prev, cols, nrows = _preview_csv(fpath, max_rows=5, max_cols=6)
        if df_prev is not None:
            st.caption(f"Preview (first 5 rows) • columns shown: {len(cols)} • total rows: {nrows}")
            st.dataframe(df_prev, use_container_width=True, height=180)
    else:
        st.info("No dataset file attached for this result.")

    if fpath and os.path.exists(fpath):
        with st.expander("More details", expanded=False):
            df_prev2, cols2, nrows2 = _preview_csv(fpath, max_rows=20, max_cols=10)
            if df_prev2 is not None:
                st.caption(f"Preview (first 20 rows) • total rows: {nrows2}")
                st.dataframe(df_prev2, use_container_width=True, height=280)

    st.markdown("<div class='inline-actions'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    with c1:
        if st.button("Clean", use_container_width=True, disabled=not bool(clean_actions), key=f"{key_prefix}_clean"):
            afile = clean_actions[0].get("file") or fpath
            _queue_action(f"action:clean file:{afile}", "Clean Text")
            st.rerun()

    with c2:
        if st.button("Visualize", use_container_width=True, disabled=not bool(viz_actions), key=f"{key_prefix}_viz"):
            afile = viz_actions[0].get("file") or fpath
            _queue_action(f"action:visualize file:{afile}", "Visualize")
            st.rerun()

    with c3:
        if analyze_actions:
            task_to_action = {}
            for a in analyze_actions:
                t = (a.get("task") or "").strip().lower()
                if t:
                    task_to_action[t] = a

            ordered = [t for t in ["toxicity", "sentiment", "hate", "cyberbullying", "emotion"] if t in task_to_action]
            if not ordered:
                ordered = list(task_to_action.keys())

            sel = st.selectbox(
                "Analysis",
                ordered,
                format_func=lambda x: x.title(),
                key=f"{key_prefix}_task",
                label_visibility="collapsed",
            )
            if st.button("Run", use_container_width=True, key=f"{key_prefix}_run"):
                a = task_to_action[sel]
                afile = a.get("file") or fpath
                task = a.get("task") or sel
                _queue_action(f"action:analyze file:{afile} task:{task}", f"Run {task.title()}")
                st.rerun()
        else:
            st.button("Analyze", use_container_width=True, disabled=True, key=f"{key_prefix}_an_disabled")

    with c4:
        if st.button("Export Excel", use_container_width=True, disabled=not bool(export_actions), key=f"{key_prefix}_export"):
            afile = export_actions[0].get("file") or fpath
            _queue_action(f"action:export file:{afile}", "Export Excel")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _render_chat_bubbles(thread: dict):
    messages = thread.get("messages", [])
    st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)

    for idx, m in enumerate(messages):
        role = m.get("role", "assistant")
        raw = (m.get("text") or "").strip()
        ts = m.get("timestamp", "")
        kind = m.get("kind", "text")
        payload = m.get("payload")

        safe_text = html.escape(raw).replace("\n", "<br/>")
        safe_ts = html.escape(ts)

        if role == "user":
            st.markdown(
                f"""
                <div class="msg-row msg-right">
                  <div class="bubble bubble-user">
                    <div class="bubble-head">
                      <div class="avatar">👤</div>
                      <div class="name-tag">You</div>
                    </div>
                    {safe_text}
                    <div class="meta">{safe_ts}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="msg-row msg-left">
                  <div class="bubble bubble-assistant">
                    <div class="bubble-head">
                      <div class="avatar">🐯</div>
                      <div class="name-tag">ICOAR</div>
                    </div>
                    {safe_text}
                    <div class="meta">{safe_ts}</div>
                    <div class="bubble-attachment">
                """,
                unsafe_allow_html=True,
            )

            if kind == "result" and isinstance(payload, dict):
                _render_dataset_and_buttons_inline(thread, payload, key_prefix=f"{thread['id']}_{idx}")

            st.markdown(
                """
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# AI ASSISTANT PAGE
# =========================================================
def render_ai_assistant_main():
    st.markdown("<div class='ai-wrap'>", unsafe_allow_html=True)

    sidebar_open = st.session_state.get("ai_sidebar_open", True)
    if sidebar_open:
        left, right = st.columns([0.28, 0.72], gap="large")
    else:
        left, right = st.columns([0.07, 0.93], gap="large")

    # PASS 0: convert queued session action -> thread.pending_action + Thinking bubble
    pending = st.session_state.get("assistant_pending_action")
    if pending and isinstance(pending, dict) and pending.get("action"):
        thread0 = _ensure_active_thread()
        thread0["pending_action"] = {"action": pending["action"], "label": pending.get("label", "Action")}
        _clear_pending_action()

        _append_msg(thread0, "user", f"⏭️ Continue: {thread0['pending_action']['label']}")
        _append_msg(thread0, "assistant", "Thinking...")
        _update_thread_title_if_needed(thread0)
        st.rerun()

    # ---------------- LEFT: full sidebar OR icon rail ----------------
    with left:
        # marker so CSS can tint the REAL Streamlit column container
        st.markdown("<div id='ai-sidebar-marker'></div>", unsafe_allow_html=True)

        if sidebar_open:
            st.markdown("<div class='ai-card ai-sticky'>", unsafe_allow_html=True)

            # Header row: title + close
            cA, cB = st.columns([0.85, 0.15])
            with cA:
                st.markdown("### AI Assistant")
            with cB:
                if st.button("✕", key="ai_close_sidebar", help="Close sidebar"):
                    st.session_state.ai_sidebar_open = False
                    st.rerun()

            if st.button("➕  New chat", use_container_width=True, key="ai_new_chat"):
                _create_new_thread()
                st.rerun()

            q = st.text_input("Search chats", placeholder="Search...", key="ai_search")

            st.markdown("<div class='ai-section-title'>Continue analysis</div>", unsafe_allow_html=True)

            active = _get_active_thread() or _ensure_active_thread()
            res = active.get("last_result") or {}
            fpath = (res or {}).get("file")
            clean_actions, viz_actions, export_actions, analyze_actions = _extract_actions(res)

            if st.button("Clean Text", use_container_width=True, disabled=not bool(clean_actions), key="side_clean"):
                afile = clean_actions[0].get("file") or fpath
                _queue_action(f"action:clean file:{afile}", "Clean Text")
                st.rerun()

            if st.button("Visualize", use_container_width=True, disabled=not bool(viz_actions), key="side_viz"):
                afile = viz_actions[0].get("file") or fpath
                _queue_action(f"action:visualize file:{afile}", "Visualize")
                st.rerun()

            if analyze_actions:
                task_to_action = {}
                for a in analyze_actions:
                    t = (a.get("task") or "").strip().lower()
                    if t:
                        task_to_action[t] = a

                ordered = [t for t in ["toxicity", "sentiment", "hate", "cyberbullying", "emotion"] if t in task_to_action]
                if not ordered:
                    ordered = list(task_to_action.keys())

                sel = st.selectbox("Text Analysis", ordered, format_func=lambda x: x.title(), key="side_task")
                if st.button("Run Analysis", use_container_width=True, key="side_run"):
                    a = task_to_action[sel]
                    afile = a.get("file") or fpath
                    task = a.get("task") or sel
                    _queue_action(f"action:analyze file:{afile} task:{task}", f"Run {task.title()}")
                    st.rerun()
            else:
                st.caption("Text analysis available after collect/clean.")

            if st.button("Export Excel", use_container_width=True, disabled=not bool(export_actions), key="side_xlsx"):
                afile = export_actions[0].get("file") or fpath
                _queue_action(f"action:export file:{afile}", "Export Excel")
                st.rerun()

            st.markdown("---")
            st.markdown("**Your chats**")

            threads = st.session_state.get("assistant_threads") or []
            shown = threads
            if q and q.strip():
                qq = q.strip().lower()
                shown = [t for t in threads if qq in (t.get("title", "").lower())]

            for t in reversed(shown[-40:]):
                title = t.get("title") or "New chat"
                if st.button(title, use_container_width=True, key=f"thr_{t['id']}"):
                    _set_active_thread(t["id"])
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)  # close ai-card

        else:
            st.markdown("<div class='icon-rail ai-sticky'>", unsafe_allow_html=True)

            st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
            if st.button("☰", key="rail_open", help="Open sidebar"):
                st.session_state.ai_sidebar_open = True
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
            if st.button("✎", key="rail_new", help="New chat"):
                _create_new_thread()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
            if st.button("🔍", key="rail_search", help="Search chats"):
                st.session_state.ai_sidebar_open = True
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
            if st.button("💬", key="rail_chats", help="Your chats"):
                st.session_state.ai_sidebar_open = True
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
            if st.button("⚙️", key="rail_actions", help="Continue analysis"):
                st.session_state.ai_sidebar_open = True
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)  # close icon-rail

    # ---------------- RIGHT: chat ----------------
    with right:
        thread = _ensure_active_thread()

        if not thread["messages"]:
            st.markdown(
                """
                <div class="ai-welcome">
                  <div>
                    <h2>Ready when you are.</h2>
                    <p>Ask me to collect data, clean it, visualize patterns, or run text analysis.</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns([0.18, 0.64, 0.18])
            with c2:
                with st.form("ai_center_form", clear_on_submit=True):
                    first_prompt = st.text_input("", placeholder="Ask anything", key="ai_center_input")
                    submitted = st.form_submit_button("Send")
            prompt = first_prompt.strip() if (submitted and first_prompt and first_prompt.strip()) else None

        else:
            chat_area = st.container(height=620)
            with chat_area:
                _render_chat_bubbles(thread)

            components.html(
                """
                <script>
                const scrollers = parent.document.querySelectorAll('[data-testid="stVerticalBlock"]');
                let target = null;
                scrollers.forEach(el => {
                  const s = getComputedStyle(el);
                  if ((s.overflowY === 'auto' || s.overflowY === 'scroll') && el.scrollHeight > el.clientHeight) {
                    target = el;
                  }
                });
                if (target) target.scrollTop = target.scrollHeight;
                </script>
                """,
                height=0,
            )

            prompt = st.chat_input("Ask anything")

        if prompt:
            uname = st.session_state.get("username") or st.session_state.get("name") or "anonymous"
            os.environ.setdefault("ICOAR_USERNAME", str(uname))

            _append_msg(thread, "user", prompt)
            _append_msg(thread, "assistant", "Thinking...")
            _update_thread_title_if_needed(thread)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # close ai-wrap

    # ---------------------------
    # SECOND-PASS EXECUTION:
    # If last assistant message is Thinking..., replace it with real result
    # and mark it kind="result" so attachment renders inside that message.
    # ---------------------------
    thread = _get_active_thread() or _ensure_active_thread()
    msgs = thread.get("messages", [])

    if msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("text") == "Thinking...":
        # Priority 1: run pending action (sidebar/inline buttons)
        if thread.get("pending_action") and thread["pending_action"].get("action"):
            action_input = thread["pending_action"]["action"]
            label = thread["pending_action"].get("label", "Action")
            thread["pending_action"] = None

            result = _run_action(action_input, "Working...")

            msgs[-1]["text"] = result.get("text", "Done.")
            msgs[-1]["timestamp"] = _now_ts()
            msgs[-1]["kind"] = "result"
            msgs[-1]["payload"] = result

            _set_last_result(thread, result, f"Continue: {label}")
            _update_thread_title_if_needed(thread)
            st.rerun()

        # Priority 2: normal user question
        last_user = ""
        for m in reversed(thread["messages"]):
            if m.get("role") == "user" and (m.get("text") or "").strip():
                last_user = m["text"].strip()
                break

        try:
            result = run_agent_response(last_user)
        except Exception as e:
            result = {"text": f"Error calling agent: {e}", "file": None, "actions": [], "plot_png": None}

        msgs[-1]["text"] = result.get("text", "Done.")
        msgs[-1]["timestamp"] = _now_ts()
        msgs[-1]["kind"] = "result"
        msgs[-1]["payload"] = result

        _set_last_result(thread, result, last_user)
        _update_thread_title_if_needed(thread)
        st.rerun()


# -------------------------------------------------
# Render tabs (based on header_tab)
# -------------------------------------------------
selected_value = int(discrete_slider())

if selected_value == 0:
    from tabs.login import login

    with open(".streamlit/authenticator.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)

    st.session_state.authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["preauthorized"],
    )
    login(st.session_state.authenticator, config)

elif selected_value == 1:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        from tabs.Data_Collection.data_collection_tab import data_collection_tab
        data_collection_tab()

elif selected_value == 2:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        from tabs.Data_Collection.data_preprocessing_tab import data_preprocessing_tab
        data_preprocessing_tab()

elif selected_value == 3:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        from tabs.validation.validation import validation
        validation()

elif selected_value == 4:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        from tabs.Visualisation.Text_Visualisation import Text_Visualisation_tab
        Text_Visualisation_tab()

elif selected_value == 5:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        user_choice_2 = selection_bar_2()
        if user_choice_2 == "Cyberbullying Image Analysis":
            from tabs.image.bully_classifification import bully_classification
            bully_classification()
        elif user_choice_2 == "Meme Analysis":
            from tabs.image.meme_classification import meme_classification
            meme_classification()
        elif user_choice_2 == "Deepfake Detection":
            from tabs.image.deepfake_detection import df_detection
            df_detection()
        elif user_choice_2 == "Customized Image Analysis":
            from tabs.image.huggingface_image_analysis import huggingface_image_analysis
            huggingface_image_analysis()
        elif user_choice_2 == "Cyberbullying Detection using GPT":
            from tabs.image.bully_classifification import image_classification_llm
            image_classification_llm()

elif selected_value == 6:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        render_ai_assistant_main()

elif selected_value == 7:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        user_choice = selection_bar_1()
        if user_choice == "Text Annotaion":
            from tabs.Text_Annotation.Text_annotation import text_annotation_tab
            text_annotation_tab(labeling_mode="Text Labeling")
        elif user_choice == "Image Labeling":
            from tabs.Text_Annotation.Text_annotation import text_annotation_tab
            text_annotation_tab(labeling_mode="Image Labeling")
        elif user_choice == "Prompt Optimization":
            from tabs.Prompt_Engineering import generate_prompt
            generate_prompt()
        elif user_choice == "In-Context Learning":
            from tabs.Text_Annotation.In_context_leanring import in_context_learning
            in_context_learning()

elif selected_value == 8:
    if not st.session_state["authentication_status"]:
        st.warning("You're logged out. Please sign in to access the features")
    else:
        st.subheader("Account Details")
        st.markdown("**Name**: " + str(st.session_state.get("name", "")))
        st.markdown("**Username**: " + str(st.session_state.get("username", "")))
        st.session_state.authenticator.logout("Logout", "main", key="unique_key")
