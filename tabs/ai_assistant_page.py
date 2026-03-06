import os
from datetime import datetime
import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from icoar_agent import run_agent_response


# -----------------------------
# Session defaults (AI only)
# -----------------------------
def _ss_default(k, v):
    if k not in st.session_state:
        st.session_state[k] = v


_ss_default("assistant_threads", [])
_ss_default("assistant_active_thread", None)
_ss_default("assistant_pending_action", None)   # queue actions safely
_ss_default("ai_sidebar_open", True)

# hard guard: prevents accidental rerun storms
_ss_default("_ai_render_lock", False)


# -----------------------------
# CSS (only for AI assistant)
# Put this in styles.css later if you want — but inline is fine.
# -----------------------------
AI_CSS = """
<style>
  .ai-wrap{ margin-left: 4.5%; margin-right: 4.5%; margin-top: 0%; }

  #ai-sidebar-marker { display:none; }
  div[data-testid="column"]:has(#ai-sidebar-marker) > div{
    background: #F1F1F1 !important;
    border-radius: 16px !important;
    padding: 10px !important;
    min-height: 92vh !important;
  }

  .ai-section-title{
    font-weight: 700;
    margin-top: 0.9rem;
    margin-bottom: 0.35rem;
    display:block;
  }

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
  .bubble-user{ background: #f1f7ff; border-color: #d7e7ff; }
  .bubble-assistant{ background: #ffffff; }

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
  .meta{ font-size: 0.75rem; color:#888; margin-top: 6px; }

  .bubble-attachment{
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #ededed;
  }

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

  .chat-wrap{
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 10px;
    padding-right: 4px;
    padding-bottom: 10px;
  }

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
</style>
"""


# -----------------------------
# Threads helpers
# -----------------------------
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
    st.session_state["assistant_active_thread"] = tid


def _create_new_thread():
    tid = _thread_id()
    thread = {
        "id": tid,
        "title": "New chat",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],       # list[{role,text,timestamp,kind,payload}]
        "last_result": None,  # dict from agent
    }
    st.session_state["assistant_threads"].append(thread)
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


def _extract_actions(result: dict):
    actions = (result or {}).get("actions") or []
    clean_actions = [a for a in actions if a.get("type") == "clean"]
    viz_actions = [a for a in actions if a.get("type") == "visualize"]
    export_actions = [a for a in actions if a.get("type") == "export"]
    analyze_actions = [a for a in actions if a.get("type") == "analyze"]
    return clean_actions, viz_actions, export_actions, analyze_actions


def _preview_csv(path: str, max_rows: int = 5, max_cols: int = 6):
    try:
        df = pd.read_csv(path)
        cols = list(df.columns)[:max_cols]
        return df[cols].head(max_rows), cols, len(df)
    except Exception:
        return None, None, None


def _queue_action(action_str: str, label: str):
    st.session_state["assistant_pending_action"] = {
        "action": action_str,
        "label": label,
        "queued_at": datetime.now().isoformat(timespec="seconds"),
    }


def _pop_queued_action():
    a = st.session_state.get("assistant_pending_action")
    st.session_state["assistant_pending_action"] = None
    return a


def _call_agent(user_text: str):
    try:
        out = run_agent_response(user_text)
    except Exception as e:
        out = {"text": f"Error calling agent: {e}", "file": None, "actions": [], "plot_png": None}
    # normalize
    if isinstance(out, str):
        return {"text": out, "file": None, "actions": [], "plot_png": None}
    if isinstance(out, dict):
        return out
    return {"text": str(out), "file": None, "actions": [], "plot_png": None}


def _render_dataset_and_buttons_inline(result: dict, key_prefix: str = ""):
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

    st.markdown("<div class='inline-actions'>", unsafe_allow_html=True)

    if st.button("Clean", use_container_width=True, disabled=not bool(clean_actions), key=f"{key_prefix}_clean"):
        afile = clean_actions[0].get("file") or fpath
        _queue_action(f"action:clean file:{afile}", "Clean Text")
        st.rerun()

    if st.button("Visualize", use_container_width=True, disabled=not bool(viz_actions), key=f"{key_prefix}_viz"):
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
                _render_dataset_and_buttons_inline(payload, key_prefix=f"{thread['id']}_{idx}")

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
# MAIN ENTRY (call this from Home.py when nav == AI Assistant)
# =========================================================
def render_ai_assistant_page():
    # Prevent accidental double renders in one run
    if st.session_state.get("_ai_render_lock", False):
        st.stop()
    st.session_state["_ai_render_lock"] = True

    st.markdown(AI_CSS, unsafe_allow_html=True)
    st.markdown("<div class='ai-wrap'>", unsafe_allow_html=True)

    sidebar_open = st.session_state.get("ai_sidebar_open", True)
    if sidebar_open:
        colL, colR = st.columns([0.28, 0.72], gap="large")
    else:
        colL, colR = st.columns([0.07, 0.93], gap="large")

    # 1) Execute queued action FIRST (single rerun cycle only when queued)
    queued = _pop_queued_action()
    if queued and queued.get("action"):
        t = _ensure_active_thread()
        _append_msg(t, "user", f"⏭️ Continue: {queued.get('label','Action')}")
        _append_msg(t, "assistant", "Thinking...")
        _update_thread_title_if_needed(t)

        result = _call_agent(queued["action"])

        # Replace the last "Thinking..." bubble with the real result
        msgs = t["messages"]
        if msgs and msgs[-1]["role"] == "assistant" and msgs[-1]["text"] == "Thinking...":
            msgs[-1]["text"] = result.get("text", "Done.")
            msgs[-1]["kind"] = "result"
            msgs[-1]["payload"] = result

        t["last_result"] = result
        _update_thread_title_if_needed(t)
        st.rerun()

    # ---------------- LEFT ----------------
    with colL:
        st.markdown("<div id='ai-sidebar-marker'></div>", unsafe_allow_html=True)

        if sidebar_open:
            headA, headB = st.columns([0.85, 0.15])
            with headA:
                st.markdown("### AI Assistant")
            with headB:
                if st.button("✕", key="ai_close_sidebar", help="Close sidebar"):
                    st.session_state["ai_sidebar_open"] = False
                    st.rerun()

            if st.button("➕  New chat", use_container_width=True, key="ai_new_chat"):
                _create_new_thread()
                st.rerun()

            q = st.text_input("Search chats", placeholder="Search...", key="ai_search")

            st.markdown("<div class='ai-section-title'>Continue analysis</div>", unsafe_allow_html=True)

            active = _ensure_active_thread()
            res = active.get("last_result") or {}
            fpath = res.get("file")
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
                    tsk = (a.get("task") or "").strip().lower()
                    if tsk:
                        task_to_action[tsk] = a

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

        else:
            st.markdown("<div class='icon-rail ai-sticky'>", unsafe_allow_html=True)
            st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
            if st.button("☰", key="rail_open", help="Open sidebar"):
                st.session_state["ai_sidebar_open"] = True
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
            if st.button("✎", key="rail_new", help="New chat"):
                _create_new_thread()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- RIGHT ----------------
    with colR:
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

        chat_area = st.container(height=620)
        with chat_area:
            _render_chat_bubbles(thread)

        prompt = st.chat_input("Ask anything")

        if prompt and prompt.strip():
            prompt = prompt.strip()
            _append_msg(thread, "user", prompt)
            _append_msg(thread, "assistant", "Thinking...")
            _update_thread_title_if_needed(thread)

            result = _call_agent(prompt)

            # Replace Thinking bubble with real content
            msgs = thread["messages"]
            if msgs and msgs[-1]["role"] == "assistant" and msgs[-1]["text"] == "Thinking...":
                msgs[-1]["text"] = result.get("text", "Done.")
                msgs[-1]["kind"] = "result"
                msgs[-1]["payload"] = result

            thread["last_result"] = result
            _update_thread_title_if_needed(thread)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # unlock at end of render
    st.session_state["_ai_render_lock"] = False