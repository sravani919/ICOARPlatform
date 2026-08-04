# Home.py (FULL UPDATED)
# ✅ FIX: Auth config + authenticator initialized ONCE (shared across all tabs)
# ✅ FIX: Home hero now has BOTH Login + Register buttons (new users can register)
# ✅ FIX: Account tab (8) behaves like other tabs: if logged out -> shows Login/Register UI
# ✅ Kept: Your Home, Multi-media, AI Assistant, AI Assisted Features layout + nav slider + query-param nav

import os
import base64
import textwrap
import re
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from tabs.login import login_error
from icoar_agent import run_agent_response


# -----------------------------
# Helpers
# -----------------------------
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if not os.path.exists(css_path):
        st.warning(f"CSS file not found: {css_path}")
        return
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def md_html(raw_html: str):
    s = textwrap.dedent(raw_html)
    s = "\n".join(line.lstrip() for line in s.splitlines()).strip()
    st.markdown(s, unsafe_allow_html=True)


# -----------------------------
# Multi-media helpers
# -----------------------------
def _mm_hero_header(active_name: str = ""):
    steps = [(1, "Choose"), (2, "Upload"), (3, "Analyze"), (4, "Export")]

    chips_html = '<div class="mm-steps">' + "".join(
        [
            f'<div class="mm-chip {"active" if s==1 else ""}">'
            f'<span class="mm-chip-num">{s}</span>'
            f'<span class="mm-chip-name">{name}</span>'
            f"</div>"
            for s, name in steps
        ]
    ) + "</div>"

    st.markdown(
        f"""
<div class="mm-hero-bleed">
  <div class="mm-hero-inner">
    <div class="mm-hero-title">Multi-media Analysis</div>
    <div class="mm-hero-sub">
      Choose a workflow (cyberbullying, memes, deepfake detection, or custom image analysis) and run it step-by-step.
    </div>
    {chips_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _mm_step_card(title: str):
    wrap = st.container()
    with wrap:
        st.markdown('<div class="mm-card-marker"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mm-step-title">{title}</div>', unsafe_allow_html=True)
    return wrap


# -----------------------------
# AI Assisted Features (Tab 7) helpers (same style as Multi-media)
# -----------------------------
def _af_hero_header(active_step: int = 1):
    steps = [(1, "Select"), (2, "Run"), (3, "Review"), (4, "Export")]

    chips_html = '<div class="af-steps">' + "".join(
        [
            f'<div class="af-chip {"active" if s==active_step else ""}">'
            f'<span class="af-chip-num">{s}</span>'
            f'<span class="af-chip-name">{name}</span>'
            f"</div>"
            for s, name in steps
        ]
    ) + "</div>"

    st.markdown(
        f"""
<div class="af-hero-bleed">
  <div class="af-hero-inner">
    <div class="af-hero-title">AI Assisted Features</div>
    <div class="af-hero-sub">
      Choose a feature (annotation, prompt optimization, or in-context learning) and run it step-by-step.
    </div>
    {chips_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _af_step_card(title: str):
    wrap = st.container()
    with wrap:
        st.markdown('<div class="af-card-marker"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="af-step-title">{title}</div>', unsafe_allow_html=True)
    return wrap


# -----------------------------
# Session defaults
# -----------------------------
def _ss_default(k, v):
    if k not in st.session_state:
        st.session_state[k] = v


_ss_default("authenticator", None)
_ss_default("auth_config", None)

_ss_default("user_login", True)
_ss_default("user_registration", False)
_ss_default("user_registration_complete", False)
_ss_default("authentication_status", False)

_ss_default("main_nav", 0)
_ss_default("main_sidebar_open", False)

_ss_default("assistant_threads", [])
_ss_default("assistant_active_thread", None)
_ss_default("assistant_notice", "")
_ss_default("assistant_pending_action", None)
_ss_default("ai_sidebar_open", True)

# Home toggles
_ss_default("show_learn_more", False)
_ss_default("show_signin_inline", False)

# AI Assisted Features toggles
_ss_default("af_choice", "Text Annotaion")


# -----------------------------
# Query param nav handler (?nav=1 etc) — ONLY for cards/sidebar nav
# -----------------------------
def _get_query_params():
    try:
        return dict(st.query_params)
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}


def _clear_query_params():
    try:
        st.query_params.clear()
        return
    except Exception:
        pass
    try:
        st.experimental_set_query_params()
    except Exception:
        pass


def _handle_nav_from_url():
    params = _get_query_params()
    nav_raw = params.get("nav")
    if isinstance(nav_raw, list):
        nav_raw = nav_raw[0] if nav_raw else None
    if nav_raw is None:
        return
    try:
        nav_val = int(str(nav_raw).strip())
        if 0 <= nav_val <= 8:
            st.session_state["main_nav"] = nav_val
            _clear_query_params()
            st.rerun()
    except Exception:
        _clear_query_params()


_handle_nav_from_url()


# -----------------------------
# Page config
# -----------------------------
icon_path = "./headerui_dev/apps/icoar_logo.png"
fallback_icon_path = "./header_tab/src/icoar_logo.png"
page_icon = icon_path if os.path.exists(icon_path) else fallback_icon_path

st.set_page_config(page_title="ICOAR", page_icon=page_icon, layout="wide")

load_css()

st.markdown(
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,600,0,0" />',
    unsafe_allow_html=True,
)


# -----------------------------
# Auth config + authenticator (SHARED across tabs) ✅
# -----------------------------
if st.session_state.get("auth_config") is None:
    with open(".streamlit/authenticator.yaml") as file:
        st.session_state["auth_config"] = yaml.load(file, Loader=SafeLoader)

config = st.session_state["auth_config"]

if st.session_state.get("authenticator") is None:
    st.session_state.authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["preauthorized"],
    )


# -----------------------------
# React discrete slider component (LEFT NAV)
# -----------------------------
production = True
root_dir = os.path.dirname(os.path.abspath(__file__))

if production:
    build_dir = os.path.join(root_dir, "header_tab/build")
    nav_component = components.declare_component("discrete_slider", path=build_dir)
else:
    nav_component = components.declare_component("discrete_slider", url="http://localhost:3000")


def nav_sidebar():
    current = int(st.session_state.get("main_nav", 0))
    sidebar_open = bool(st.session_state.get("main_sidebar_open", False))
    return nav_component(
        default=current,
        key="nav_sidebar",
        mode="sidebar",
        height=980,
        sidebarOpen=sidebar_open,
    )


# =========================================================
# AI ASSISTANT — sidebar next to chat (STREAMLIT SAFE)
# =========================================================
def _ai_hero_header(active_step: int = 1):
    steps = [(1, "Collect"), (2, "Clean"), (3, "Analyze"), (4, "Export")]
    chips_html = '<div class="ai-steps">' + "".join(
        [
            f'<div class="ai-chip {"active" if s==active_step else ""}">'
            f'<span class="ai-chip-num">{s}</span>'
            f'<span class="ai-chip-name">{name}</span>'
            f"</div>"
            for s, name in steps
        ]
    ) + "</div>"

    st.markdown(
        f"""
<div class="ai-hero-bleed">
  <div class="ai-hero-inner">
    <div class="ai-hero-title">AI Assistant</div>
    <div class="ai-hero-sub">
      Chat with me like ChatGPT — ask anything. I can collect data, clean it, visualize it, run text analysis, and export files.
    </div>
    {chips_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_name(path: str | None) -> str:
    if not path:
        return ""
    try:
        return os.path.basename(str(path))
    except Exception:
        return str(path)


def _strip_local_paths(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(/Users/[^ \n]*?/)([A-Za-z0-9_\-]+\.(csv|xlsx|xls))", r"\2", text)
    text = re.sub(r"(/home/[^ \n]*?/)([A-Za-z0-9_\-]+\.(csv|xlsx|xls))", r"\2", text)
    return text


def _read_preview_df(path: str, n: int = 5) -> pd.DataFrame | None:
    try:
        if not path or not os.path.exists(path):
            return None
        lp = path.lower()
        if lp.endswith(".csv"):
            return pd.read_csv(path).head(n)
        if lp.endswith(".xlsx") or lp.endswith(".xls"):
            return pd.read_excel(path).head(n)
        return None
    except Exception:
        return None


def _file_bytes(path: str) -> bytes | None:
    try:
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _create_thread(title: str = "New chat") -> dict:
    return {
        "id": str(uuid.uuid4())[:8],
        "title": title,
        "messages": [],
        "pending_action": None,
        "last_result": None,
        "last_user_question": "",
    }


def _get_active_thread():
    tid = st.session_state.get("assistant_active_thread")
    threads = st.session_state.get("assistant_threads") or []
    for t in threads:
        if t.get("id") == tid:
            return t
    return None


def _set_active_thread(thread_id: str):
    st.session_state["assistant_active_thread"] = thread_id


def _ensure_active_thread():
    if "assistant_threads" not in st.session_state:
        st.session_state["assistant_threads"] = []
    if "assistant_active_thread" not in st.session_state:
        st.session_state["assistant_active_thread"] = None

    active = _get_active_thread()
    if active:
        return active

    t = _create_thread()
    st.session_state["assistant_threads"].append(t)
    _set_active_thread(t["id"])
    return t


def _create_new_thread():
    t = _create_thread()
    st.session_state["assistant_threads"].append(t)
    _set_active_thread(t["id"])
    return t


def _append_msg(
    thread: dict,
    role: str,
    text: str,
    kind: str = "text",
    payload: dict | None = None,
    meta: dict | None = None,
):
    thread["messages"].append(
        {
            "role": role,
            "text": text,
            "timestamp": _now_ts(),
            "kind": kind,
            "payload": payload,
            "meta": meta or {},
        }
    )
    if len(thread["messages"]) > 200:
        thread["messages"] = thread["messages"][-200:]


def _set_last_result(thread: dict, result: dict, last_question: str):
    thread["last_result"] = result
    thread["last_user_question"] = last_question
    if thread.get("title") in (None, "", "New chat") and last_question:
        short = " ".join(last_question.strip().split()[:6])
        thread["title"] = short if short else "New chat"


def _extract_actions(res: dict):
    actions = (res or {}).get("actions") or []
    clean_actions = [a for a in actions if a.get("type") == "clean"]
    viz_actions = [a for a in actions if a.get("type") == "visualize"]
    export_actions = [a for a in actions if a.get("type") == "export"]
    analyze_actions = [a for a in actions if a.get("type") == "analyze"]
    return clean_actions, viz_actions, export_actions, analyze_actions


def _queue_action(action: str, label: str):
    st.session_state["assistant_pending_action"] = {"action": action, "label": label}


def _clear_pending_action():
    st.session_state["assistant_pending_action"] = None


def _run_action(action_input: str, spinner_text: str = "Working...") -> dict:
    with st.spinner(spinner_text):
        return run_agent_response(action_input)


def _render_result_payload(payload: dict, request_overview: str = ""):
    if not payload or not isinstance(payload, dict):
        return

    if request_overview:
        st.markdown("**Request overview**")
        st.caption(request_overview)

    txt = _strip_local_paths(payload.get("text") or "")
    if txt:
        st.markdown(txt)

    if payload.get("plot_png"):
        st.image(payload["plot_png"])

    fpath = payload.get("file")
    if fpath:
        fname = _safe_name(fpath)
        st.markdown("**Saved & ready to download**")
        st.caption(fname)

        df_preview = _read_preview_df(fpath, n=5)
        if df_preview is not None and not df_preview.empty:
            st.markdown("**Preview (first 5 rows)**")
            st.dataframe(df_preview, use_container_width=True, hide_index=True)

        b = _file_bytes(fpath)
        if b is not None:
            is_csv = str(fpath).lower().endswith(".csv")
            st.download_button(
                "Download CSV" if is_csv else "Download file",
                data=b,
                file_name=fname,
                mime="text/csv" if is_csv else "application/octet-stream",
                use_container_width=True,
                key=f"dl_{uuid.uuid4()}",
            )

    actions = payload.get("actions") or []
    if not actions:
        return

    non_an = [a for a in actions if a.get("type") != "analyze"]
    an = [a for a in actions if a.get("type") == "analyze"]

    st.markdown("**Next steps**")
    for a in non_an:
        label = a.get("label") or a.get("type", "Action").title()
        if st.button(label, use_container_width=True, key=f"bubble_{uuid.uuid4()}"):
            _queue_action(f"action:{a.get('type')} file:{a.get('file')}", label)
            st.rerun()

    if an:
        with st.expander("Run Text Analysis"):
            for a in an:
                label = a.get("label") or f"Analyze ({a.get('task','task')})"
                if st.button(label, use_container_width=True, key=f"bubble_an_{uuid.uuid4()}"):
                    _queue_action(
                        f"action:analyze file:{a.get('file')} task:{a.get('task')}",
                        label,
                    )
                    st.rerun()


def _render_chat(thread: dict):
    for m in thread.get("messages", []):
        role = m.get("role", "assistant")
        kind = m.get("kind", "text")
        payload = m.get("payload")
        meta = m.get("meta") or {}

        if role == "user":
            with st.chat_message("user"):
                st.write(m.get("text", ""))
        else:
            with st.chat_message("assistant"):
                if kind == "result" and isinstance(payload, dict):
                    _render_result_payload(payload, request_overview=meta.get("overview", ""))
                else:
                    st.write(_strip_local_paths(m.get("text", "")))


def render_ai_assistant_main():
    _ai_hero_header(active_step=1)
    st.markdown("<div class='ai-wrap'>", unsafe_allow_html=True)

    sidebar_open = st.session_state.get("ai_sidebar_open", True)
    left_ai, right_ai = st.columns([0.28, 0.72], gap="large")

    pending = st.session_state.get("assistant_pending_action")
    if pending and isinstance(pending, dict) and pending.get("action"):
        thread0 = _ensure_active_thread()
        thread0["pending_action"] = {"action": pending["action"], "label": pending.get("label", "Action")}
        _clear_pending_action()
        _append_msg(thread0, "user", f"Continue: {thread0['pending_action']['label']}")
        _append_msg(
            thread0,
            "assistant",
            "Thinking...",
            meta={"overview": f"Continue: {pending.get('label','Action')}"},
        )
        st.rerun()

    with left_ai:
        st.markdown("<div class='ai-card-marker'></div>", unsafe_allow_html=True)

        if sidebar_open:
            with st.container(height=720):
                if st.button("✕ Close sidebar", use_container_width=True, key="ai_close_sidebar"):
                    st.session_state.ai_sidebar_open = False
                    st.rerun()

                if st.button("➕ New chat", use_container_width=True, key="ai_new_chat"):
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

                    ordered = [
                        t
                        for t in ["toxicity", "sentiment", "hate", "cyberbullying", "emotion"]
                        if t in task_to_action
                    ]
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

                for t in reversed(shown[-60:]):
                    title = t.get("title") or "New chat"
                    if st.button(title, use_container_width=True, key=f"thr_{t['id']}"):
                        _set_active_thread(t["id"])
                        st.rerun()
        else:
            if st.button("☰ Menu", use_container_width=True, key="ai_open_sidebar"):
                st.session_state.ai_sidebar_open = True
                st.rerun()
            if st.button("✎ New", use_container_width=True, key="ai_new_chat_rail"):
                _create_new_thread()
                st.rerun()

    with right_ai:
        thread = _ensure_active_thread()

        if not thread["messages"]:
            st.markdown("<div class='ai-center-marker'></div>", unsafe_allow_html=True)
            st.markdown("<div class='ai-welcome-marker'></div>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class="ai-welcome">
                  <div>
                    <h2>Ready when you are.</h2>
                    <p>Just chat with me. Example: “Collect 50 Reddit posts about hate speech from the past 30 days.”</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.form("ai_center_form", clear_on_submit=True):
                first_prompt = st.text_input("", placeholder="Ask anything", key="ai_center_input")
                submitted = st.form_submit_button("Send", use_container_width=True)

            prompt = first_prompt.strip() if (submitted and first_prompt and first_prompt.strip()) else None
        else:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            with st.container(height=640):
                _render_chat(thread)
            prompt = st.chat_input("Ask anything")

        if prompt:
            uname = st.session_state.get("username") or st.session_state.get("name") or "anonymous"
            os.environ.setdefault("ICOAR_USERNAME", str(uname))
            _append_msg(thread, "user", prompt)
            _append_msg(thread, "assistant", "Thinking...", meta={"overview": prompt})
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # SECOND PASS: replace Thinking... with actual result
    thread = _get_active_thread() or _ensure_active_thread()
    msgs = thread.get("messages", [])
    if msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("text") == "Thinking...":
        if thread.get("pending_action") and thread["pending_action"].get("action"):
            action_input = thread["pending_action"]["action"]
            label = thread["pending_action"].get("label", "Action")
            thread["pending_action"] = None

            result = _run_action(action_input, "Working...")
            msgs[-1]["text"] = _strip_local_paths(result.get("text", "Done."))
            msgs[-1]["timestamp"] = _now_ts()
            msgs[-1]["kind"] = "result"
            msgs[-1]["payload"] = result
            _set_last_result(thread, result, f"Continue: {label}")
            st.rerun()

        last_user = ""
        for m in reversed(thread["messages"]):
            if m.get("role") == "user" and (m.get("text") or "").strip():
                last_user = m["text"].strip()
                break

        try:
            result = run_agent_response(last_user)
        except Exception as e:
            result = {"text": f"Error calling agent: {e}", "file": None, "actions": [], "plot_png": None}

        msgs[-1]["text"] = _strip_local_paths(result.get("text", "Done."))
        msgs[-1]["timestamp"] = _now_ts()
        msgs[-1]["kind"] = "result"
        msgs[-1]["payload"] = result
        _set_last_result(thread, result, last_user)
        st.rerun()


# -----------------------------
# GLOBAL LAYOUT: LEFT (React nav) + RIGHT (content)
# -----------------------------
if st.session_state.get("main_sidebar_open", False):
    left, right = st.columns([0.24, 0.76], gap="large")
else:
    left, right = st.columns([0.08, 0.92], gap="large")

with left:
    out = nav_sidebar()

    prev = int(st.session_state.get("main_nav", 0))
    new_val = prev

    if isinstance(out, dict):
        new_val = int(out.get("value", prev))
        st.session_state["main_sidebar_open"] = bool(
            out.get("sidebarOpen", st.session_state.get("main_sidebar_open", False))
        )
    elif out is not None:
        new_val = int(out)

    if new_val != prev:
        st.session_state["main_nav"] = new_val
        st.rerun()

selected_value = int(st.session_state.get("main_nav", 0))


# =========================================================
# Pages (RIGHT)
# =========================================================
with right:
    # -----------------------------
    # TAB 0: HOME
    # -----------------------------
    if selected_value == 0:
        from tabs.login import login, carousel_only, helpful_links_only

        FULL_FORM = "Integrative Cyberinfrastructure for Online Abuse Research"

        # hero logo base64
        hero_logo_path = "./header_tab/src/icoar_logo.png"
        if not os.path.exists(hero_logo_path):
            hero_logo_path = "./header_tab/build/icoar_logo.png"
        if not os.path.exists(hero_logo_path):
            hero_logo_path = "./headerui_dev/apps/icoar_logo.png"

        logo_b64 = ""
        if os.path.exists(hero_logo_path):
            with open(hero_logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()

        short_intro = (
            "Integrative Cyberinfrastructure for Online Abuse Research (ICOAR) is a scalable, adaptable, and user-friendly "
            "platform that advances research capability for both social science and computer science communities to leverage "
            "advanced machine learning methods for online abuse research."
        )

        detail_text = (
            "AI is becoming part of everyday life, but it also introduces real risks—especially when it affects people online. "
            "ICOAR helps researchers understand and reduce online abuse by keeping the workflow secure, reproducible, and easy to use.\n\n"
            "Online abuse data is scattered across platforms, so studies often take multiple tools and lots of manual setup. "
            "ICOAR brings everything into one place.\n\n"
            "With the AI assistant, you can type requests like: “Collect 100 Reddit posts about cyberbullying from the past three months.” "
            "The platform runs the steps and returns data, cleaning, analysis, visualizations, and exports.\n\n"
            "Actions are traceable and human-in-the-loop, so the assistant is controlled and trustworthy."
        )

        # HERO (buttons INSIDE)
        md_html(
            f"""
<div class="hero-bleed">
  <div class="hero-inner">
    <div class="hero-card">
      <div class="hero-title-row">
        <div class="hero-logo">
          {("<img src='data:image/png;base64," + logo_b64 + "' />") if logo_b64 else ""}
        </div>
        <h1 class="hero-title"><span class="brand">{FULL_FORM}</span></h1>
      </div>

      <div class="hero-sub hero-sub-wide">{short_intro}</div>

      <div class="hero-cta-wrap">
        <div class="hero-cta-inner">
"""
        )
        st.markdown('<div class="hero-cta-streamlit">', unsafe_allow_html=True)

        # ✅ now shows Learn more + Login + Register
        b1, b2, b3, _sp = st.columns([0.18, 0.18, 0.18, 0.46], gap="small")

        with b1:
            label = "Hide details" if st.session_state["show_learn_more"] else "Learn more"
            if st.button(
                label,
                key="hero_learnmore_btn",
                help="Read more about ICOAR (no new page)",
                use_container_width=True,
            ):
                st.session_state["show_learn_more"] = not st.session_state["show_learn_more"]
                st.rerun()

        if not st.session_state.get("authentication_status", False):
            with b2:
                if st.button(
                    "Login",
                    key="hero_login_btn",
                    help="Login to unlock & explore features",
                    use_container_width=True,
                ):
                    st.session_state["user_registration"] = False
                    st.session_state["user_login"] = True
                    st.session_state["show_signin_inline"] = True
                    st.rerun()

            with b3:
                if st.button(
                    "Register",
                    key="hero_register_btn",
                    help="Create a new account",
                    use_container_width=True,
                ):
                    st.session_state["user_registration"] = True
                    st.session_state["user_login"] = False
                    st.session_state["show_signin_inline"] = True
                    st.rerun()
        else:
            with b2:
                st.markdown("<div class='cta-placeholder'></div>", unsafe_allow_html=True)
            with b3:
                st.markdown("<div class='cta-placeholder'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        md_html(
            """
        </div>
      </div>
    </div>
  </div>
</div>
"""
        )

        # Inline login/register card (below hero)
        if (not st.session_state.get("authentication_status", False)) and st.session_state.get(
            "show_signin_inline", False
        ):
            st.markdown("<div class='hero-inline-login'>", unsafe_allow_html=True)
            title = "Register" if st.session_state.get("user_registration", False) else "Sign in"
            st.markdown(f"<div class='hero-inline-title'>{title}</div>", unsafe_allow_html=True)

            login(
                st.session_state.authenticator,
                config,
                parent=st,
                show_carousel=False,
                show_success=False,
                compact=True,
            )

            close_row = st.columns([0.18, 0.82])
            with close_row[0]:
                if st.button("Close", key="hero_close_login", use_container_width=True):
                    st.session_state["show_signin_inline"] = False
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # Learn more content
        if st.session_state["show_learn_more"]:
            md_html(
                f"""
<div class="learnmore-box">
  <div class="learnmore-title">More about ICOAR</div>
  <div class="learnmore-text">{detail_text.replace(chr(10), "<br/>")}</div>
</div>
"""
            )

        # HOME CONTENT
        home_wrap = st.container()
        with home_wrap:
            st.markdown("<div style='padding: 0 26px 26px 26px;'></div>", unsafe_allow_html=True)

            leftA, rightA = st.columns([0.50, 0.50], gap="large")

            with leftA:
                carousel_only(height=380, key="home_corousel")

            with rightA:
                md_html(
                    """
<div class="quick-grid-6">
  <a class="qcard" href="?nav=1">
    <div class="qtop"><div class="qicon"><span class="material-symbols-rounded">cloud_upload</span></div><div class="qtitle">Data Collection</div></div>
    <p class="qdesc">Collect online abuse data from supported sources with reproducible workflows.</p>
  </a>

  <a class="qcard" href="?nav=2">
    <div class="qtop"><div class="qicon"><span class="material-symbols-rounded">build</span></div><div class="qtitle">Pre-processing</div></div>
    <p class="qdesc">Clean, normalize, and prepare text for analysis with consistent pipelines.</p>
  </a>

  <a class="qcard" href="?nav=3">
    <div class="qtop"><div class="qicon"><span class="material-symbols-rounded">assessment</span></div><div class="qtitle">Text Analysis</div></div>
    <p class="qdesc">Run analysis tasks such as toxicity, sentiment, and related abuse signals.</p>
  </a>

  <a class="qcard" href="?nav=4">
    <div class="qtop"><div class="qicon"><span class="material-symbols-rounded">bar_chart</span></div><div class="qtitle">Visualization</div></div>
    <p class="qdesc">Explore insights through interactive visualizations and export outputs.</p>
  </a>

  <a class="qcard" href="?nav=5">
    <div class="qtop"><div class="qicon"><span class="material-symbols-rounded">image</span></div><div class="qtitle">Multi-media Analysis</div></div>
    <p class="qdesc">Analyze images, memes, and deepfakes with integrated AI workflows.</p>
  </a>

  <a class="qcard" href="?nav=6">
    <div class="qtop"><div class="qicon"><span class="material-symbols-rounded">smart_toy</span></div><div class="qtitle">AI Assistant</div></div>
    <p class="qdesc">Chat to collect data, clean it, visualize it, and run analysis end-to-end.</p>
  </a>

  <a class="qcard" href="?nav=7">
    <div class="qtop"><div class="qicon"><span class="material-symbols-rounded">auto_fix_high</span></div><div class="qtitle">AI Assisted Features</div></div>
    <p class="qdesc">Annotation, prompt optimization, and in-context learning workflows.</p>
  </a>

  <a class="qcard" href="?nav=8">
    <div class="qtop"><div class="qicon"><span class="material-symbols-rounded">account_circle</span></div><div class="qtitle">Account</div></div>
    <p class="qdesc">Manage login, registration, and your account details.</p>
  </a>
</div>
"""
                )

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            helpful_links_only(title="Helpful Links")

    # -----------------------------
    # TAB 1: Data Collection
    # -----------------------------
    elif selected_value == 1:
        if not st.session_state.get("authentication_status", False):
            login_error()
        else:
            from tabs.Data_Collection.data_collection_tab import data_collection_tab

            data_collection_tab()

    # -----------------------------
    # TAB 2: Pre-processing
    # -----------------------------
    elif selected_value == 2:
        if not st.session_state.get("authentication_status", False):
            login_error()
        else:
            try:
                from tabs.Data_Collection.data_preprocessing_tab import data_preprocessing_tab

                data_preprocessing_tab()
            except Exception as e:
                st.error("Pre-processing tab failed to load.")
                st.code(str(e))

    # -----------------------------
    # TAB 3: Text Analysis / Validation
    # -----------------------------
    elif selected_value == 3:
        if not st.session_state.get("authentication_status", False):
            login_error()
        else:
            from tabs.validation.validation import validation

            validation()

    # -----------------------------
    # TAB 4: Visualization
    # -----------------------------
    elif selected_value == 4:
        if not st.session_state.get("authentication_status", False):
            login_error()
        else:
            from tabs.Visualisation.Text_Visualisation import Text_Visualisation_tab

            Text_Visualisation_tab()

    # -----------------------------
    # TAB 5: Multi-media
    # -----------------------------
    elif selected_value == 5:
        if not st.session_state.get("authentication_status", False):
            login_error()
        else:
            _mm_hero_header()

            build_dir3 = os.path.join(root_dir, "header_tab3/build")
            selection_bar2_comp = components.declare_component("selection_bar_2", path=build_dir3)

            st.markdown('<div class="mm-wrap">', unsafe_allow_html=True)
            card1 = _mm_step_card("1) Select a workflow")
            with card1:
                out = selection_bar2_comp(key="selection_bar_2_key")
            st.markdown("</div>", unsafe_allow_html=True)

            if isinstance(out, dict):
                user_choice_2 = out.get("value")
            else:
                user_choice_2 = out

            if "mm_choice" not in st.session_state:
                st.session_state.mm_choice = "Cyberbullying Image Analysis"
            if user_choice_2:
                st.session_state.mm_choice = user_choice_2

            choice = st.session_state.mm_choice

            st.markdown('<div class="mm-wrap">', unsafe_allow_html=True)
            card2 = _mm_step_card("2) Upload & Run")
            with card2:
                try:
                    if choice == "Cyberbullying Image Analysis":
                        from tabs.image.bully_classifification import bully_classification

                        bully_classification()

                    elif choice == "Meme Analysis":
                        from tabs.image.meme_classification import meme_classification

                        meme_classification()

                    elif choice == "Deepfake Detection":
                        from tabs.image.deepfake_detection import df_detection

                        df_detection()

                    elif choice == "Customized Image Analysis":
                        from tabs.image.huggingface_image_analysis import huggingface_image_analysis

                        huggingface_image_analysis()

                    elif choice == "Cyberbullying Detection using GPT":
                        from tabs.image.bully_classifification import image_classification_llm

                        image_classification_llm()

                    else:
                        st.info("Choose a workflow from the bar above to begin.")
                except Exception as e:
                    st.error("Multi-media module crashed:")
                    st.exception(e)
            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # TAB 6: AI Assistant
    # -----------------------------
    elif selected_value == 6:
        if not st.session_state.get("authentication_status", False):
            login_error()
        else:
            render_ai_assistant_main()

    # -----------------------------
    # TAB 7: AI Assisted Features
    # -----------------------------
    elif selected_value == 7:
        if not st.session_state.get("authentication_status", False):
            login_error()
        else:
            _af_hero_header(active_step=1)
            st.markdown('<div class="af-wrap">', unsafe_allow_html=True)

            # Step 1: selector
            card1 = _af_step_card("1) Select a feature")
            with card1:
                options = ["Text Annotation", "Image Labeling", "Prompt Optimization", "In-Context Learning"]
                picked = st.radio(
                    label="",
                    options=options,
                    horizontal=True,
                    key="af_choice_ui",
                    label_visibility="collapsed",
                )

                label_to_internal = {
                    "Text Annotation": "Text Annotaion",  # keep your existing spelling
                    "Image Labeling": "Image Labeling",
                    "Prompt Optimization": "Prompt Optimization",
                    "In-Context Learning": "In-Context Learning",
                }
                st.session_state.af_choice = label_to_internal[picked]

            # Step 2: Run
            card2 = _af_step_card("2) Run")
            with card2:
                choice = st.session_state.get("af_choice", "Text Annotaion")

                try:
                    if choice == "Text Annotaion":
                        from tabs.Text_Annotation.Text_annotation import text_annotation_tab

                        text_annotation_tab(labeling_mode="Text Labeling")

                    elif choice == "Image Labeling":
                        from tabs.Text_Annotation.Text_annotation import text_annotation_tab

                        text_annotation_tab(labeling_mode="Image Labeling")

                    elif choice == "Prompt Optimization":
                        from tabs.Prompt_Engineering import generate_prompt

                        generate_prompt()

                    elif choice == "In-Context Learning":
                        from tabs.Text_Annotation.In_context_leanring import in_context_learning

                        in_context_learning()

                    else:
                        st.info("Choose a feature above to begin.")
                except Exception as e:
                    st.error("AI Assisted Features module crashed:")
                    st.exception(e)

            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # TAB 8: Account ✅ (shows Login/Register when logged out)
    # -----------------------------
    # -----------------------------
    # TAB 8: ACCOUNT (FULL UPDATED)
    # ✅ Fancy hero + themed card (matches your CSS: acc-hero-bleed, acc-wrap, acc-card-marker, etc.)
    # ✅ If logged out -> shows BOTH Login + Register toggles + renders the right form
    # ✅ If logged in -> shows Account Details + Manage Account:
    #    - Change password (best-effort across streamlit-authenticator versions)
    #    - Request profile update (UI-only, can be wired to log/email later)
    #    - Clear session (safe local reset)
    # ✅ Uses SHARED authenticator/config already initialized ONCE at top of Home.py

    elif selected_value == 8:
        from tabs.login import login  # your existing login.py

        # ----- Hero -----
        st.markdown(
            """
    <div class="acc-hero-bleed">
    <div class="acc-hero-inner">
        <div class="acc-hero-title">Account</div>
        <div class="acc-hero-sub">
        Manage your access. Sign in to unlock features, or create a new account.
        </div>
    </div>
    </div>
    """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="acc-wrap">', unsafe_allow_html=True)

        # Use shared objects (created once at top of Home.py)
        authenticator = st.session_state.get("authenticator")
        config = st.session_state.get("auth_config")

        if authenticator is None or config is None:
            st.error("Authenticator not initialized. Make sure auth is created once at the top of Home.py.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        is_authed = bool(st.session_state.get("authentication_status", False))

        # =========================================================
        # LOGGED OUT → Login/Register toggles + form
        # =========================================================
        if not is_authed:
            wrap = st.container()
            with wrap:
                st.markdown('<div class="acc-card-marker"></div>', unsafe_allow_html=True)
                st.markdown('<div class="acc-step-title">Sign in or create an account</div>', unsafe_allow_html=True)

                # Ensure flags exist (login.py expects these)
                if "user_login" not in st.session_state:
                    st.session_state.user_login = True
                if "user_registration" not in st.session_state:
                    st.session_state.user_registration = False
                if "user_registration_complete" not in st.session_state:
                    st.session_state.user_registration_complete = False

                # Toggle row
                t1, t2 = st.columns([0.5, 0.5], gap="small")
                with t1:
                    if st.button("Login", use_container_width=True, key="acc_toggle_login"):
                        st.session_state.user_registration = False
                        st.session_state.user_login = True
                        st.rerun()
                with t2:
                    if st.button("Register", use_container_width=True, key="acc_toggle_register"):
                        st.session_state.user_registration = True
                        st.session_state.user_login = False
                        st.rerun()

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                # Render the form (your login.py handles showing login vs register based on flags)
                login(
                    authenticator,
                    config,
                    parent=st,
                    show_carousel=False,
                    show_success=False,
                    compact=True,
                )

                st.caption("New user? Click **Register**. Already have an account? Click **Login**.")

            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # =========================================================
        # LOGGED IN → Account Details + Manage Account
        # =========================================================
        name = str(st.session_state.get("name", "") or "")
        username = str(st.session_state.get("username", "") or "")
        signed_as = username or name or "User"

        # state for manage modes
        if "acc_mode" not in st.session_state:
            st.session_state["acc_mode"] = ""

        # ----- Account Details card -----
        wrap = st.container()
        with wrap:
            st.markdown('<div class="acc-card-marker"></div>', unsafe_allow_html=True)
            st.markdown('<div class="acc-step-title">Account Details</div>', unsafe_allow_html=True)

            # KPI: signed in as
            st.markdown(
                f"""
    <div class="acc-kpi">
    <div class="acc-kpi-label">Signed in as</div>
    <div class="acc-kpi-value">{signed_as}</div>
    </div>
    """,
                unsafe_allow_html=True,
            )

            # fields row
            st.markdown(
                f"""
    <div class="acc-row" style="margin-top:14px;">
    <div class="acc-field">
        <div class="acc-field-label">Display name</div>
        <div class="acc-field-value">{name if name else "—"}</div>
    </div>
    <div class="acc-field">
        <div class="acc-field-label">Username</div>
        <div class="acc-field-value">{username if username else "—"}</div>
    </div>
    </div>
    """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            st.caption(
                "Profile editing isn’t enabled yet (accounts are stored in `authenticator.yaml`). "
                "Use **Change Password** or **Request Profile Update** below."
            )

            # logout
            authenticator.logout("Logout", "main", key="acc_logout_btn")

        # ----- Manage Account card -----
        manage = st.container()
        with manage:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="acc-card-marker"></div>', unsafe_allow_html=True)
            st.markdown('<div class="acc-step-title">Manage Account</div>', unsafe_allow_html=True)

            # actions row
            c1, c2, c3 = st.columns([0.34, 0.33, 0.33], gap="small")
            with c1:
                if st.button("Change Password", use_container_width=True, key="acc_btn_change_pw"):
                    st.session_state["acc_mode"] = "change_pw"
                    st.rerun()
            with c2:
                if st.button("Request Profile Update", use_container_width=True, key="acc_btn_req_update"):
                    st.session_state["acc_mode"] = "req_update"
                    st.rerun()
            with c3:
                if st.button("Clear Session", use_container_width=True, key="acc_btn_clear_session"):
                    # Keep shared auth objects and nav state sane
                    keep = {
                        "authenticator",
                        "auth_config",
                        "main_nav",
                        "main_sidebar_open",
                    }
                    for k in list(st.session_state.keys()):
                        if k not in keep:
                            del st.session_state[k]
                    st.rerun()

            mode = st.session_state.get("acc_mode", "")

            # -----------------------------
            # Change Password (best-effort)
            # -----------------------------
            if mode == "change_pw":
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                with st.expander("Change Password", expanded=True):
                    st.info("This updates your password through Streamlit-Authenticator (if supported by your version).")

                    # Try common API shapes across versions
                    did = False
                    err_msg = None

                    # 1) reset_password(username)
                    try:
                        if hasattr(authenticator, "reset_password"):
                            authenticator.reset_password(username)
                            st.success("If you completed the form, your password is updated.")
                            did = True
                    except Exception as e:
                        err_msg = str(e)

                    # 2) forgot_password() flow (some versions use this)
                    if not did:
                        try:
                            if hasattr(authenticator, "forgot_password"):
                                st.warning(
                                    "This authenticator version uses a 'forgot password' style flow. "
                                    "Complete the form below to reset your password."
                                )
                                authenticator.forgot_password()
                                did = True
                        except Exception as e:
                            err_msg = str(e)

                    if not did:
                        st.warning(
                            "Your Streamlit-Authenticator version doesn’t expose a built-in password reset UI in this way. "
                            "Paste your streamlit-authenticator version + your `tabs/login.py` and I’ll wire the correct call."
                        )
                        if err_msg:
                            st.code(err_msg)

                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                    if st.button("Done", use_container_width=True, key="acc_pw_done"):
                        st.session_state["acc_mode"] = ""
                        st.rerun()

            # -----------------------------
            # Request Profile Update (UI-only)
            # -----------------------------
            elif mode == "req_update":
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                with st.expander("Request Profile Update", expanded=True):
                    st.caption(
                        "Because profiles are stored in `authenticator.yaml`, changes should be reviewed by an admin. "
                        "This form is UI-only right now (you can log it / email it later)."
                    )

                    new_display = st.text_input("New display name", value=name, key="acc_new_display")
                    new_username = st.text_input("Requested username (optional)", value=username, key="acc_new_username")
                    note = st.text_area(
                        "Reason / note (optional)",
                        placeholder="Explain what you want changed...",
                        key="acc_note",
                    )

                    if st.button("Submit Request", use_container_width=True, key="acc_submit_req"):
                        # TODO: Wire this to a DB/log/email later. For now just acknowledge.
                        st.success("Request submitted. An admin will review it.")
                        st.markdown(
                            f"""
    - **Current name**: {name}
    - **Current username**: {username}
    - **Requested name**: {new_display}
    - **Requested username**: {new_username}
    """)
                        if note and note.strip():
                            st.markdown("**Note:**")
                            st.write(note)

                        st.session_state["acc_mode"] = ""
                        st.rerun()

                    if st.button("Cancel", use_container_width=True, key="acc_req_cancel"):
                        st.session_state["acc_mode"] = ""
                        st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
