# Home.py (FULL UPDATED — carousel left + clickable cards right + NO duplicates + Material icons FIXED)

import os
import base64
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

_ss_default("main_nav", 0)
_ss_default("main_sidebar_open", False)

_ss_default("assistant_threads", [])
_ss_default("assistant_active_thread", None)
_ss_default("assistant_notice", "")
_ss_default("assistant_pending_action", None)
_ss_default("ai_sidebar_open", True)


# -----------------------------
# Query param nav handler (click cards -> ?nav=1 etc)
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
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass


def _handle_nav_from_url():
    params = _get_query_params()
    nav_raw = None

    if isinstance(params.get("nav"), list):
        nav_raw = params.get("nav")[0] if params.get("nav") else None
    else:
        nav_raw = params.get("nav")

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
# Page config + base CSS
# -----------------------------
ORANGE = "#ff8c00"

icon_path = "./headerui_dev/apps/icoar_logo.png"
fallback_icon_path = "./header_tab/src/icoar_logo.png"
page_icon = icon_path if os.path.exists(icon_path) else fallback_icon_path

st.set_page_config(page_title="ICOAR", page_icon=page_icon, layout="wide")

# ✅ IMPORTANT: load Material Symbols using <link> (reliable)
st.markdown(
    """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,600,0,0" />
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<style>
  /* Remove Streamlit chrome */
  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  header[data-testid="stHeader"] {{ display: none; }}

  /* Kill top gap / padding */
  html, body {{ margin: 0 !important; padding: 0 !important; }}
  div[data-testid="stAppViewContainer"] {{ padding: 0 !important; margin: 0 !important; }}
  section.main > div {{ padding: 0 !important; margin: 0 !important; }}
  .block-container {{ padding: 0 !important; max-width: 100% !important; margin: 0 !important; }}

  /* Remove internal column padding so left rail sits clean */
  div[data-testid="column"] {{ padding: 0 !important; margin: 0 !important; }}
  div[data-testid="stHorizontalBlock"] {{ gap: 0.75rem; padding-left: 0 !important; margin-left: 0 !important; }}

  :root {{
    --orange: {ORANGE};
    --ink: #0f172a;
    --muted: #475569;
    --border: rgba(2,6,23,0.10);
    --shadow: 0 12px 26px rgba(0,0,0,0.06);
    --soft: rgba(255,140,0,0.10);
  }}

  .right-wrap {{
    padding: 0px 26px 26px 26px;
  }}

  .hero-bleed {{
    width: 100%;
    padding: 10px 0 22px 0;
    background:
      radial-gradient(1200px 600px at 20% 10%, rgba(255,140,0,0.22), transparent 60%),
      linear-gradient(180deg, #fff7ed 0%, #ffffff 65%);
    border-bottom: 1px solid var(--border);
  }}

  .hero-inner {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 0 26px;
  }}

  .hero-title {{
    margin: 0;
    font-size: 2.55rem;
    line-height: 1.12;
    letter-spacing: -0.03em;
    font-weight: 900;
    color: var(--ink);
  }}

  .brand {{
    color: var(--orange);
    font-weight: 900;
  }}

  .hero-sub {{
    margin-top: 12px;
    max-width: 92ch;
    font-size: 1.06rem;
    line-height: 1.65;
    color: var(--muted);
  }}

  .hero-banner {{
    width: 100%;
    height: 240px;
    border-radius: 22px;
    border: 1px solid var(--border);
    overflow: hidden;
    box-shadow: var(--shadow);
    margin-top: 12px;
    margin-bottom: 16px;
  }}
  .hero-banner img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }}

  .section {{
    margin-top: 18px;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: white;
    box-shadow: var(--shadow);
    overflow: hidden;
  }}
  .section-head {{
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    font-weight: 900;
    color: var(--ink);
  }}
  .section-body {{
    padding: 14px 16px 18px 16px;
  }}

  .right-wrap div[data-testid="stAlert"] {{
    display: none !important;
  }}

  .stButton > button {{
    border-radius: 14px !important;
    font-weight: 900 !important;
    border: 1px solid var(--border) !important;
    padding: 0.6rem 0.8rem !important;
  }}

  /* ✅ Icons */
  .material-symbols-rounded {{
    font-variation-settings: "FILL" 0, "wght" 600, "GRAD" 0, "opsz" 24;
    font-size: 24px;
    line-height: 1;
    color: var(--orange);
  }}

  /* ✅ Quick cards */
  .quick-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 14px;
  }}
  .quick-wide {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 14px;
    margin-top: 14px;
  }}

  .qcard {{
    display: block;
    text-decoration: none !important;
    border: 1px solid var(--border);
    border-radius: 20px;
    background: #ffffff;
    box-shadow: 0 10px 22px rgba(0,0,0,0.05);
    padding: 16px 16px;
    transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    height: 100%;
  }}
  .qcard:hover {{
    transform: translateY(-1px);
    box-shadow: 0 14px 28px rgba(0,0,0,0.08);
    border-color: rgba(255,140,0,0.35);
  }}

  .qtop {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
  }}
  .qicon {{
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: var(--soft);
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 auto;
  }}
  .qtitle {{
    font-weight: 900;
    color: var(--ink);
    font-size: 1.05rem;
    margin: 0;
  }}
  .qdesc {{
    margin: 0;
    color: var(--muted);
    line-height: 1.55;
    font-size: 0.95rem;
  }}

  /* ✅ Carousel fills its left column */
  div[data-testid="stCustomComponentV1"]:has(iframe[title*="corousel"]) {{
    width: 100% !important;
  }}
  iframe[title*="corousel"] {{
    width: 100% !important;
    height: 450px !important;
    border: 0 !important;
    display: block !important;
    border-radius: 18px;
    overflow: hidden;
  }}

  @media (max-width: 820px) {{
    .quick-grid, .quick-wide {{ grid-template-columns: 1fr; }}
  }}
</style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# React discrete slider component
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
    iframe_h = 980 if sidebar_open else 860
    return nav_component(
        default=current,
        key="nav_sidebar",
        mode="sidebar",
        height=iframe_h,
        sidebarOpen=sidebar_open,
    )


# =========================================================
# AI ASSISTANT (unchanged behavior)
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
        "messages": [],
        "last_result": None,
        "last_user_question": "",
        "pending_action": None,
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


def _preview_csv(path: str, max_rows: int = 5, max_cols: int = 6):
    try:
        df = pd.read_csv(path)
        cols = list(df.columns)[:max_cols]
        return df[cols].head(max_rows), cols, len(df)
    except Exception:
        return None, None, None


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
                "Download CSV",
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
            task_to_action = {
                (a.get("task") or "").strip().lower(): a
                for a in analyze_actions
                if (a.get("task") or "").strip()
            }
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


def render_ai_assistant_main():
    sidebar_open = st.session_state.get("ai_sidebar_open", True)
    left_ai, right_ai = st.columns([0.28, 0.72] if sidebar_open else [0.08, 0.92], gap="large")

    pending = st.session_state.get("assistant_pending_action")
    if pending and isinstance(pending, dict) and pending.get("action"):
        thread0 = _ensure_active_thread()
        thread0["pending_action"] = {"action": pending["action"], "label": pending.get("label", "Action")}
        _clear_pending_action()
        _append_msg(thread0, "user", f"Continue: {thread0['pending_action']['label']}")
        _append_msg(thread0, "assistant", "Thinking...")
        _update_thread_title_if_needed(thread0)
        st.rerun()

    with left_ai:
        if sidebar_open:
            cA, cB = st.columns([0.82, 0.18])
            with cA:
                st.markdown("### AI Assistant")
            with cB:
                if st.button("X", key="ai_close_sidebar", help="Close sidebar"):
                    st.session_state.ai_sidebar_open = False
                    st.rerun()

            if st.button("New chat", use_container_width=True, key="ai_new_chat"):
                _create_new_thread()
                st.rerun()

            q = st.text_input("Search chats", placeholder="Search...", key="ai_search")

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
            if st.button("Menu", key="ai_open_sidebar", help="Open AI sidebar"):
                st.session_state.ai_sidebar_open = True
                st.rerun()

    with right_ai:
        thread = _ensure_active_thread()

        if not thread["messages"]:
            st.markdown("### Ask ICOAR to collect data, clean it, visualize, or run analysis.")
            prompt = st.chat_input("Ask anything")
        else:
            for idx, m in enumerate(thread["messages"]):
                role = m.get("role", "assistant")
                text = (m.get("text") or "")
                kind = m.get("kind", "text")
                payload = m.get("payload")
                if role == "user":
                    st.chat_message("user").write(text)
                else:
                    with st.chat_message("assistant"):
                        st.write(text)
                        if kind == "result" and isinstance(payload, dict):
                            _render_dataset_and_buttons_inline(thread, payload, key_prefix=f"{thread['id']}_{idx}")

            prompt = st.chat_input("Ask anything")

        if prompt:
            uname = st.session_state.get("username") or st.session_state.get("name") or "anonymous"
            os.environ.setdefault("ICOAR_USERNAME", str(uname))

            _append_msg(thread, "user", prompt)
            _append_msg(thread, "assistant", "Thinking...")
            _update_thread_title_if_needed(thread)
            st.rerun()

    thread = _get_active_thread() or _ensure_active_thread()
    msgs = thread.get("messages", [])
    if msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("text") == "Thinking...":
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


# -----------------------------
# GLOBAL LAYOUT: LEFT (React nav) + RIGHT (page content)
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
# Pages (render in RIGHT)
# =========================================================
with right:
    if selected_value == 0:
        from tabs.login import login, carousel_only, helpful_links_only

        with open(".streamlit/authenticator.yaml") as file:
            config = yaml.load(file, Loader=SafeLoader)

        st.session_state.authenticator = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
            config["preauthorized"],
        )

        FULL_FORM = "Integrative Cyberinfrastructure for Online Abuse Research"
        title_html = f'<span class="brand">{FULL_FORM}</span>'

        # HERO
        st.markdown(
            f"""
            <div class="hero-bleed">
              <div class="hero-inner">
                <h1 class="hero-title">{title_html}</h1>
                <div class="hero-sub">
                  Integrative Cyberinfrastructure for Online Abuse Research (ICOAR) is a scalable, adaptable, and user-friendly
                  platform that advances research capability for both social science and computer science communities to
                  leverage advanced machine learning methods for online abuse research.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='right-wrap'>", unsafe_allow_html=True)

        # Optional banner image
        hero_img = "./assets/icoar_hero.png"
        if os.path.exists(hero_img):
            b64 = base64.b64encode(open(hero_img, "rb").read()).decode()
            st.markdown(
                f"""
                <div class="hero-banner">
                  <img src="data:image/png;base64,{b64}" />
                </div>
                """,
                unsafe_allow_html=True,
            )

        # CTA
        c1, c2, c3 = st.columns([0.22, 0.22, 0.56])
        with c1:
            if st.button("Ready to explore", use_container_width=True):
                st.session_state["main_nav"] = 1
                st.rerun()
        with c2:
            if st.button("Try AI Assistant", use_container_width=True):
                st.session_state["main_nav"] = 6
                st.rerun()

        # ✅ Row: Carousel (left) + Quick cards (right)
        leftA, rightA = st.columns([0.50, 0.50], gap="large")

        with leftA:
            carousel_only(height=380, key="home_corousel")

        with rightA:
            st.markdown(
                """
                <div class="quick-grid">
                  <a class="qcard" href="?nav=1">
                    <div class="qtop">
                      <div class="qicon"><span class="material-symbols-rounded">cloud_upload</span></div>
                      <div class="qtitle">Data Collection</div>
                    </div>
                    <p class="qdesc">Collect online abuse data from supported sources with reproducible workflows.</p>
                  </a>

                  <a class="qcard" href="?nav=2">
                    <div class="qtop">
                      <div class="qicon"><span class="material-symbols-rounded">build</span></div>
                      <div class="qtitle">Pre-processing</div>
                    </div>
                    <p class="qdesc">Clean, normalize, and prepare text for analysis with consistent pipelines.</p>
                  </a>

                  <a class="qcard" href="?nav=3">
                    <div class="qtop">
                      <div class="qicon"><span class="material-symbols-rounded">assessment</span></div>
                      <div class="qtitle">Text Analysis</div>
                    </div>
                    <p class="qdesc">Run analysis tasks such as toxicity, sentiment, and related abuse signals.</p>
                  </a>

                  <a class="qcard" href="?nav=4">
                    <div class="qtop">
                      <div class="qicon"><span class="material-symbols-rounded">bar_chart</span></div>
                      <div class="qtitle">Visualization</div>
                    </div>
                    <p class="qdesc">Explore insights through interactive visualizations and export outputs.</p>
                  </a>
                </div>

                <div class="quick-wide">
                  <a class="qcard" href="?nav=5">
                    <div class="qtop">
                      <div class="qicon"><span class="material-symbols-rounded">image</span></div>
                      <div class="qtitle">Multi-media Analysis</div>
                    </div>
                    <p class="qdesc">Analyze images, memes, and deepfakes with integrated AI workflows.</p>
                  </a>

                  <a class="qcard" href="?nav=6">
                    <div class="qtop">
                      <div class="qicon"><span class="material-symbols-rounded">smart_toy</span></div>
                      <div class="qtitle">AI Assistant</div>
                    </div>
                    <p class="qdesc">Chat to collect data, clean it, visualize it, and run analysis end-to-end.</p>
                  </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ✅ Login UI only when logged out
        if not st.session_state.get("authentication_status", False):
            st.markdown(
                """
                <div class="section">
                  <div class="section-head">Sign in</div>
                  <div class="section-body">
                """,
                unsafe_allow_html=True,
            )
            login(st.session_state.authenticator, config, parent=st, show_carousel=False, show_success=False)
            st.markdown("</div></div>", unsafe_allow_html=True)

        # Platform Overview
        st.markdown(
            """
            <div class="section">
              <div class="section-head">Platform Overview</div>
              <div class="section-body">
            """,
            unsafe_allow_html=True,
        )

        demo_mp4 = "./assets/icoar_demo.mp4"
        demo_url = ""
        if demo_url:
            st.video(demo_url)
        elif os.path.exists(demo_mp4):
            st.video(demo_mp4)
        else:
            st.info("Add your ICOAR demo video path/URL in Home.py (demo_mp4 or demo_url).")

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Helpful links at the bottom
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        helpful_links_only(title="Helpful Links")

        st.markdown("</div>", unsafe_allow_html=True)  # close right-wrap

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
            try:
                from tabs.Data_Collection.data_preprocessing_tab import data_preprocessing_tab
                data_preprocessing_tab()
            except Exception as e:
                st.error("Pre-processing tab failed to load.")
                st.code(str(e))

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
            build_dir3 = os.path.join(root_dir, "header_tab3/build")
            selection_bar2_comp = components.declare_component("selection_bar_2", path=build_dir3)
            user_choice_2 = selection_bar2_comp(key="selection_bar_2_key")

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
            build_dir2 = os.path.join(root_dir, "header_tab2/build")
            selection_bar1_comp = components.declare_component("selection_bar_1", path=build_dir2)
            user_choice = selection_bar1_comp(key="selection_bar_1_key")

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
