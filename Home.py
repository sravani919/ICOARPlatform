import os
from datetime import datetime

import streamlit as st
import streamlit_authenticator as stauth
import yaml
import requests  # noqa: F401
from yaml.loader import SafeLoader

from tabs.login import login_error
from icoar_agent import run_agent_response


# -----------------------------
# Session defaults
# -----------------------------
if "authenticator" not in st.session_state:
    st.session_state.authenticator = None
if "user_login" not in st.session_state:
    st.session_state.user_login = True
if "user_registration" not in st.session_state:
    st.session_state.user_registration = False
if "user_registration_complete" not in st.session_state:
    st.session_state.user_registration_complete = False
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = False
if "show_ai" not in st.session_state:
    # default hidden until login
    st.session_state.show_ai = False
if "assistant_result" not in st.session_state:
    st.session_state.assistant_result = None
# simple in-session history of Q&A
if "assistant_history" not in st.session_state:
    st.session_state.assistant_history = []
# notice when we reuse an old result
if "assistant_notice" not in st.session_state:
    st.session_state.assistant_notice = ""
# chat-style conversation (user + assistant turns)
if "assistant_chat" not in st.session_state:
    st.session_state.assistant_chat = []


# -----------------------------
# Helper: normalize query text
# -----------------------------
def _normalize_query(text: str) -> str:
    """
    Normalize user query so we can detect duplicates:
    - strip spaces
    - lowercase
    - collapse multiple spaces
    """
    text = (text or "").strip().lower()
    return " ".join(text.split())


# -----------------------------
# Page config + base CSS
# -----------------------------
st.set_page_config(
    page_title="ICOAR",
    page_icon="./header_tab/src/icoar_logo.png",
    layout="wide",
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# -----------------------------
# Sidebar layout CSS (wide vs hidden)
# -----------------------------
AI_SIDEBAR_VISIBLE_CSS = """
<style>
/* widen sidebar when assistant is shown */
[data-testid="stSidebar"] {
  min-width: 520px;
  max-width: 520px;
}
[data-testid="stSidebar"] .stTextArea textarea {
  min-height: 160px;
  line-height: 1.35;
  font-size: 0.95rem;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }
</style>
"""

AI_SIDEBAR_HIDDEN_CSS = """
<style>
/* completely collapse the sidebar when assistant is hidden */
[data-testid="stSidebar"] {
  min-width: 0 !important;
  max-width: 0 !important;
  overflow: hidden !important;
}
section[data-testid="stSidebar"] { display: none; } /* older builds */
</style>
"""


def _apply_ai_sidebar_css():
    if st.session_state.get("show_ai", False):
        st.markdown(AI_SIDEBAR_VISIBLE_CSS, unsafe_allow_html=True)
    else:
        st.markdown(AI_SIDEBAR_HIDDEN_CSS, unsafe_allow_html=True)


# apply now so layout is correct
_apply_ai_sidebar_css()


# -----------------------------
# "Open Assistant" pill under header
# -----------------------------
def render_open_ai_button_after_header():
    # only show if logged in AND currently hidden
    if st.session_state.get("authentication_status") and not st.session_state.get(
        "show_ai", False
    ):
        st.markdown(
            """
        <style>
        .ai-topbar {
          display: flex;
          flex-direction: column;
          align-items: center;
          margin-top: 0.4rem;
          margin-bottom: 0.9rem;
        }
        .ai-topbar p {
          font-size: 16px;
          font-weight: 500;
          color: #333;
          margin-bottom: 0.3rem;
          text-align: center;
        }
        .ai-topbar .stButton > button {
          font-weight: 800 !important;
          font-size: 18px !important;
          padding: 10px 22px !important;
          border-radius: 12px !important;
          border: 1px solid #d0d0d0 !important;
          box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
          background: #fff7e6 !important;   /* ICOAR accent */
          color: #7a3e00 !important;
        }
        .ai-topbar .stButton > button:hover {
          background: #ffe8c2 !important;
          border-color: #caa56d !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="ai-topbar">', unsafe_allow_html=True)
        st.markdown(
            "<p>Need help collecting data or running analysis? "
            "Your personal <b>AI Assistant</b> can do tasks for you &mdash; just ask!</p>",
            unsafe_allow_html=True,
        )
        clicked = st.button(
            "Open Assistant", key="open_ai_topbar_center", use_container_width=False
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if clicked:
            st.session_state.show_ai = True
            _apply_ai_sidebar_css()
            try:
                st.rerun()
            except Exception:
                pass


# -----------------------------
# Navigation helpers (pure Streamlit)
# -----------------------------
def discrete_slider() -> int:
    """
    Main navigation strip replacement.
    Returns an integer 0–7 matching the old tab indices:
      0: Login
      1: Data Collection
      2: Data Preprocessing
      3: Validation
      4: Text Visualisation
      5: Vision (image tools)
      6: Text Annotation
      7: Account
    """
    labels = [
        "Login",
        "Data Collection",
        "Data Preprocessing",
        "Validation",
        "Text Visualisation",
        "Vision",
        "Text Annotation",
        "Account",
    ]
    choice = st.radio(
        "Navigation",
        labels,
        horizontal=True,
        index=0,
    )
    return labels.index(choice)


def selection_bar_1() -> str:
    """
    Secondary menu for Text Annotation tab group.
    Returns one of the following strings:
      - 'Text Annotation'
      - 'Image Labeling'
      - 'Prompt Optimization'
      - 'In-Context Learning'
    """
    options = [
        "Text Annotation",
        "Image Labeling",
        "Prompt Optimization",
        "In-Context Learning",
    ]
    return st.radio(
        "Choose annotation tool:",
        options,
        horizontal=True,
    )


def selection_bar_2() -> str:
    """
    Secondary menu for Vision tab group.
    Returns one of these strings:
      - 'Cyberbullying Image Analysis'
      - 'Meme Analysis'
      - 'Deepfake Detection'
      - 'Customized Image Analysis'
      - 'Cyberbullying Detection using GPT'
    """
    options = [
        "Cyberbullying Image Analysis",
        "Meme Analysis",
        "Deepfake Detection",
        "Customized Image Analysis",
        "Cyberbullying Detection using GPT",
    ]
    return st.radio(
        "Choose vision tool:",
        options,
        horizontal=True,
    )


# -----------------------------
# Page-level CSS tuning
# -----------------------------
st.markdown(
    f"""
    <style>
        .block-container {{
            padding: {0}rem;
        }}
        header[data-testid="stHeader"] {{
            display: none;
        }}
        div[data-testid="stVerticalBlock"] {{
            gap: {0}rem
        }}
        div[data-testid="stForm"]  {{
            margin-left: {5}%;
            margin-right: {5}%;
            margin-top: {1.5}%;
        }}
        div[data-testid="stFormSubmitButton"]  {{
            margin-top: {1.5}%;
        }}
        .stAlert {{
            margin-left: {5}%;
            margin-right: {5}%;
            margin-top: {1.5}%;
        }}
        div[data-testid="stHorizontalBlock"] {{
            margin-left: {7.5}%;
            margin-right: {7.5}%;
            margin-top: {1}%;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Render tabs
# -------------------------------------------------
selected_value = int(discrete_slider())

# Sticky "Open Assistant" teaser under the header
render_open_ai_button_after_header()

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
        cols = st.columns(1)
        with cols[0]:
            from tabs.Data_Collection.data_preprocessing_tab import (
                data_preprocessing_tab,
            )

            data_preprocessing_tab()

elif selected_value == 3:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        from tabs.validation.validation import validation

        cols = st.columns(1)
        with cols[0]:
            validation()

elif selected_value == 4:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        from tabs.Visualisation.Text_Visualisation import Text_Visualisation_tab

        cols = st.columns(1)
        with cols[0]:
            Text_Visualisation_tab()

elif selected_value == 5:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        placeholder = st.empty()
        with placeholder.container():
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
                from tabs.image.huggingface_image_analysis import (
                    huggingface_image_analysis,
                )

                huggingface_image_analysis()
            elif user_choice_2 == "Cyberbullying Detection using GPT":
                from tabs.image.bully_classifification import image_classification_llm

                image_classification_llm()

elif selected_value == 6:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        user_choice = selection_bar_1()
        if user_choice == "Text Annotation":
            from tabs.Text_Annotation.Text_annotation import text_annotation_tab

            cols = st.columns(1)
            with cols[0]:
                text_annotation_tab(labeling_mode="Text Labeling")
        elif user_choice == "Image Labeling":
            from tabs.Text_Annotation.Text_annotation import text_annotation_tab

            cols = st.columns(1)
            with cols[0]:
                text_annotation_tab(labeling_mode="Image Labeling")
        elif user_choice == "Prompt Optimization":
            from tabs.Prompt_Engineering import generate_prompt

            generate_prompt()()
        elif user_choice == "In-Context Learning":
            from tabs.Text_Annotation.In_context_leanring import in_context_learning

            cols = st.columns(1)
            with cols[0]:
                in_context_learning()

elif selected_value == 7:
    if not st.session_state["authentication_status"]:
        st.warning("You're logged out. Please sign in to access the features")
    else:
        cols = st.columns(1)
        with cols[0]:
            st.subheader("Account Details")
            st.markdown("**Name**: " + st.session_state["name"])
            st.markdown("**Username**: " + st.session_state["username"])
            st.session_state.authenticator.logout("Logout", "main", key="unique_key")


# -------------------------------------------------
# Assistant Sidebar (only if logged in + toggled on)
# -------------------------------------------------
if st.session_state.get("authentication_status") and st.session_state.get(
    "show_ai", False
):
    with st.sidebar:
        # Close button styling + chat bubble CSS
        st.markdown(
            """
            <style>
              .ai-close .stButton>button {
                background: transparent !important;
                border: none !important;
                font-size: 20px !important;
                font-weight: 800 !important;
                color: #444 !important;
                padding: 0 !important;
                height: auto !important;
                line-height: 1 !important;
              }
              .chat-container {
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
                max-height: 420px;
                overflow-y: auto;
                padding-right: 4px;
                margin-bottom: 0.5rem;
              }
              .chat-bubble {
                padding: 0.45rem 0.7rem;
                border-radius: 12px;
                font-size: 0.9rem;
                line-height: 1.35;
                word-wrap: break-word;
                white-space: pre-wrap;
              }
              .chat-user {
                background: #e8f0ff;
                align-self: flex-end;
                text-align: right;
              }
              .chat-assistant {
                background: #f7f7f7;
                align-self: flex-start;
              }
              .chat-meta {
                font-size: 0.7rem;
                color: #777;
                margin-top: 0.05rem;
              }
            </style>
        """,
            unsafe_allow_html=True,
        )

        # Header row: title + close "x"
        hcol1, hcol2 = st.columns([1, 0.08])
        with hcol1:
            st.markdown("### AI Assistant (GPT-4o)")
        with hcol2:
            with st.container():
                st.markdown('<div class="ai-close">', unsafe_allow_html=True)
                if st.button("x", key="ai_close_icon"):
                    # collapse sidebar and keep last result in memory
                    st.session_state.show_ai = False
                    _apply_ai_sidebar_css()
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        # ---- New Chat button (clears this session's conversation) ----
        ncol1, _ = st.columns([0.7, 0.3])
        with ncol1:
            if st.button("New Chat", key="new_chat"):
                st.session_state["assistant_chat"] = []
                st.session_state["assistant_result"] = None
                st.session_state["assistant_history"] = []
                st.session_state["assistant_notice"] = ""
                st.session_state["assistant_last_input"] = ""
                st.rerun()

        # tell backend who this user is (used in file paths)
        uname = (
            st.session_state.get("username")
            or st.session_state.get("name")
            or "anonymous"
        )
        os.environ.setdefault("ICOAR_USERNAME", str(uname))

        # -------------------------------
        # Chat-like conversation history
        # -------------------------------
        chat_messages = st.session_state.get("assistant_chat", [])

        with st.container():
            st.markdown("#### Conversation")
            st.markdown(
                '<div class="chat-container">', unsafe_allow_html=True
            )

            if not chat_messages:
                st.markdown(
                    "<div class='chat-assistant chat-bubble'>"
                    "Hi, I'm your ICOAR assistant. Ask me to collect data, clean it, "
                    "visualize patterns, or summarize your dataset.</div>",
                    unsafe_allow_html=True,
                )
            else:
                # show last ~20 messages
                for msg in chat_messages[-20:]:
                    role = msg.get("role", "assistant")
                    text = msg.get("text", "")
                    ts = msg.get("timestamp", "")
                    cls = "chat-user" if role == "user" else "chat-assistant"

                    # bubble
                    st.markdown(
                        f"<div class='{cls} chat-bubble'>{text}</div>",
                        unsafe_allow_html=True,
                    )
                    # tiny timestamp meta (optional)
                    if ts:
                        align = "right" if role == "user" else "left"
                        st.markdown(
                            f"<div class='chat-meta' style='text-align:{align};'>{ts}</div>",
                            unsafe_allow_html=True,
                        )

            st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------
        # User prompt input (like ChatGPT)
        # -------------------------------
        user_prompt = st.text_area(
            "Ask me anything related to ICOAR:",
            key="gpt_input",
            height=140,
            placeholder=(
                "e.g., Collect 15 Reddit posts on cyberbullying from the last month."
            ),
        )

        submit = st.button("Submit", key="gpt_submit", use_container_width=True)

        # if user pressed "Submit": either reuse history or call agent
        if submit:
            cleaned = (user_prompt or "").strip()
            if not cleaned:
                st.warning("Please enter a prompt.")
            else:
                norm = _normalize_query(cleaned)

                # check if this question was already asked in this session
                reused = None
                for item in st.session_state.get("assistant_history", []):
                    if item.get("normalized") == norm:
                        reused = item
                        break

                # store user message in chat log
                chat = st.session_state.get("assistant_chat", [])
                chat.append(
                    {
                        "role": "user",
                        "text": cleaned,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }
                )

                if reused is not None:
                    # reuse previous result instead of recollecting
                    result = reused["result"]
                    st.session_state["assistant_result"] = result
                    st.session_state["assistant_last_input"] = cleaned
                    st.session_state["assistant_notice"] = (
                        "You already asked this question earlier in this session. "
                        "I'm showing the saved result. If you want fresh data, "
                        "change the wording or timeframe and submit again."
                    )

                    # add assistant message to chat from reused result
                    text = (result or {}).get("text") or ""
                    if text:
                        chat.append(
                            {
                                "role": "assistant",
                                "text": text,
                                "timestamp": datetime.now().strftime(
                                    "%H:%M:%S"
                                ),
                            }
                        )

                    st.session_state["assistant_chat"] = chat[-40:]
                    st.rerun()
                else:
                    # new question: call the agent and store in history
                    with st.spinner("Thinking..."):
                        try:
                            result = run_agent_response(cleaned)
                        except Exception as e:
                            result = {
                                "text": f"L Error calling agent: {e}",
                                "file": None,
                                "actions": [],
                                "plot_png": None,
                            }

                    # append to history (Q+result)
                    history = st.session_state.get("assistant_history", [])
                    history.append(
                        {
                            "question": cleaned,
                            "normalized": norm,
                            "result": result,
                            "timestamp": datetime.now().isoformat(
                                timespec="seconds"
                            ),
                        }
                    )
                    st.session_state["assistant_history"] = history[-10:]

                    # add assistant message to chat
                    text = (result or {}).get("text") or ""
                    if text:
                        chat.append(
                            {
                                "role": "assistant",
                                "text": text,
                                "timestamp": datetime.now().strftime(
                                    "%H:%M:%S"
                                ),
                            }
                        )

                    st.session_state["assistant_chat"] = chat[-40:]
                    st.session_state["assistant_result"] = result
                    st.session_state["assistant_last_input"] = cleaned
                    st.session_state["assistant_notice"] = ""
                    st.session_state["show_followup"] = True
                    st.rerun()

        # -------------------------------------------------
        # Render latest result: downloads, plots, buttons
        # (text body already shown in the chat bubbles)
        # -------------------------------------------------
        if st.session_state.get("assistant_result"):
            result = st.session_state["assistant_result"]

            # show notice if we reused an old result
            notice = st.session_state.get("assistant_notice")
            if notice:
                st.info(notice)
                # clear notice after showing once
                st.session_state["assistant_notice"] = ""

            st.markdown("#### Dataset & Actions")

            # show visualization image if we have one
            if result.get("plot_png"):
                st.image(
                    result["plot_png"],
                    caption="Top keywords (preview)",
                    use_column_width=True,
                )

            # download CSV if available
            fpath = result.get("file")
            if fpath and os.path.exists(fpath):
                try:
                    with open(fpath, "rb") as f:
                        st.download_button(
                            "Download CSV",
                            f,
                            file_name=os.path.basename(fpath),
                            mime="text/csv",
                            key="dl_csv",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.warning(f"Could not open file for download: {e}")

            # follow-up action buttons (Clean / Visualize / Summarize / Export)
            actions = result.get("actions") or []
            app_actions = [
                a
                for a in actions
                if a.get("type")
                in {"clean", "visualize", "summarize", "export"}
            ]

            if app_actions:
                st.markdown("#### Continue Analysis")
                cols = st.columns(len(app_actions))

                for i, a in enumerate(app_actions):
                    label = a.get("label") or a.get("type", "").title()
                    afile = a.get("file") or fpath
                    atype = a.get("type")

                    with cols[i]:
                        if st.button(label, key=f"act_{atype}"):
                            # call agent again using synthetic command
                            action_input = f"action:{atype} file:{afile}"
                            with st.spinner(f"Running {label}..."):
                                sub_result = run_agent_response(action_input)

                            # update session with the sub_result so it persists
                            st.session_state["assistant_result"] = sub_result

                            # also add to chat as a short system-style message
                            chat = st.session_state.get("assistant_chat", [])
                            text = (sub_result or {}).get("text") or ""
                            if text:
                                chat.append(
                                    {
                                        "role": "assistant",
                                        "text": text,
                                        "timestamp": datetime.now().strftime(
                                            "%H:%M:%S"
                                        ),
                                    }
                                )
                                st.session_state["assistant_chat"] = chat[-40:]

                            # immediate visual feedback
                            st.success(f"{label} completed successfully!")
                            st.rerun()

            # ---- Previous questions (simple history view) ----
            history = st.session_state.get("assistant_history", [])
            if history:
                with st.expander(
                    "Previous questions in this session", expanded=False
                ):
                    for item in reversed(history[-10:]):
                        q = item.get("question", "")
                        ts = item.get("timestamp", "")
                        res = item.get("result") or {}
                        if isinstance(res, dict):
                            text = res.get("text") or ""
                        else:
                            text = str(res)

                        preview = text.strip()
                        if len(preview) > 260:
                            preview = preview[:260] + "..."

                        st.markdown(f"**Q:** {q}")
                        if ts:
                            st.markdown(
                                f"<sub>{ts}</sub>", unsafe_allow_html=True
                            )
                        if preview:
                            st.markdown(preview)
                        st.markdown("---")
