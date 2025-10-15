# tabs/common/rasa_ui.py
import os
import streamlit as st
import requests

RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

def send_to_rasa(message: str):
    """Send a single user message to Rasa and return the JSON response."""
    try:
        r = requests.post(
            RASA_URL,
            json={"sender": "streamlit-user", "message": message},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error contacting Rasa: {e}")
        return []

def render_rasa_message(m: dict):
    """Render one Rasa message: text, image, and optional json/custom payload (buttons/results/files)."""
    # 1) Text
    if "text" in m and m["text"]:
        st.write(m["text"])

    # 2) Image
    if "image" in m and m["image"]:
        st.image(m["image"])

    # 3) Custom/JSON payloads
    payload = m.get("json_message") or m.get("custom")
    if not isinstance(payload, dict):
        return

    # 3a) Buttons from actions
    buttons = payload.get("buttons") or []
    if buttons:
        cols = st.columns(min(3, len(buttons)))
        for i, b in enumerate(buttons):
            label = b.get("label") or "Action"
            do = b.get("do")
            args = b.get("args") or {}
            if cols[i % len(cols)].button(label, key=f"assistant_btn_{i}_{label}"):
                if do == "nav":
                    st.session_state["nav_tab"] = args.get("tab")
                    st.success(f"Navigated to {args.get('tab')}")
                elif do == "prefill":
                    platform = args.get("platform", "")
                    keywords = args.get("keywords", "")
                    q = f"find datasets about {keywords} on {platform}"
                    for m2 in send_to_rasa(q):
                        render_rasa_message(m2)
                elif do == "run_text_model":
                    st.info(f"Would run text model: {args.get('model','')}")
                elif do == "run_image_model":
                    st.info(f"Would run image model: {args.get('model','')}")
                elif do == "viz":
                    st.info(f"Would open visualization: {args.get('type','')}")
                elif do == "hf_search":
                    q = f"search {args.get('query','')} datasets on huggingface"
                    for m2 in send_to_rasa(q):
                        render_rasa_message(m2)
                else:
                    # Fallback: just send the label or query back
                    q = args.get("query") or label
                    for m2 in send_to_rasa(q):
                        render_rasa_message(m2)

    # 3b) Search results (dataset list)
    results = payload.get("results") or []
    if results:
        st.markdown("**Search results:**")
        for r in results:
            plat = r.get("platform")
            dsid = r.get("dataset_id")
            title = r.get("title") or dsid
            colA, colB = st.columns([3, 1])
            with colA:
                st.write(f"- **{title}**  \n`{dsid}`  \n*{plat}*")
            with colB:
                default_path = st.session_state.get("default_download_path", "/data/icoar/datasets")
                path = st.text_input(
                    f"Save to (for {dsid})",
                    value=default_path,
                    key=f"assistant_path_{dsid}"
                )
                if st.button("Download", key=f"assistant_dl_{dsid}"):
                    st.session_state["default_download_path"] = path
                    msg = f"download {dsid} to {path}"
                    for m2 in send_to_rasa(msg):
                        render_rasa_message(m2)

    # 3c) Downloaded files
    files = payload.get("files") or []
    if files:
        st.markdown("**Downloaded files:**")
        for f in files:
            name = f.get("name", "file")
            path = f.get("path")
            size = f.get("size_bytes", 0)

            col1, col2, col3 = st.columns([4, 2, 2])
            with col1:
                st.write(f"- **{name}**")
                if size:
                    try:
                        kb = size / 1024
                        mb = kb / 1024
                        st.caption(f"~{mb:.1f} MB" if mb >= 1 else f"~{kb:.0f} KB")
                    except Exception:
                        pass

            with col2:
                st.caption(path or "")

            with col3:
                try:
                    if path and os.path.exists(path) and os.path.isfile(path):
                        with open(path, "rb") as fh:
                            st.download_button(
                                label="Download",
                                data=fh,
                                file_name=name,
                                mime="application/octet-stream",
                                key=f"assistant_dlbtn_{path}",
                            )
                    else:
                        st.warning("File not found on server")
                except Exception as e:
                    st.warning(f"Unavailable: {e}")

