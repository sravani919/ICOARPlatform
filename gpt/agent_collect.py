import os
from datetime import datetime
import streamlit as st

from ICOAR_core.data_collection.runner import collect_data

# ---- Minimal NL  (platform, method, query) parser ----
def parse_user_request(txt: str):
    t = txt.lower()

    # platform
    platform = None
    if "reddit" in t:
        platform = "reddit"
    elif "kaggle" in t:
        platform = "kaggle_"
    elif "huggingface" in t or "hugging face" in t or "hf " in t:
        platform = "huggingface"

    # defaults (we'll support the 3 most-used first)
    if platform == "reddit":
        # extract count
        import re
        m = re.search(r"\b(\d{2,6})\b", t)
        count = int(m.group(1)) if m else 200

        # keywords = quoted phrases or after 'on/about/for'
        qs = re.findall(r'"([^"]+)"', txt)
        if qs:
            keywords = " ".join(qs)
        else:
            m2 = re.search(r"\b(on|about|for)\b\s+(.+)", txt, re.I)
            keywords = m2.group(2) if m2 else txt

        # flags
        include_comments = any(w in t for w in ["comment", "comments", "replies"])
        only_images = any(w in t for w in ["images only", "only images", "image-only"])

        query = {
            "count": count,
            "keywords": keywords,
            "images": only_images,        # matches Collector.query_options
            "get_comments": include_comments,
            "comment_limit": None,        # or set small int if you want
        }
        method = "Scraper"

    elif platform == "huggingface":
        # expects full HF datasets link like https://huggingface.co/datasets/owner/name
        # or "owner/name"
        import re
        m = re.search(r"(https?://)?(www\.)?huggingface\.co/datasets/([\w\-\./]+)/?", t)
        if m:
            hf = "https://huggingface.co/datasets/" + m.group(3)
        else:
            # try simple "owner/name"
            m2 = re.search(r"\b([\w\-]+)/([\w\-\.\_]+)\b", t)
            if m2:
                hf = f"https://huggingface.co/datasets/{m2.group(1)}/{m2.group(2)}"
            else:
                hf = "https://huggingface.co/datasets/civilcomments"  # fallback example
        query = {"huggingface_dataset": hf}
        method = "API"

    elif platform == "kaggle_":
        # expects https://www.kaggle.com/datasets/owner/name
        import re
        m = re.search(r"(https?://)?(www\.)?kaggle\.com/datasets/([\w\-\./]+)/?", t)
        if not m:
            # fallback (owner/name)
            m2 = re.search(r"\b([\w\-]+)/([\w\-\.\_]+)\b", t)
            if m2:
                ds = f"https://www.kaggle.com/datasets/{m2.group(1)}/{m2.group(2)}"
            else:
                ds = "https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge"  # example
        else:
            ds = "https://www.kaggle.com/datasets/" + m.group(3)

        # Kaggle collector requires "kaggle.username" and "kaggle.key" in query_values
        k_user = st.secrets.get("kaggle", {}).get("username")
        k_key  = st.secrets.get("kaggle", {}).get("key")
        query = {
            "kaggle_dataset": ds,
            "delete_temp_data": True,
            "kaggle.username": k_user,
            "kaggle.key": k_key,
        }
        method = "API"

    else:
        # default to reddit
        platform = "reddit"
        query = {"count": 200, "keywords": txt, "images": False, "get_comments": False, "comment_limit": None}
        method = "Scraper"

    return platform, method, query

# ---- Streamlit UI ----
st.set_page_config(page_title="ICOAR  Auto Collect Agent", page_icon=">")

st.title("ICOAR Auto-Collect Agent >")
st.caption("Say what you need  Ill collect it and give you a download.")

# utils.save_data needs st.session_state['username']; ensure it exists
if "username" not in st.session_state or not st.session_state["username"]:
    st.session_state["username"] = os.environ.get("ICOAR_USERNAME", "anonymous")

# show current user storage path
st.info(f"Your files will be saved under: `data/{st.session_state['username']}/...`")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("e.g., Collect 1000 Reddit posts on \"hateful memes\" with comments since Aug 1")
if user_msg:
    st.session_state["messages"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.status("Parsing your request&", expanded=False) as s:
            platform, method, query_values = parse_user_request(user_msg)
            st.write("**Parsed task**")
            st.code(f"platform={platform}, method={method}\nquery={query_values}")

            # filename base
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = f"{platform}_{ts}"

            s.update(label="Collecting data&", state="running")
            try:
                out_file, rows = collect_data(platform, method, query_values, save_name=base)
                if not out_file:
                    st.error("No results returned.")
                    s.update(label="No data", state="complete")
                else:
                    st.success("Done! Your dataset is ready.")
                    full_path = f"data/{st.session_state['username']}/{out_file}"
                    st.write(f"Saved to `{full_path}`")

                    # Offer download
                    mime = "text/csv"
                    with open(full_path, "rb") as f:
                        st.download_button(" Download CSV", f, file_name=os.path.basename(full_path), mime=mime)

                    # Small preview
                    try:
                        import pandas as pd
                        df = pd.read_csv(full_path)
                        st.dataframe(df.head(25))
                    except Exception:
                        pass

                    s.update(label="Completed", state="complete")
            except Exception as e:
                st.error(f"Collection failed: {e}")
                s.update(label="Failed", state="complete")
