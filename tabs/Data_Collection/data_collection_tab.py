# tabs/Data_Collection/data_collection_tab.py
# FULL UPDATED — Wizard UI + full-bleed orange hero + Streamlit-safe cards
# ✅ FIXED: Next button too big (consistent button sizing across steps)

import pkgutil
import pandas as pd
import streamlit as st

from ICOAR_core import data_collection
from ICOAR_core.data_collection.utils import ProgressUpdate, download_images

from .query_builder import query_builder


# -----------------------------
# Session defaults (set once)
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "collector" not in st.session_state:
    st.session_state.collector = None
if "socialmedia" not in st.session_state:
    st.session_state.socialmedia = None
if "social_media_option" not in st.session_state:
    st.session_state.social_media_option = None
if "collector_option" not in st.session_state:
    st.session_state.collector_option = None
if "query_values" not in st.session_state:
    st.session_state.query_values = {}
if "dc_step" not in st.session_state:
    st.session_state.dc_step = 1


def reset_query_values():
    st.session_state.query_values = {}
    st.session_state.results = None


def _load_social_medias():
    social_medias_package_names = [
        name
        for _, name, is_pkg in pkgutil.walk_packages([data_collection.__path__[0]])
        if is_pkg
    ]
    social_medias = {}
    for social_media_package_name in social_medias_package_names:
        sm = getattr(data_collection, social_media_package_name)
        social_medias[sm.name] = sm
    return social_medias


def _get_secret_value(key: str):
    parts = key.split(".")
    value = st.secrets.get(parts[0], {})
    for part in parts[1:]:
        value = value.get(part, {})
    return None if value == {} else value


def _hero_header():
    steps = [
        (1, "Platform"),
        (2, "Collection type"),
        (3, "Query options"),
        (4, "Authentication"),
        (5, "Summary"),
    ]
    active = int(st.session_state.get("dc_step", 1))

    chips_html = '<div class="dc-steps">' + "".join(
        [
            f'<div class="dc-chip {"active" if s==active else ""}">'
            f'<span class="dc-chip-num">{s}</span>'
            f'<span class="dc-chip-name">{name}</span>'
            f"</div>"
            for s, name in steps
        ]
    ) + "</div>"

    st.markdown(
        f"""
<div class="dc-hero-bleed">
  <div class="dc-hero-inner">
    <div class="dc-hero-title">Data Collection</div>
    <div class="dc-hero-sub">Follow the guided steps below. Each step unlocks the next.</div>
    {chips_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _step_card(title: str):
    """
    Streamlit-safe "card" container:
    - marker div inside container
    - CSS uses :has(.dc-card-marker) on the parent VerticalBlock to draw the card
    """
    wrap = st.container()
    with wrap:
        st.markdown('<div class="dc-card-marker"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dc-step-title">{title}</div>', unsafe_allow_html=True)
    return wrap


def data_collection_tab():
    social_medias = _load_social_medias()
    platform_names = sorted(list(social_medias.keys()))

    # defaults valid
    if platform_names and (st.session_state.social_media_option not in platform_names):
        st.session_state.social_media_option = platform_names[0]

    _hero_header()

    # =========================
    # STEP 1 — PLATFORM
    # =========================
    if st.session_state.dc_step == 1:
        st.markdown('<div class="dc-wrap">', unsafe_allow_html=True)
        card = _step_card("1) Choose a platform")

        with card:
            chosen = st.selectbox(
                "Platform",
                platform_names,
                index=platform_names.index(st.session_state.social_media_option)
                if st.session_state.social_media_option in platform_names
                else 0,
            )

            changed = chosen != st.session_state.social_media_option
            st.session_state.social_media_option = chosen
            st.session_state.socialmedia = social_medias[chosen]

            if changed:
                reset_query_values()
                st.session_state.collector_option = None
                st.session_state.collector = None

            spacer, next_col = st.columns([0.76, 0.24])
            with next_col:
                if st.button("Next →", type="primary", use_container_width=True):
                    st.session_state.dc_step = 2
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # STEP 2 — COLLECTION TYPE
    # =========================
    elif st.session_state.dc_step == 2:
        st.markdown('<div class="dc-wrap">', unsafe_allow_html=True)
        card = _step_card("2) Select a collection type")

        with card:
            if st.session_state.socialmedia is None:
                st.warning("Please choose a platform first.")
                if st.button("← Back", use_container_width=False):
                    st.session_state.dc_step = 1
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                return

            methods = list(st.session_state.socialmedia.collection_methods.keys())
            if not methods:
                st.error("No collection methods available for this platform.")
                if st.button("← Back"):
                    st.session_state.dc_step = 1
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                return

            if st.session_state.collector_option not in methods:
                st.session_state.collector_option = methods[0]

            chosen = st.selectbox(
                "Collection type",
                methods,
                index=methods.index(st.session_state.collector_option)
                if st.session_state.collector_option in methods
                else 0,
            )

            changed = chosen != st.session_state.collector_option
            st.session_state.collector_option = chosen
            st.session_state.collector = (
                st.session_state.socialmedia.collection_methods[chosen].Collector()
            )

            if changed:
                reset_query_values()

            spacer, back_col, next_col = st.columns([0.58, 0.18, 0.24])
            with back_col:
                if st.button("← Back", use_container_width=True):
                    st.session_state.dc_step = 1
                    st.rerun()
            with next_col:
                if st.button("Next →", type="primary", use_container_width=True):
                    st.session_state.dc_step = 3
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # STEP 3 — QUERY OPTIONS
    # =========================
    elif st.session_state.dc_step == 3:
        collector = st.session_state.get("collector")
        if collector is None:
            st.warning("Please select platform and collection type first.")
            if st.button("← Back"):
                st.session_state.dc_step = 2
                st.rerun()
            return

        query_options = collector.query_options()

        st.markdown('<div class="dc-wrap">', unsafe_allow_html=True)
        card = _step_card("3) Fill query options")

        with card:
            with st.form("dc_query_form", clear_on_submit=False):
                for q in query_options:
                    st.session_state.query_values[q] = query_builder(q, st.container())

                spacer, back_col, next_col = st.columns([0.58, 0.18, 0.24])
                with back_col:
                    back = st.form_submit_button("← Back", use_container_width=True)
                with next_col:
                    next_ = st.form_submit_button("Next →", use_container_width=True)

            if back:
                st.session_state.dc_step = 2
                st.rerun()
            if next_:
                st.session_state.dc_step = 4
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # STEP 4 — AUTHENTICATION
    # =========================
    elif st.session_state.dc_step == 4:
        collector = st.session_state.get("collector")
        needed_keys = collector.auth() if collector else []

        st.markdown('<div class="dc-wrap">', unsafe_allow_html=True)
        card = _step_card("4) Authentication")

        with card:
            st.caption("Values are preloaded from `secrets.toml` when available.")

            with st.form("dc_auth_form", clear_on_submit=False):
                if not needed_keys:
                    st.success("No authentication needed!")
                else:
                    for key in needed_keys:
                        default_val = _get_secret_value(key)
                        st.session_state.query_values[key] = st.text_input(
                            key, type="password", value=default_val
                        )

                spacer, back_col, next_col = st.columns([0.58, 0.18, 0.24])
                with back_col:
                    back = st.form_submit_button("← Back", use_container_width=True)
                with next_col:
                    next_ = st.form_submit_button("Next →", use_container_width=True)

            if back:
                st.session_state.dc_step = 3
                st.rerun()
            if next_:
                st.session_state.dc_step = 5
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # STEP 5 — SUMMARY + COLLECT
    # =========================
    elif st.session_state.dc_step == 5:
        collector = st.session_state.get("collector")
        if collector is None:
            st.warning("Please complete earlier steps first.")
            if st.button("← Back"):
                st.session_state.dc_step = 1
                st.rerun()
            return

        query_options = collector.query_options()
        needed_keys = collector.auth()

        st.markdown('<div class="dc-wrap">', unsafe_allow_html=True)
        card = _step_card("5) Summary & Collect")

        with card:
            platform = (
                st.session_state.socialmedia.name
                if st.session_state.get("socialmedia")
                else "—"
            )
            method = (
                st.session_state.collector.__class__.__module__.split(".")[-1].title()
                if st.session_state.get("collector")
                else "—"
            )

            summary = f"""
| Field | Value |
| --- | --- |
| **Platform** | {platform} |
| **Collection method** | {method} |
"""
            for q in query_options:
                v = st.session_state.query_values.get(q, None)
                summary += f"| {q} | {('None' if v is None else v)} |\n"

            if needed_keys:
                summary += "| **Authentication** | |\n"
                for k in needed_keys:
                    v = st.session_state.query_values.get(k, None)
                    summary += f"| {k} | {'••••••••' if v else 'None'} |\n"

            st.markdown(summary)
            st.divider()

            spacer, back_col, collect_col = st.columns([0.58, 0.18, 0.24])
            with back_col:
                if st.button("← Back", use_container_width=True):
                    st.session_state.dc_step = 4
                    st.rerun()
            with collect_col:
                if st.button("Collect", type="primary", use_container_width=True):
                    st.session_state.results = None
                    gen = collector.collect_generator(**st.session_state.query_values)
                    progress_bar = st.progress(0)

                    for data in gen:
                        if isinstance(data, ProgressUpdate):
                            progress_bar.progress(data.progress, text=data.text)
                        else:
                            st.session_state.results = data
                            break

        st.markdown("</div>", unsafe_allow_html=True)

        # Results below
        if st.session_state.results is not None:
            st.subheader("Results")
            st.write("Found ", len(st.session_state.results), " results")
            df = pd.DataFrame(st.session_state.results)

            tabs = st.tabs(["Results", "Raw Data"])
            with tabs[0]:
                st.dataframe(df)
            with tabs[1]:
                st.write(st.session_state.results)

            if "keywords" in st.session_state.query_values:
                save_name = st.text_input(
                    "Save name",
                    value=f"{st.session_state.socialmedia.name}-{st.session_state.query_values['keywords']}",
                )
            else:
                save_name = st.text_input(
                    "Save name", value=f"{st.session_state.socialmedia.name}-"
                )

            do_download_images = st.checkbox("Download images with save")
            if do_download_images and "image_urls" not in df.columns:
                st.error(
                    "Cannot download images because the results do not have an 'image_urls' column"
                )

            if st.button("Save"):
                data_collection.utils.save_data(st.session_state.results, save_name)
                username = st.session_state["username"]
                st.success(f"Saved as data/{username}/{save_name}.csv")

                if do_download_images:
                    download_images_progress_bar = st.progress(0)
                    image_path = ""
                    for i in range(len(st.session_state.results)):
                        image_path = download_images(st.session_state.results, save_name, i)
                        download_images_progress_bar.progress(
                            i / len(st.session_state.results),
                            text=f"Downloading images ({i + 1}/{len(st.session_state.results)})",
                        )
                    st.success("Successfully downloaded all the images to '" + image_path + "'")