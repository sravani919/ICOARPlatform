import os
import pandas as pd
import streamlit as st

from data_preprocessing import options, preprocess
from tabs.Data_Collection.data_upload import data_upload_element
from tabs.validation.validation import get_csv_string


# ---------- helper for AI Assistant ----------
def run_preprocess_file(input_path: str):
    """
    Used by the AI Assistant when you click 'Clean Text'.
    Cleans the CSV at input_path using ALL preprocessing steps, writes <name>__clean.csv,
    and returns (cleaned_path, cleaned_rows).
    """
    safe_path = input_path.replace("/icoar/", "/ICOAR/")

    if not os.path.exists(safe_path):
        raise FileNotFoundError(
            f"File not found at {safe_path}. "
            "This can happen if the app restarted or the path changed. "
            "Please hit Submit again to recollect before cleaning."
        )

    default_selected = [True] * len(options)

    ok, df_clean = preprocess(safe_path, default_selected)
    if not ok:
        raise RuntimeError("Preprocess pipeline returned failure flag")

    base_dir = os.path.dirname(safe_path)
    stem, _ext = os.path.splitext(os.path.basename(safe_path))
    cleaned_path = os.path.join(base_dir, f"{stem}__clean.csv")

    df_clean.to_csv(cleaned_path, index=False)
    return cleaned_path, len(df_clean)


# ---------- UI helpers (same style as Data Collection) ----------
def _pp_hero_header():
    steps = [
        (1, "Select data"),
        (2, "Choose cleaning"),
        (3, "Process & export"),
    ]
    active = int(st.session_state.get("pp_step", 1))

    chips_html = '<div class="pp-steps">' + "".join(
        [
            f'<div class="pp-chip {"active" if s==active else ""}">'
            f'<span class="pp-chip-num">{s}</span>'
            f'<span class="pp-chip-name">{name}</span>'
            f"</div>"
            for s, name in steps
        ]
    ) + "</div>"

    st.markdown(
        f"""
<div class="pp-hero-bleed">
  <div class="pp-hero-inner">
    <div class="pp-hero-title">Pre-processing</div>
    <div class="pp-hero-sub">Select a dataset, choose cleaning steps, then process and export.</div>
    {chips_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _pp_step_card(title: str):
    wrap = st.container()
    with wrap:
        st.markdown('<div class="pp-card-marker"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pp-step-title">{title}</div>', unsafe_allow_html=True)
    return wrap


# ---------- EXISTING UI: updated to match Data Collection format ----------
def data_preprocessing_tab():
    # Session defaults
    if "pp_step" not in st.session_state:
        st.session_state.pp_step = 1

    if "preprocessing_status" not in st.session_state:
        st.session_state.preprocessing_status = False
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None
    if "pp_selected_options" not in st.session_state:
        st.session_state.pp_selected_options = [False] * len(options)

    _pp_hero_header()

    # Wrap like DC
    st.markdown('<div class="pp-wrap">', unsafe_allow_html=True)

    # =========================
    # STEP 1 — SELECT DATA
    # =========================
    if st.session_state.pp_step == 1:
        card = _pp_step_card("1) Select data")

        with card:
            # This returns file path when get_filepath_instead=True
            df_name = data_upload_element(st.session_state.username, get_filepath_instead=True)

            if df_name is None:
                st.info("Choose a dataset to continue.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            st.session_state.filename = df_name

            try:
                df = pd.read_csv(df_name)
                st.dataframe(df, use_container_width=True, height=360)
            except Exception as e:
                st.error("Failed to read the selected CSV.")
                st.exception(e)
                st.markdown("</div>", unsafe_allow_html=True)
                return

            spacer, next_col = st.columns([0.76, 0.24])
            with next_col:
                if st.button("Next →", type="primary", use_container_width=True):
                    st.session_state.pp_step = 2
                    st.rerun()

    # =========================
    # STEP 2 — CHOOSE CLEANING
    # =========================
    elif st.session_state.pp_step == 2:
        card = _pp_step_card("2) Choose cleaning options")

        with card:
            if not st.session_state.filename:
                st.warning("No file selected. Please go back and select a dataset.")
                if st.button("← Back", use_container_width=False):
                    st.session_state.pp_step = 1
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                return

            # Clean 3-column layout with simple slicing (no skipping bugs)
            cols = st.columns(3)
            n = len(options)
            chunk = (n + 2) // 3  # ceil(n/3)

            new_selected = []
            for i, opt in enumerate(options):
                col_idx = min(i // chunk, 2)
                with cols[col_idx]:
                    checked = st.checkbox(
                        opt,
                        value=st.session_state.pp_selected_options[i],
                        key=f"pp_opt_{i}",
                    )
                new_selected.append(checked)

            st.session_state.pp_selected_options = new_selected

            spacer, back_col, next_col = st.columns([0.58, 0.18, 0.24])
            with back_col:
                if st.button("← Back", use_container_width=True):
                    st.session_state.pp_step = 1
                    st.rerun()
            with next_col:
                if st.button("Next →", type="primary", use_container_width=True):
                    st.session_state.pp_step = 3
                    st.rerun()

    # =========================
    # STEP 3 — PROCESS + EXPORT
    # =========================
    elif st.session_state.pp_step == 3:
        card = _pp_step_card("3) Process & export")

        with card:
            if not st.session_state.filename:
                st.warning("No file selected. Please go back and select a dataset.")
                if st.button("← Back", use_container_width=False):
                    st.session_state.pp_step = 1
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                return

            spacer, back_col, run_col = st.columns([0.58, 0.18, 0.24])
            with back_col:
                if st.button("← Back", use_container_width=True):
                    st.session_state.pp_step = 2
                    st.rerun()

            with run_col:
                if st.button("Process", type="primary", use_container_width=True):
                    with st.spinner("Pre-processing..."):
                        try:
                            ok, df_clean = preprocess(
                                st.session_state.filename,
                                st.session_state.pp_selected_options,
                            )
                            st.session_state.preprocessing_status = ok
                            st.session_state.processed_df = df_clean if ok else None
                        except Exception as e:
                            st.session_state.preprocessing_status = False
                            st.session_state.processed_df = None
                            st.error("Pre-processing failed.")
                            st.exception(e)

            if st.session_state.preprocessing_status and st.session_state.processed_df is not None:
                st.success("Pre-processing complete.")
                st.dataframe(st.session_state.processed_df, use_container_width=True, height=360)

                name = st.text_input(
                    "Enter a file name or leave as is to overwrite",
                    value=st.session_state.filename,
                )

                c1, c2 = st.columns([0.25, 0.75])
                with c1:
                    if st.button("Save", use_container_width=True):
                        st.session_state.processed_df.to_csv(name, index=False)
                        st.success("Saved to " + name)

                csv_data = get_csv_string(st.session_state.processed_df)
                st.download_button(
                    label="Download",
                    data=csv_data,
                    file_name=f"{name}",
                    mime="text/csv",
                    help="Click to download the cleaned CSV.",
                )
            else:
                st.caption("Click **Process** to generate the cleaned dataset.")

    st.markdown("</div>", unsafe_allow_html=True)