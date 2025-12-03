import os
import pandas as pd
import streamlit as st

from data_preprocessing import options, preprocess
from tabs.Data_Collection.data_upload import data_upload_element
from tabs.validation.validation import get_csv_string

# ---------- NEW: helper for AI Assistant ----------

def run_preprocess_file(input_path: str):
    """
    Used by the AI Assistant when you click 'Clean Text'.
    Cleans the CSV at input_path using ALL preprocessing steps, writes <name>__clean.csv,
    and returns (cleaned_path, cleaned_rows).
    """
    # --- normalize path case just in case ---
    safe_path = input_path.replace("/icoar/", "/ICOAR/")

    # 1. make sure the file actually exists
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
    base_name = os.path.basename(safe_path)
    stem, ext = os.path.splitext(base_name)
    cleaned_name = f"{stem}__clean.csv"
    cleaned_path = os.path.join(base_dir, cleaned_name)

    df_clean.to_csv(cleaned_path, index=False)

    return cleaned_path, len(df_clean)




# ---------- EXISTING UI: manual preprocessing tab ----------

def data_preprocessing_tab():
    if "preprocessing_status" not in st.session_state:
        st.session_state.preprocessing_status = False
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None
    option_count = len(options)
    df_name = data_upload_element(st.session_state.username, get_filepath_instead=True)
    if df_name is not None:
        st.session_state.filename = df_name
        df = pd.read_csv(df_name)
        # st.session_state.processed_df = None

        st.dataframe(df)
        columns = st.columns(3)
        selected_options = []
        for c in range(3):
            for r in range(
                c * option_count // 3, min((c + 1) * (option_count // 3), option_count)
            ):  # Puts the options in 3 columns
                selected_options.append(columns[c].checkbox(options[r], value=False))
        remainder = option_count % 3
        if remainder != 0:
            for c in range(remainder):
                for r in range(option_count - remainder, option_count):
                    selected_options.append(columns[c].checkbox(options[r], value=False))

        if st.button("Process"):
            st.session_state.preprocessing_status, st.session_state.processed_df = preprocess(
                st.session_state.filename, selected_options
            )

    if st.session_state.preprocessing_status:
        st.dataframe(st.session_state.processed_df)

        name = st.text_input("Enter a file name or leave as is to overwrite", value=st.session_state.filename)

        if st.button("Save"):
            st.session_state.processed_df.to_csv(name, index=False)
            st.success("Saved to " + name)
        csv_data = get_csv_string(st.session_state.processed_df)
        st.download_button(
            label="Download",
            data=csv_data,
            file_name=f"{name}",
            mime="text/csv",
            help="Click to download the CSV file with predicted data.",
        )
