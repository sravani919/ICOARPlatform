import glob

import pandas as pd
import streamlit as st

from data_preprocessing import options, preprocess
from tabs.Data_Collection.data_upload import data_upload_element


def data_preprocessing_tab():
    if "preprocessing_status" not in st.session_state:
        st.session_state.preprocessing_status = False
    if "filename" not in st.session_state:
        st.session_state.filename = None
    option_count = len(options)
    df_name = data_upload_element(get_filepath_instead=True)
    if df_name is not None:
        st.session_state.filename = df_name
        df = pd.read_csv(df_name)
        st.session_state.processed_df = None

        st.dataframe(df)
        columns = st.columns(3)
        selected_options = []
        for c in range(3):
            for r in range(c * option_count // 3, (c + 1) * option_count // 3):  # Puts the options in 3 columns
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
