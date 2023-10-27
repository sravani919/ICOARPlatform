"""
Handles the retrieval and uploading of data into other tabs within the ICOAR project
Data will either already be stored internally in the data folder or can be uploaded by the user from their local machine
"""

import os

import pandas as pd
import streamlit as st

if "ldf" not in st.session_state:
    st.session_state.ldf = None  # Loaded data frame

if "column_replace" not in st.session_state:
    st.session_state.column_replace = None

if "managing_new_data" not in st.session_state:
    st.session_state.managing_new_data = False


def find_data_files(email, upload_new_data=False):
    # Get the list of files in the data folder
    data_files = os.listdir(f"data/{email}")

    # Keep only the files with the csv extension
    data_files = [filename for filename in data_files if filename.endswith(".csv")]

    data_files = [filename.split(".")[0] for filename in data_files]

    # If the user wants to upload new data
    if upload_new_data:
        # Add the option to upload new data
        data_files.append("Upload new data")

    return data_files


def rename_column():
    # Changing the column's name
    st.session_state.ldf.rename(columns={st.session_state.column_replace: "text"}, inplace=True)


def get_column_for_text():
    """
    Uses streamlit to preview the data so the user can choose which column to use for
    analysis
    """

    st.markdown("### Select column for text")

    st.session_state.column_replace = st.selectbox(
        "Select column for text", [None] + list(st.session_state.ldf.columns)
    )

    # Dropdown to show preview
    with st.expander("Preview of data"):
        st.dataframe(st.session_state.ldf.head())

    st.markdown("-------------------")


def data_upload_element(email, get_filepath_instead=False):
    """
    Has a drop-down menu to select data already stored in the data folder or upload new data
    :param get_filepath_instead: If True, returns the filepath instead of the data
    :return: The data selected in a pandas dataframe
    """

    if "managing_new_data" not in st.session_state:
        st.session_state.managing_new_data = False

    if "column_replace" not in st.session_state:
        st.session_state.column_replace = None

    st.session_state.data_files = find_data_files(email, upload_new_data=True)

    # Create the drop-down menu
    selected_data = st.selectbox("Select data", st.session_state.data_files)

    # If the user selects the upload option
    if selected_data == "Upload new data":
        # Upload the data
        uploaded_file = st.file_uploader("Upload data", type=["csv", "json"])

        if uploaded_file is not None and not st.session_state.managing_new_data:
            # Get the file extension
            extension = uploaded_file.name.split(".")[-1]

            # Read the file
            if extension == "csv":
                st.session_state.ldf = pd.read_csv(uploaded_file)
            elif extension == "json":
                st.session_state.ldf = pd.read_json(uploaded_file)
            else:
                raise ValueError(f"Unknown file type: {extension}")

            st.session_state.managing_new_data = True

        upload_button = st.button("Confirm upload")

        if "text" not in st.session_state.ldf.columns:
            if st.session_state.column_replace is not None:
                rename_column()
            else:
                st.warning("No column labeled 'text' found in the data, please select the column to use for text")
                get_column_for_text()

        if upload_button and "text" in st.session_state.ldf.columns:
            # Save the file
            st.session_state.ldf.to_csv(f"data/{uploaded_file.name}", index=False)
            st.session_state.managing_new_data = False  # Finished managing the new data
            st.session_state.data_files = find_data_files(upload_new_data=True)

            if get_filepath_instead:
                return f"data/{email}/{uploaded_file.name}"

            # Return the file
            return st.session_state.ldf

    # If the user selects a file already in the data folder
    elif selected_data != "":
        # Read the file
        st.session_state.ldf = pd.read_csv(f"data/{email}/{selected_data}.csv")

        if get_filepath_instead:
            return f"data/{selected_data}.csv"

        # Return the data
        return st.session_state.ldf

    # If the user selects no file
    else:
        return None
