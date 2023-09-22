"""
Handles the retrieval and uploading of data into other tabs within the ICOAR project
Data will either already be stored internally in the data folder or can be uploaded by the user from their local machine
"""

import os

import pandas as pd
import streamlit as st


def find_data_files(upload_new_data=False):
    # Get the list of files in the data folder
    data_files = os.listdir("data")

    # Keep only the files with the csv extension
    data_files = [filename for filename in data_files if filename.endswith(".csv")]

    data_files = [filename.split(".")[0] for filename in data_files]

    # If the user wants to upload new data
    if upload_new_data:
        # Add the option to upload new data
        data_files.append("Upload new data")

    return data_files


def data_upload_element(get_filepath_instead=False):
    """
    Has a drop-down menu to select data already stored in the data folder or upload new data
    :param get_filepath_instead: If True, returns the filepath instead of the data
    :return: The data selected in a pandas dataframe
    """

    st.session_state.data_files = find_data_files(upload_new_data=True)

    # Create the drop-down menu
    selected_data = st.selectbox("Select data", st.session_state.data_files)

    # If the user selects the upload option
    if selected_data == "Upload new data":
        # Upload the data
        uploaded_file = st.file_uploader("Upload data", type=["csv", "json"])

        if uploaded_file is not None:
            # Get the file extension
            extension = uploaded_file.name.split(".")[-1]

            # Read the file
            if extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif extension == "json":
                df = pd.read_json(uploaded_file)
            else:
                raise ValueError(f"Unknown file type: {extension}")

            # Save the file
            df.to_csv(f"data/{uploaded_file.name}", index=False)

            st.session_state.data_files = find_data_files(upload_new_data=True)

            if get_filepath_instead:
                return uploaded_file.name

            # Return the file
            return df
        else:
            return None

    # If the user selects a file already in the data folder
    elif selected_data != "":
        # Read the file
        df = pd.read_csv(f"data/{selected_data}.csv")

        if get_filepath_instead:
            return f"data/{selected_data}.csv"

        # Return the data
        return df

    # If the user selects no file
    else:
        return None
