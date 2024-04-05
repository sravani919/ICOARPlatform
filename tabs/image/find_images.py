import os

import streamlit as st


def image_folder_upload_element(email):
    """
    Has a drop-down menu to select folders already stored in the data folder
    :param email: The email of the user, used to know which data folder to use
    :return: The path of the selected folder
    """

    # Find all folders in the user's data directory
    folder_path = f"data/{email}"
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    folders.insert(0, "Upload new folder")

    # Create the drop-down menu
    selected_folder = st.selectbox("Select folder", folders)

    # If the user selects the upload option
    if selected_folder == "Upload new folder":
        # Upload the folder
        uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

        if uploaded_files is not None:
            # Let the user name a new folder
            new_folder_name = st.text_input("Enter a name for the new folder")
            confirm_button = st.button("Confirm Upload")
            # If the user has entered a new folder name
            if new_folder_name != "" and confirm_button:
                # Create the new folder
                new_folder_path = os.path.join(folder_path, new_folder_name)
                os.makedirs(new_folder_path, exist_ok=True)

                # Save the images to the new folder
                for uploaded_file in uploaded_files:
                    with open(os.path.join(new_folder_path, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success("Images uploaded successfully")
                # Return the path of the new folder
                return new_folder_path

    # If the user selects a folder already in the data folder
    elif selected_folder != "":
        # Return the path of the selected folder
        return os.path.join(folder_path, selected_folder)

    # If the user selects no folder
    else:
        return None
