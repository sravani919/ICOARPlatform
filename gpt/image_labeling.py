import base64
import os
import shutil
import time

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit import secrets

from gpt.utils import key_directions


class SampleSet:
    """
    A class to represent a sample set of images and their labels
    Includes a list of image paths and a list of corresponding labels

    Can be saved and loaded to a folder
    """

    def __init__(self):
        self.image_paths = None
        self.labels = None

    def load(self, folder_path):
        """
        Loads the sample set from a folder
        The folder should have all images and then a csv file with a column "Image Path" and a column "Label"
        :param folder_path: Path to the folder
        """

        # Load the csv file
        csv_file = os.path.join(folder_path, "labels.csv")
        labels_df = pd.read_csv(csv_file)

        self.image_paths = labels_df["Image Path"].tolist()
        self.labels = labels_df["Label"].tolist()

    def save(self, folder_path):
        """
        Saves the sample set to a folder
        Saves the images and a csv file with the image paths and labels
        The folder should have all images and then a csv file with a column "Image Path" and a column "Label"
        :param folder_path: Path to the folder
        """

        # Copy the images to here, and use their new paths in the csv file
        new_paths = []
        for i, image_path in enumerate(self.image_paths):
            base_name = os.path.basename(image_path)
            new_path = os.path.join(folder_path, base_name)
            new_paths.append(new_path)
            # Copy the image using shutil.copyfile(image_path, new_path)
            shutil.copyfile(image_path, new_path)

        # Write the csv file
        labels_df = pd.DataFrame({"Image Path": new_paths, "Label": self.labels})
        csv_file = os.path.join(folder_path, "labels.csv")
        labels_df.to_csv(csv_file, index=False)

    def add_image(self, image_path, label):
        """
        Adds an image to the sample set
        :param image_path: Path to the image
        :param label: Label for the image
        """
        self.image_paths.append(image_path)
        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return self.image_paths[index], self.labels[index]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# formats files with their labels for easy use with GPT API
def prepare_sample_images(images_list):
    """
    Formats the sample images and their labels for use with the GPT API
    :param images_list: A DataFrame containing the images paths and their labels with columns "Image Path" and "Label"
    :return: A list of messages in the form of dictionaries that can be used with the GPT API
    """
    image_messages = []
    for _, row in images_list.iterrows():
        image_path = row["Image Path"]
        label = row["Label"]

        base64_image = encode_image(image_path)

        image_message = {
            "type": "text",
            "text": f"Label for image: {label}",
        }
        image_url_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }

        image_messages.append(image_message)
        image_messages.append(image_url_message)

    return image_messages


def labels_to_prompt(labels: list[str]) -> str:
    """
    Converts the labels to a prompt for the GPT API

    For examaple if the lables are ["not cyberbullying", "cyberbullying"], the prompt will be:
    "Label the following image with either

    1. cyberbullying
    2. not cyberbullying.

    Give only the number."

    :param labels: List of labels
    :return: A prompt for the GPT API
    """

    prompt = "Label the following image with either\n\n"
    for i, label in enumerate(labels):
        prompt += f"{i + 1}. {label}\n"
    prompt += "\nYou must give only the number corresponding to the label that best fits the image."
    return prompt


def choice_label(api_key, inference_image_paths, labels, sample_set) -> list[str]:
    """
    Using the labels list, a sample set of images, and the OpenAI API key, this function will return the label that
    the GPT-3 model predicts for each image in the inference_image_paths list.

    :param api_key: OpenAI API key
    :param inference_image_paths: List of paths to images where labels need to be added
    :param labels: List of potential labels for the images where an image can only be assigned one label
    :param sample_set: A dataframe containing the image names and their labels with columns "Image Name" and "Label"
    :return: Which label got assigned to each image of corresponding index in image_paths
    """

    # Generate a prompt for labeling images

    prepared_sample_images = prepare_sample_images(sample_set)
    label_prompt = labels_to_prompt(labels)

    results = []

    for image in inference_image_paths:
        base64_image = encode_image(image)
        # Call GPT Vision API for image labeling

        # Call GPT Vision API for image labeling
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *prepared_sample_images,
                        {
                            "type": "text",
                            "text": "Use the above examples of images and their labels to answer the following prompt:",
                        },
                        {"type": "text", "text": label_prompt},
                        {"type": "text", "text": "Here is the image to label based on the prompt:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            "max_tokens": 50,
        }
        print("Payload: ", payload)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        data = response.json()
        content_value = data["choices"][0]["message"]["content"]
        results.append(content_value)

    return results


def image_labeling(api_key):
    if "predict" not in st.session_state:
        st.session_state.predict = False
    st.markdown(
        f"""
                <style>
                    div[data-testid="stButton"] {{
                        margin-top: {8}px;
                        margin-bottom: {6}px;
                    }}
                    div[data-testid="stMarkdownContainer"] {{
                        margin-top: {2}px;
                        margin-bottom: {2}px;
                    }}
                </style>
                """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """In this section, you have the opportunity to utilize ChatGPT to annotate images according to your
        custom labels. In the preset example below, we are labeling an image based on  whether it is
        cyberbullying or not."""
    )

    key_directions()

    st.markdown(
        """**2. Input Image(s):** Choose the folder that contains the images you want to use as a sample,
        and label those images. The program is looking specifically in the data folder of the project.
        The demo has some preset images that are already labeled."""
    )
    data_directory = "./data/"

    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the demo directory
    default_folder = os.path.join(current_file_directory, "examples/demo")
    relative_demo_path = os.path.relpath(default_folder, current_file_directory)
    # Initialize an empty list to store subdirectories with full paths
    subdirectories = [relative_demo_path]

    # Iterate over the list of directories
    for d in os.listdir(data_directory):
        full_path = os.path.join(data_directory, d)

        # Check if the item is a directory
        if os.path.isdir(full_path):
            # Append the full path to the subdirectories list
            subdirectories.append(full_path)

    # Allow the user to choose a folder with demo
    selected_folder_path = st.selectbox("Select a folder that contains images", subdirectories, key="unique_key_1")

    # Get the full path for the selected folder
    option = selected_folder_path
    if option == relative_demo_path:
        option = default_folder

    images_list = [
        image for image in os.listdir(option) if image.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]

    # Create a DataFrame to store labels for each image
    image_labels_df = pd.DataFrame(columns=["Image Name", "Label"])
    image_labels_df["Image Name"] = images_list

    if option == default_folder:
        image_labels_df.loc[image_labels_df["Image Name"].isin(["CB 1.png", "CB 2.png"]), "Label"] = "1"
        image_labels_df.loc[image_labels_df["Image Name"].isin(["NCB 1.png", "NCB 2.png"]), "Label"] = "0"

    # Use st.data_editor to allow users to label images in a table-like interface
    edited_labels = st.data_editor(key="labeling_editor", data=image_labels_df, num_rows="dynamic", width=1000)

    st.markdown(
        """**3. Edit the prompt (Optional):** You can modify the prompt that instructs ChatGPT. The choice of
        prompt greatly influences the generated responses and the quality of the annotation. Ensure that your
        edits are clear and relevant to the task."""
    )

    # Add a text area for prompt editing
    st.subheader("Edit Prompt:")
    prompt = st.text_area(
        "Customize your prompt:",
        value="Q: Label this image as cyberbullying based on the sample"
        " provided, answering 0 for 'No' and 1 for 'Yes' Give "
        "only the number.",
        height=100,
    )

    if "openai" not in secrets:
        st.error("Enter your API key in secrets.toml under [openai].")
        return

    st.markdown(
        """**4. Test with a Single Example (Optional):** Before labeling your entire dataset, it's a good practice
        to test ChatGPT performance with a single example to see if the predictions are accurate and align with your
        instructions."""
    )
    # allow the user to choose a single image to predict, first selecting a folder and then an image
    selected_folder_path2 = st.selectbox("Select a folder that contains images", subdirectories, key="unique_key_2")

    # Get the full path for the selected folder for the demo image
    option2 = selected_folder_path2
    if option2 == relative_demo_path:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        option2 = os.path.join(current_file_directory, "examples/demo_test")

    image_files = [f for f in os.listdir(option2) if os.path.isfile(os.path.join(option2, f))]

    image_test_name = st.selectbox("Select an image to predict", image_files)

    image_test_path = os.path.join(option2, image_test_name)

    image = Image.open(image_test_path)
    image = image.resize((200, 200))
    st.image(image, caption="Image to predict")

    base64_image = encode_image(image_test_path)

    if st.button("Predict Label", disabled="openai" not in secrets):
        prepared_images = prepare_sample_images(edited_labels, option)
        # Call GPT Vision API for image labeling
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *prepared_images,
                        {
                            "type": "text",
                            "text": "Use these images and their labels to answer the following prompt:",
                        },
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "Here is the image to label based on the prompt:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            "max_tokens": 50,
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        data = response.json()
        content_value = data["choices"][0]["message"]["content"]
        st.markdown(f"**Predicted Label:** {content_value}")

    st.markdown(
        """**4. Test with a Multiple Images:** Select a folder of images to label and then
        click the button to predict the labels for all the images in the folder."""
    )

    label_subdirectories = [x for x in subdirectories if x != relative_demo_path]
    label_folder_path = st.selectbox(
        "Select a folder that contains images", label_subdirectories, key="unique_" "key_3"
    )

    # Get the full path for the selected folder
    option_label = label_folder_path

    if st.button("Predict Labels", disabled="openai" not in secrets):
        st.session_state.predict = True
        st.session_state.filename_pred = option

        prepared_images = prepare_sample_images(edited_labels, option)
        image_files = [f for f in os.listdir(option_label) if os.path.isfile(os.path.join(option_label, f))]
        # Create a DataFrame with the names of the files
        df = pd.DataFrame({"Image Name": image_files})

        total_rows = df.shape[0]
        progress_bar = st.empty()

        for index, row in df.iterrows():
            image_name = row["Image Name"]  # Replace with the column containing image paths
            image_path = os.path.join(option_label, image_name)

            if not os.path.isfile(image_path) or not image_name.lower().endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".bmp")
            ):
                continue

            base64_image = encode_image(image_path)

            # Call GPT Vision API for image labeling
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            *prepared_images,
                            {
                                "type": "text",
                                "text": "Use these images and their labels to answer the following prompt:",
                            },
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "Here is the image to label based on the prompt:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                "max_tokens": 50,
            }
            try:
                time.sleep(10)
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()
                content_value = data["choices"][0]["message"]["content"]

                # Update DataFrame with predicted label
                df.loc[index, "predicted_label"] = content_value
                progress = (index + 1) / total_rows
                progress_bar.progress(progress, text=f"Predicting image labels: {progress * 100:.2f}% complete")
            except requests.exceptions.RequestException as e:
                st.error(
                    f"An error occurred while predicting the label for image {image_name}: {e}. Please retry "
                    f"the prediction."
                )

        st.dataframe(df)
        progress_bar.empty()

        st.session_state.output = df
        st.success("Prediction completed", icon="✅")

    if st.session_state.predict:
        filename = st.text_input("Enter file name to save predicted data")
        save = st.button("Save File")
        username = st.session_state["username"]
        if save:
            if not os.path.exists("predicted"):
                os.makedirs("predicted")
            os.makedirs(f"""predicted/{username}""")
            file_path = f"predicted/{username}/{filename}.csv"
            st.session_state.output.to_csv(file_path, index=False)

            st.session_state.predict = False
            st.success("Saved to '" + file_path + "'")
