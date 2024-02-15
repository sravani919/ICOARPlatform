import base64
import os

import requests
import streamlit as st

from gpt.image_sample_set import SampleSet, get_image_paths
from gpt.utils import key_directions


def encode_image(image_path):
    """
    Encodes an image to base64
    :param image_path: Path to the image
    :return: The base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# formats files with their labels for easy use with GPT API
def prepare_sample_images(sample_set):
    """
    Formats the sample images and their labels for use with the GPT API
    :param images_list: A DataFrame containing the images paths and their labels with columns "Image Path" and "Label"
    :return: A list of messages in the form of dictionaries that can be used with the GPT API
    """
    image_messages = []
    for i in range(len(sample_set.image_paths)):
        image_path = sample_set.image_paths[i]
        label = sample_set.labels[i]

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

    prompt = "Label the following image with one of the following:\n\n"
    for i, label in enumerate(labels):
        prompt += f"{i + 1}. {label}\n"
    prompt += "\nYou must give only the number corresponding to the label that best fits the image."
    prompt += "\nOnly give the number. For example: 1"
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

    if sample_set is not None:
        prepared_sample_images = prepare_sample_images(sample_set)
    else:
        prepared_sample_images = []
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
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            "max_tokens": 50,
        }
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
        custom labels. In the preset example below, we are labeling an image based on whether it is
        cyberbullying or not."""
    )

    key_directions()

    st.markdown("""1. Select the labels you want ChatGPT to use while labeling your images.""")

    row_input = st.columns(4)
    # username input at column 1
    with row_input[0]:
        number_of_labels = st.number_input("Number of labels", min_value=1, max_value=10, value=2)

    labels = ["Cyberbullying", "Noncyberbullying"]

    # Keeping the labels list size equal to number_of_labels
    if len(labels) > number_of_labels:
        labels = labels[:number_of_labels]
    elif len(labels) < number_of_labels:
        for i in range(number_of_labels - len(labels)):
            labels.append(f"Label {len(labels) + 1}")

    with st.container(border=True):
        columns = st.columns(2)
        for i in range(number_of_labels):
            with columns[i % 2]:
                labels[i] = st.text_input("", value=labels[i], placeholder="Label {}".format(i + 1))

    image_directories = ["gpt/examples/demo"]
    st.info("CWD: " + os.getcwd())

    # Iterate through the data/username
    my_data_directory = os.path.join("data/" + st.session_state["username"])
    for root, dirs, files in os.walk(my_data_directory):
        for name in dirs:
            image_directories.append(os.path.join(root, name))

    st.info("Image Directories: " + str(image_directories))

    """
    Get a sample set
    """
    if st.checkbox("Use a sample set to improve accuracy"):
        sample_set = SampleSet()
        # Dropdown to select any folder that has images in it
        st.subheader("Building a Sample Set")
        selected_folder = st.selectbox("Choose a folder with images", image_directories, key="sample_set")

        # If the selected folder does not have a labels.csv file, then it needs to be created
        if not os.path.isfile(os.path.join(selected_folder, "labels.csv")):
            sample_set.build(selected_folder, labels)
        else:
            sample_set.load(selected_folder)
    else:  # If the user does not want to use a sample set
        sample_set = None

    """
    Get the set of images to label / perform inference on
    """

    st.subheader("Select the images to label")
    print(image_directories)
    selected_folder = st.selectbox("Choose a folder with images", image_directories, key="inference_set")
    image_paths = get_image_paths(selected_folder)

    # Performing inference
    if st.button("Predict Labels"):
        st.session_state.predict = True
        st.session_state.filename_pred = selected_folder
        results = choice_label(api_key, image_paths, labels, sample_set)
        st.session_state.output = results
        st.success("Prediction completed", icon="✅")

        # Display the results, showing the image and the label
        st.markdown("### Results")
        for i, result in enumerate(results):
            st.image(image_paths[i], width=200)
            st.write(f"Label: {result}")
