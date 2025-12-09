import base64
import os
import pickle
import tempfile
import zipfile

import pandas as pd
import requests
import streamlit as st

from gpt.image_sample_set import SampleSet, get_image_paths
from gpt.utils import key_directions


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def prepare_sample_images(sample_set):
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


def labels_to_prompt(labels, context=""):
    prompt = "Label the following image with one of the following:\n\n"
    for i, label in enumerate(labels):
        prompt += f"{i + 1}. {label}\n"
    prompt += "\nYou must give only the number corresponding to the label that best fits the image."
    prompt += "\nOnly give the number. For example: 1"
    if context != "":
        prompt += "\nHere is some additional context for this task:" + context
    return prompt


def choice_label(api_key, inference_image_paths, labels, sample_set, context):
    if sample_set is not None:
        prepared_sample_images = prepare_sample_images(sample_set)
    else:
        prepared_sample_images = []
    label_prompt = labels_to_prompt(labels, context)

    results = []
    progress_bar = st.empty()

    for image in inference_image_paths:
        base64_image = encode_image(image)
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
                            "text": "Use the above examples of images and their labels to complete the following task:",
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
        progress = (inference_image_paths.index(image) + 1) / len(inference_image_paths)
        progress_bar.progress(progress, text=f"Predicting: {progress * 100:.2f}% complete")
    return results


def handle_zip_upload(uploaded_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir


def prompt_history():
    prompt_mp = {}
    if os.path.exists("././data/prompts.pickle"):
        with open("././data/prompts.pickle", "rb") as f:
            prompt_mp = pickle.load(f)

    selected_prompt = st.selectbox("Select prompt:", prompt_mp.keys())
    if selected_prompt:
        st.text_area("", prompt_mp[selected_prompt], height=200)


def image_labeling(api_key):
    """
    Image labeling UI.
    api_key can be passed from BasePage, but the user can also provide their own key.
    """
    if "predict2" not in st.session_state:
        st.session_state.predict2 = False

    # Default values to avoid UnboundLocalError
    sample_set = None
    temp_dir = None
    image_paths = []
    image_names = []

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

    st.markdown(
        """***2. Select Labels:*** Select the labels you want ChatGPT to use while labeling your images.
    You can also optionally provide some context for the labels you provided (i.e a definition for your labels)"""
    )

    row_input = st.columns(4)
    with row_input[0]:
        number_of_labels = st.number_input("Number of labels", min_value=1, max_value=10, value=2)

    labels = ["Cyberbullying", "Noncyberbullying"]

    # Keeping the labels list size equal to number_of_labels
    if len(labels) > number_of_labels:
        labels = labels[:number_of_labels]
    elif len(labels) < number_of_labels:
        for i in range(number_of_labels - len(labels)):
            labels.append(f"Label {len(labels) + 1}")

    with st.container():
        columns = st.columns(2)
        for i in range(number_of_labels):
            with columns[i % 2]:
                labels[i] = st.text_input("", value=labels[i], placeholder=f"Label {i + 1}")

    if st.checkbox("Provide context for the labels"):
        context = st.text_area(
            "Context",
            """Cyberbullying images typically have the following features:
        - Subjects oriented towards the viewer, conveying a threatening or confrontational stance.
        - Facial expressions showing mocking joy, ridicule, or taunting.
        - Specific hand gestures like the middle finger, "loser" sign, and thumbs down.
        - Presence of threatening objects such as guns and knives used to intimidate the viewer.""",
        )
    else:
        context = ""

    # --- Optional sample set to improve accuracy ---
    st.markdown(
        """***3. Use a labeled sample set*** (optional) Use a sample set of images to give to ChatGPT for use
    in labeling. This is looking for a folder of images with a labels.csv file in it, formatted with the first column
    being "Image Path" and the second column being "Label" for that image. If there is no labels.csv file provided,
    you will be prompted to label each image in the folder."""
    )
    use_sample_set = st.checkbox("Use a sample set to improve accuracy")
    if use_sample_set:
        sample_set = SampleSet()
        uploaded_zip_sample = st.file_uploader("Upload a ZIP file with images", type="zip", key="sample_set_zip")

        if uploaded_zip_sample is not None:
            temp_dir_sample = handle_zip_upload(uploaded_zip_sample)

            if not os.path.isfile(os.path.join(temp_dir_sample, "labels.csv")):
                sample_set.build(temp_dir_sample, labels)
            else:
                sample_set.load(temp_dir_sample)
        else:
            sample_set = None

    label_prompt = labels_to_prompt(labels, context)
    sub_cols = st.columns(2)
    with sub_cols[0]:
        edited_prompt = st.text_area("Prompt Preview", label_prompt, height=200)

        prompt_name = st.text_input("Save Prompt (Optional):")
        if st.button("Save Prompt"):
            prompts = {}
            prompt_file_path = "././data/prompts.pickle"

            if os.path.exists(prompt_file_path):
                with open(prompt_file_path, "rb") as f:
                    prompts = pickle.load(f)

            prompts[prompt_name] = edited_prompt
            with open(prompt_file_path, "wb") as f:
                pickle.dump(prompts, f)
    with sub_cols[1]:
        load_prompt_history = st.toggle("Load Prompt History")

        if load_prompt_history:
            prompt_history()

    st.subheader("Select the images to label")
    uploaded_zip = st.file_uploader("Upload a ZIP file with images", type="zip", key="inference_set_zip")

    if uploaded_zip is not None:
        temp_dir = handle_zip_upload(uploaded_zip)
        image_paths = get_image_paths(temp_dir)
        image_names = [os.path.basename(path) for path in image_paths]

    # --- API KEY HANDLING ---
    # Allow user to enter GPT-4 key (even if api_key argument is None)
    user_key = st.text_input(
        "Enter your [GPT-4 API Key](https://platform.openai.com/api-keys)",
        type="password",
    )
    if user_key:
        api_key = user_key

    if not api_key:
        st.info("Please enter your GPT-4 API key above to enable image labeling.")
        return
    # --- END API KEY HANDLING ---

    if st.button("Predict Labels"):
        if not image_paths:
            st.error("Please upload a ZIP file with images before predicting labels.")
            return

        st.session_state.predict2 = True
        st.session_state.filename_pred = temp_dir
        results = choice_label(api_key, image_paths, labels, sample_set, context)
        st.success("Prediction completed", icon="✅")

        # Display the results, showing the image and the label
        st.markdown("### Results")
        for i, result in enumerate(results):
            st.image(image_paths[i], width=200)
            st.write(f"Label: {result}")

        results_df = {"Image Name": image_names, "Label": results}
        results_df = pd.DataFrame(results_df)
        st.session_state.output = results_df

    # Save the results to a file
    if st.session_state.predict2:
        filename = st.text_input("Enter file name to save predicted data")
        save = st.button("Save File")
        username = st.session_state["username"]
        if save:
            if not os.path.exists("predicted"):
                os.makedirs("predicted")
            if not os.path.exists(f"predicted/{username}"):
                os.makedirs(f"predicted/{username}")
            file_path = f"predicted/{username}/{filename}.csv"
            st.session_state.output.to_csv(file_path, index=False)

            st.session_state.predict2 = False
            st.success("Saved to '" + file_path + "'")

            with open(file_path, "rb") as file:
                st.download_button(
                    label="Download Predicted Data",
                    data=file,
                    file_name=f"{filename}.csv",
                    mime="text/csv",
                )
