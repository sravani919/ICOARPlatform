import os

import pandas as pd
import streamlit as st
from huggingface_hub import HfApi
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification

from . import find_images


def fetch_models_from_hf(search_text):
    # Or configure a HfApi client
    hf_api = HfApi(
        endpoint="https://huggingface.co",  # Can be a Private Hub endpoint.
        token=st.secrets.api_token.hf,  # Token is not persisted on the machine.
    )

    print("Fetching model list from hugging face...")
    models = list(hf_api.list_models(filter="image-classification", search=search_text))
    # sort models by downloads to get the user higher quality models.
    models.sort(key=lambda model: model.downloads, reverse=True)

    model_list = []

    for model in models:
        model_list.append(model.modelId)

    return model_list


def huggingface_image_analysis():
    if "img_disabled" not in st.session_state:
        st.session_state.img_disabled = True
    if "img_model_list" not in st.session_state:
        st.session_state.img_model_list = []
    if "img_predict" not in st.session_state:
        st.session_state.img_predict = False
    if "image_folder" not in st.session_state:
        st.session_state.image_folder = None
    multi = """:bulb:
                    Select a folder that contains images you want to analyze. You can also select images to
                    construct a new folder in the data directory if you don't have one already"""
    st.markdown(multi)
    st.session_state.image_folder = find_images.image_folder_upload_element(st.session_state.username)
    if st.session_state.image_folder is None:
        st.warning("Please select a folder to proceed")
        return
    else:
        image_folder = st.session_state.image_folder

    MODEL = ""
    multi = """:bulb: **Steps** -
                \n 1. Input your Huggingface API token in the secrets.toml file. If you don't have
                one, you can get one [here](https://huggingface.co/settings/tokens)
                \n2. Input the name of the model you want to use. You can also just input a related keyword if you
                are not sure of which one to use.
                \n3. After that, click on the select/search button and verify that we are using the right model. If you
                searched by keyword, select the model you want to use from the list that appears. If you want to see
                more information about your selected model, there will be an expander below with a link to the model's
                page. These pages usually have a demo on the right side of the page, so that you can test the model
                before using it."""
    st.markdown(multi)
    choice = st.radio("Select an option:", ["Use a specific model", "Search by keyword"])
    if choice == "Use a specific model":
        model_name = st.text_input("Enter the name of the huggingface model (not the URL!):")
        model_button = st.button("Select")
        if model_button:
            st.session_state.img_model_list = [model_name]
            st.session_state.img_disabled = False
        if st.session_state.img_model_list:
            MODEL = st.radio("Model: ", [st.session_state.img_model_list[0]])
            st.markdown("-------------------")
            st.write(f"Verify we have the right model: [{MODEL}](https://huggingface.co/{MODEL})")

    if choice == "Search by keyword":
        search_text = st.text_input("Enter model name")
        search_button = st.button("Search")
        if search_button:
            st.session_state.img_model_list = fetch_models_from_hf(search_text)
            search_button = False
            st.session_state.img_disabled = False
        if st.session_state.img_model_list:
            MODEL = st.radio(
                "Top Three Models:",
                st.session_state.img_model_list[:3],
            )
            # display the rest of the models if the user wants to see more
            if st.checkbox("Show more"):
                MODEL = st.radio(
                    "All Results",
                    st.session_state.img_model_list[3:],
                )
        if MODEL:
            with st.expander("Model Details"):
                model_url = f"https://huggingface.co/{MODEL}"
                st.write(f"Model URL: [{MODEL}]({model_url})")

    if st.session_state.img_disabled:
        st.warning("Please select a model to proceed")
        return

    predict_button = st.button("Predict")
    if predict_button:
        st.session_state.img_predict = True
    st.markdown("-------------------")
    if st.session_state.img_predict:
        with st.spinner("In progress..."):
            try:
                model = AutoModelForImageClassification.from_pretrained(MODEL)
            except Exception as e:
                st.error(f"Error loading model: {e}. Please try a different model.")
                return

            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            df = pd.DataFrame(columns=["Image Name", "Label"])
            all_files = os.listdir(image_folder)
            image_files = [file for file in all_files if file.endswith((".jpg", ".jpeg", ".png"))]
            for image_file in image_files:
                full_path = os.path.join(image_folder, image_file)
                image = Image.open(full_path)
                image = image.convert("RGB")
                image_tensor = transform(image).unsqueeze(0)
                outputs = model(image_tensor)
                predicted_label = outputs.logits.argmax().item()
                config = model.config
                if hasattr(config, "id2label"):
                    predicted_label = config.id2label[predicted_label]
                else:
                    predicted_label = predicted_label

                new_row = pd.DataFrame({"Image Name": [os.path.basename(image_file)], "Label": [predicted_label]})
                df = pd.concat([df, new_row], ignore_index=True)

        st.success("Prediction complete!")
        if st.session_state.img_predict:
            file_name = st.text_input("Enter the name of the file to save the predictions to:")
            save_button = st.button("Save Predictions")
            if save_button:
                file_path = f"data/{st.session_state.username}/{file_name}.csv"
                df.to_csv(file_path, index=False)
                st.success(f"Predictions saved to {file_path}")
                st.session_state.img_predict = False

                # Add download button
                with open(file_path, "rb") as file:
                    st.download_button(
                        label="Download Predictions",
                        data=file,
                        file_name=f"{file_name}.csv",
                        mime="text/csv",
                    )
