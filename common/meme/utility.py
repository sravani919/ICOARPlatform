import json
import os
import zipfile

import easyocr as ocr
import numpy as np
from PIL import Image

data_folder = "././data/"
memes_file_name = "memes.zip"
memes_folder_name = "memes"
valid_extensions = {".jpeg", ".jpg", ".png"}
test_json_file_path = data_folder + "test_seen2.jsonl"


def extract_file():
    if not (os.path.exists(data_folder + memes_folder_name) and os.path.isdir(data_folder + memes_folder_name)):
        os.makedirs(data_folder + memes_folder_name)
    zip_file_path = data_folder + memes_file_name
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(data_folder + memes_folder_name)
    os.remove(data_folder + memes_file_name)


def verify_files():
    file_list = os.listdir(data_folder + memes_folder_name)

    renamed_file_list = []
    for index, file_name in enumerate(file_list):
        if "placeholder" in file_name:
            continue
        _, extension = os.path.splitext(file_name)

        # Rename file names to the way the model needs them
        old_file_name = data_folder + memes_folder_name + "/" + file_name
        new_file_name = data_folder + memes_folder_name + "/" + str(index) + extension
        os.rename(old_file_name, new_file_name)

        renamed_file_list.append(str(index) + extension)

        if extension.lower() not in valid_extensions:
            return [renamed_file_list, False]

    return [renamed_file_list, True]


def clear_files():
    for file in os.listdir(data_folder + memes_folder_name):
        if "placeholder" in file:
            continue
        os.remove(data_folder + memes_folder_name + "/" + file)


def load_model():
    reader = ocr.Reader(["en"], model_storage_directory=".")
    return reader


def get_caption(image):
    reader = load_model()
    input_image = Image.open(image)

    result = reader.readtext(np.array(input_image))
    result_text = ""
    for text in result:
        result_text += " " + text[1]

    return result_text


data = {"id": 1, "img": "", "text": ""}


def generate_json(file_list, caption_list):
    json_list = ""
    for i in range(len(file_list)):
        if "placeholder" in file_list[i]:
            continue
        id, extension = os.path.splitext(file_list[i])
        id = int(id)
        data["id"] = id
        data["img"] = "memes/" + file_list[i]
        data["text"] = caption_list[i]
        if i == 0:
            json_list = json_list + json.dumps(data)
        else:
            json_list = json_list + "\n" + json.dumps(data)

    with open(test_json_file_path, "w") as file:
        json_list = json_list + "\n" + json.dumps(data)
    with open(test_json_file_path, "w") as file:
        file.write(json_list)
