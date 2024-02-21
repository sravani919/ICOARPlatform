"""
Contains the SampleSet class and a the setup for a streamlit component set to be used to allow the user
to build a sample set given a folder of images.
The user will be able to assign a label to each image.
"""

import os
import shutil

import pandas as pd
import streamlit as st


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
            try:
                shutil.copyfile(image_path, new_path)
            except shutil.SameFileError:
                pass

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

    def build(self, folder_path, labels: list[str]):
        """
        Builds a sample set from a folder of images
        """
        st.info("labels.csv is missing, please label each image below.")
        self.image_paths = get_image_paths(folder_path)

        # Shows the image and then a dropdown box side by side for each
        with st.container(height=400, border=True):
            # Make columns where the first column is more narrow
            # and the second column is wider
            columns = st.columns([0.10, 0.20, 0.70])
            for i, image_path in enumerate(self.image_paths):
                with columns[0]:
                    st.image(image_path, width=100)
                    st.write("##")
                with columns[1]:
                    label = st.selectbox(f"Label for image {i}", labels)
                    st.write("#")
                    st.write("###")
                    if self.labels is not None:
                        self.labels.append(label)
                    else:
                        self.labels = [label]

            if st.button("Save Labels"):
                self.save(folder_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return self.image_paths[index], self.labels[index]


def get_image_paths(folder_path: str) -> list[str]:
    """
    Returns a list of all the absolute image paths of all the images in the folder
    :param folder_path: Path to the folder that should be searched for images
    :return: A list of all the absolute image paths
    """
    image_paths = []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            image_path = os.path.join(folder_path, file)
            image_paths.append(image_path)
    return image_paths
