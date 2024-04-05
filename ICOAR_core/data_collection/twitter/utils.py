import os

import toml

from ..utils import download_image


def save_tweet_images(tweet, images_path):
    """
    Takes in a single tweet dictionary and then downloads the images off the internet
    and saves them to the given file path
    :param tweet: The tweet dictionary
    :param images_path: The file path to save the images to

    """
    # Checking if the tweet has any images
    urls = tweet["image_urls"]
    if urls is None or len(urls) == 0:
        return []

    id = tweet["id"]
    tweet_folder = images_path + "/" + str(id)  # Folder for all the images from this tweet
    # Creating a folder for the images if it doesn't already exist
    if not os.path.exists(tweet_folder):
        os.makedirs(tweet_folder)

    # Downloading and saving the images
    for i in range(len(urls)):
        image = download_image(urls[i])
        if image is not None:
            image.save(f"{tweet_folder}/{i}.jpg")


def save_images(tweets, filename, i):
    """
    Saves the images from the given tweet to the given file path
    Uses an index i so a loading bar can be implemented on the front end while still having all the code here
    :param tweets: The full list of tweet dictionaries
    :param filename: The filename being used to save the tweets
    :param i: The index of the tweet to save the images from (allows for easy iteration for a loading bar)
    :return: The file path to the folder containing the images from the tweet
    """
    images_path = f"data/{filename}_images"
    if i == 0:
        if not os.path.exists(images_path):
            os.makedirs(images_path)
    save_tweet_images(tweets[i], images_path)

    return images_path


def load_accounts_txt(path_to_txt):
    """
    Loads the accounts from a txt file and appends them to the streamlit secrets.toml file
    :param path_to_txt: The path to the txt file
    """
    with open(path_to_txt, "r") as file:
        accounts = file.readlines()

    current_accounts = toml.load(".streamlit/secrets.toml")["twitter"]["accounts"]
    for account in accounts:
        account = account.strip()
        if account not in current_accounts:
            current_accounts += "," + account

    # Replace the current accounts in the secrets.toml file
    with open(".streamlit/secrets.toml", "r") as file:
        data = toml.load(file)
        data["twitter"]["accounts"] = current_accounts
    with open(".streamlit/secrets.toml", "w") as file:
        toml.dump(data, file)


if __name__ == "__main__":
    load_accounts_txt("/Users/ethan/Downloads/order5202644.txt")
