import time

from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..utils import BaseDataCollector, ProgressUpdate


def format_comments(data):
    """
    Formats the given list of comments into a list of dictionaries
    :param data: Tuple with username and comment
    :return: A list of dictionaries containing the formatted comment data
    """
    formatted_comments = []
    for username, comment in data:
        formatted_comments.append(
            {
                "username": username,
                "comment": comment,
            }
        )
    return formatted_comments


def extract_comments(video_url, count):
    dataset = set()
    with Chrome() as driver:
        yield ProgressUpdate(0, "Starting YouTube scraper")
        wait = WebDriverWait(driver, 15)

        driver_good = False
        for attempt in range(10):
            yield ProgressUpdate(attempt / 10, f"Loading video webpage... attempt {attempt + 1}")
            try:
                driver.get(video_url)
                driver_good = True
                break
            except Exception as e:
                yield ProgressUpdate(attempt / 10, f"Error loading video webpage: {e}, retrying...")

        if not driver_good:
            yield ProgressUpdate(
                1, "Could not load video webpage after 10 tries." " Check internet connection and video URL."
            )
            return list(dataset)

        time.sleep(5)

        # Mutes the video
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys("m")

        for _ in range(10):
            # Scrolls down to load initial comments
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)

        while True:
            # Pushing the end key once everything is loaded to move to the bottom
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
            # "a.style-scope.ytd-comment-renderer"
            # Collecting the comment and the username
            comments = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#comment #content-text")))
            usernames = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#comment #author-text")))

            # Slice out the comments and usernames that we already have
            comments = comments[len(dataset):]
            usernames = usernames[len(dataset):]

            for comment, username in zip(comments, usernames):
                dataset.add((username.text, comment.text))

                # If we have enough comments, stop scrolling
                if len(dataset) >= count:
                    yield ProgressUpdate(1, "Formatting comments...")
                    formatted = format_comments(list(dataset))
                    yield formatted
                    return
            yield ProgressUpdate(len(dataset) / count, f"Loading comments: {len(dataset)} / {count}")


if __name__ == "__main__":
    # For testing purposes
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    start_time = time.time()
    # Each progressupdate.progress is a num from 0 to 1
    last_progress = 0
    for comments in extract_comments(video_url, 500):
        if isinstance(comments, ProgressUpdate):
            if comments.progress > last_progress + .1:
                print(comments.progress, comments.text)
                last_progress = comments.progress
        else:
            print(comments)
            print("number of comments:", len(comments))
            break

    print("Time taken: " + str(time.time() - start_time))


class Collector(BaseDataCollector):
    def __init__(self):
        pass

    def query_options(self):
        return ["video_url", "count"]

    def collect_generator(self, video_url, count):
        yield from extract_comments(video_url, count)
