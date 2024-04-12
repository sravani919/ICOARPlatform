import json

from ICOAR_core.data_collection import reddit


def main():
    collector = reddit.scraper.Collector()
    for result in collector.collect_generator(1000, ["privacy label"], False, True, comment_limit=None):
        if isinstance(result, reddit.scraper.ProgressUpdate):
            print(round(result.progress, 2), result.text)
        else:
            break

    # Save the result (a list of dictionaries) to a file
    with open("reddit_privacy_label.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    # Load the json and print the size of it
    with open("reddit_privacy_label.json", "r") as f:
        data = json.load(f)
        print("Number of posts: ", len(data))
        number_of_comments = sum([len(post["comments"]) for post in data])
        print("Number of comments: ", number_of_comments)
