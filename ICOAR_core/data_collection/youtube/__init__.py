from . import comments, youtube_data_api_videos

name = "YouTube"
collection_methods = {
    "Comment Scraping": comments,
    "YouTube Data API (Videos)": youtube_data_api_videos,
}
