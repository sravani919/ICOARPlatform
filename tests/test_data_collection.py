import unittest


class Twitter(unittest.TestCase):
    def test_scraping(self):
        from data_collection.twitter.twitter_scraper import grab_tweets

        tweets = grab_tweets("covid", 5, False, None, None)
        # Make sure we got one or more tweets
        self.assertEqual(len(tweets) > 0, True)

    def test_api(self):
        from data_collection.twitter.twitter_api import grab_tweets

        tweets = grab_tweets("covid", 5, False, None, None)
        self.assertEqual(len(tweets) > 0, True)


class Reddit(unittest.TestCase):
    def test_scraping(self):
        from data_collection.reddit import grab_posts

        start, end, post_count, posts = grab_posts("covid", 5, False)
        self.assertEqual(len(posts) > 0, True)

    def test_api(self):
        pass


class TikTok(unittest.TestCase):
    def test_scraping(self):
        from data_collection.tiktok import hashtag_search

        tiktoks = hashtag_search("fun", 5)
        self.assertEqual(len(tiktoks) > 0, True)

    def test_api(self):
        pass


class Facebook(unittest.TestCase):
    def test_scraping(self):
        from data_collection.facebook import scraper

        posts = scraper.grab_posts("movie", 5)
        self.assertEqual(len(posts) > 0, True, "No posts found, scraper may be broken")

    def test_api(self):
        pass


if __name__ == "__main__":
    unittest.main()
