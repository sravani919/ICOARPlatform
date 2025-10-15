import os
os.environ.setdefault("ICOAR_USERNAME", "smoke")

from icoar_agent import run_collect_tool

print(run_collect_tool({
    "platform": "reddit",
    "method": "Scraper",
    "query": {
        "count": 5,
        "keywords": "cyberbullying",
        "get_comments": False,
        "comment_limit": 0,
        "images": False
    }
}))

