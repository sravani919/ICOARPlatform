# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []


# /home/spati/ICOAR/rasa_project/actions/actions.py
from ICOAR_core.data_collection.reddit.scraper import Collector
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import SlotSet, EventType
from rasa_sdk.forms import FormValidationAction

from ICOAR_core.data_collection.runner import collect_data
from ICOAR_core.data_collection.reddit.scraper import Collector

from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
import os
import re
import time
import pandas as pd




def _btn(text: str, action: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Small helper to build a button-like directive your Streamlit app can consume."""
    return {"label": text, "do": action, "args": args or {}}


class ActionIcoarNextSteps(Action):
    def name(self) -> Text:
        return "action_icoar_next_steps"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        text = (
            "Suggested flow:\n"
            "1) Collect data (platforms, keywords, date range)\n"
            "2) Preprocess & validate\n"
            "3) Run text and/or image models\n"
            "4) Explore visualizations (temporal, topics, networks, frequency)\n"
            "5) (Optional) Integrate a custom Hugging Face model"
        )

        buttons = [
            _btn("Go to Data Collection", "nav", {"tab": "data_collection"}),
            _btn("Open Preprocessing", "nav", {"tab": "preprocessing"}),
            _btn("Run Text Models", "nav", {"tab": "text_models"}),
            _btn("Open Visualizations", "nav", {"tab": "visualizations"}),
            _btn("Add HF Model", "nav", {"tab": "custom_models"}),
        ]

        # Send both human text and a machine-readable payload
        dispatcher.utter_message(text=text, json_message={"buttons": buttons})
        return []


class ActionIcoarShortcuts(Action):
    def name(self) -> Text:
        return "action_icoar_shortcuts"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        last_intent = (tracker.latest_message or {}).get("intent", {}).get("name", "")
        buttons: List[Dict[str, Any]] = []

        if last_intent == "ask_data_collection":
            buttons = [
                _btn("Open Data Collection", "nav", {"tab": "data_collection"}),
                _btn(
                    "Prefill: X + 'covid'",
                    "prefill",
                    {"tab": "data_collection", "platform": "X", "keywords": "covid"},
                ),
            ]

        elif last_intent == "ask_text_models":
            buttons = [
                _btn("Run Toxic Detection", "run_text_model", {"model": "toxic"}),
                _btn("Run Hate Speech", "run_text_model", {"model": "hate_speech"}),
                _btn("Run Sentiment", "run_text_model", {"model": "sentiment"}),
            ]

        elif last_intent == "ask_image_models":
            buttons = [
                _btn("Hateful Memes", "run_image_model", {"model": "hateful_memes"}),
                _btn("Deepfake Detection", "run_image_model", {"model": "deepfake"}),
                _btn("Cyberbullying Images", "run_image_model", {"model": "cyberbully_images"}),
            ]

        elif last_intent == "ask_visualizations":
            buttons = [
                _btn("Temporal Trends", "viz", {"type": "temporal"}),
                _btn("Topic Modeling", "viz", {"type": "topics"}),
                _btn("User Network", "viz", {"type": "network"}),
                _btn("Frequency", "viz", {"type": "frequency"}),
            ]

        elif last_intent == "ask_custom_models":
            buttons = [
                _btn("Search HF: hate speech", "hf_search", {"query": "hate speech"}),
                _btn("Search HF: sentiment", "hf_search", {"query": "sentiment"}),
            ]

        if buttons:
            dispatcher.utter_message(text="Here are some shortcuts:", json_message={"buttons": buttons})

        return []


# Optional quick test action you can trigger from a rule/story
class ActionHelloWorld(Action):
    def name(self) -> Text:
        return "action_hello_world"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Hello World from actions server!")
        return []

HF_PLATFORM = ["huggingface", "hf", "hugging face"]
KG_PLATFORM = ["kaggle"]

def _normalize_platform(p: str) -> str:
    if not p:
        return ""
    p = p.lower().strip()
    if p in HF_PLATFORM:
        return "huggingface"
    if p in KG_PLATFORM:
        return "kaggle"
    return p

def _ensure_path(path: str) -> None:
    os.makedirs(path, exist_ok=True)

class ValidateDatasetSearchForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_dataset_search_form"

    def validate_platform(
        self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        val = (slot_value or "").strip()
        val = _normalize_platform(val) if val else ""
        if not val:
            dispatcher.utter_message(response="utter_ask_platform")
            return {"platform": None}
        if val not in {"huggingface", "kaggle"}:
            dispatcher.utter_message(text=f"Unsupported platform: {val}. Use Kaggle or Hugging Face.")
            return {"platform": None}
        return {"platform": val}

    def validate_keywords(
        self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        if slot_value and str(slot_value).strip():
            return {"keywords": str(slot_value).strip()}
        dispatcher.utter_message(response="utter_ask_keywords")
        return {"keywords": None}

    def validate_max_results(
        self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        try:
            n = int(slot_value) if slot_value is not None else 10
            if 1 <= n <= 1000:
                return {"max_results": n}
        except Exception:
            pass
        return {"max_results": 10}

# BEFORE:
# class ValidateDatasetDownloadForm(Action):
# AFTER:
class ValidateDatasetDownloadForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_dataset_download_form"

    def validate_platform(
        self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        val = _normalize_platform(slot_value or "")
        if not val:
            dispatcher.utter_message(response="utter_ask_platform")
            return {"platform": None}
        return {"platform": val}

    def validate_dataset_id(
        self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        if slot_value and str(slot_value).strip():
            return {"dataset_id": str(slot_value).strip()}
        dispatcher.utter_message(response="utter_ask_dataset_id")
        return {"dataset_id": None}

    def validate_file_format(
        self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        fmt = (str(slot_value or "").lower() or None)
        if not fmt:
            return {"file_format": None}
        if fmt in {"csv","json","parquet","tsv","txt"}:
            return {"file_format": fmt}
        return {"file_format": None}

    def validate_download_path(
        self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        path = str(slot_value or "").strip()
        if not path:
            dispatcher.utter_message(response="utter_ask_download_path")
            return {"download_path": None}
        _ensure_path(path)
        return {"download_path": path}




class ActionSearchDatasets(Action):
    def name(self) -> Text:
        return "action_search_datasets"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        platform = _normalize_platform(tracker.get_slot("platform"))
        keywords = tracker.get_slot("keywords")
        max_results = int(tracker.get_slot("max_results") or 10)

        dispatcher.utter_message(response="utter_searching", platform=platform, keywords=keywords)
        results: List[Dict[str, Any]] = []

        try:
            if platform == "huggingface":
                api = HfApi()
                ds = api.list_datasets(search=keywords, limit=max_results)
                for d in ds:
                    results.append({
                        "platform": "huggingface",
                        "dataset_id": d.id,
                        "title": (d.cardData or {}).get("title") or d.id,
                    })

            elif platform == "kaggle":
                # Lazy import so missing creds don't crash the server
                try:
                    from kaggle.api.kaggle_api_extended import KaggleApi
                except Exception as e:
                    dispatcher.utter_message(text=f"Kaggle not available: {e}")
                    return []

                api = KaggleApi()
                try:
                    api.authenticate()
                except Exception as e:
                    dispatcher.utter_message(text=f"Kaggle auth failed: {e}")
                    return []

                ds = api.dataset_list(search=keywords)
                for d in ds[:max_results]:
                    results.append({
                        "platform": "kaggle",
                        "dataset_id": d.ref,
                        "title": d.title,
                        "url": f"https://www.kaggle.com/datasets/{d.ref}",
                    })
            else:
                dispatcher.utter_message(text=f"Unsupported platform: {platform}. Use Kaggle or Hugging Face.")
                return []

        except Exception as e:
            dispatcher.utter_message(text=f"Search failed: {e}")
            return []

        if not results:
            dispatcher.utter_message(text="No datasets found. Try different keywords or platform.")
            return []

        lines = [
            f"- {r['platform']}: {(r.get('title') or r['dataset_id'])}  (`{r['dataset_id']}`)"
            for r in results
        ]
        dispatcher.utter_message(
            text="Here are some matches:\n" + "\n".join(lines) + "\n\nSay download <identifier> to <path> to fetch one.",
            json_message={"results": results}
        )
        return []

class ActionDownloadDataset(Action):
    def name(self) -> Text:
        return "action_download_dataset"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        platform = _normalize_platform(tracker.get_slot("platform"))
        dataset_id = tracker.get_slot("dataset_id")
        download_path = tracker.get_slot("download_path")
        needs_auth = bool(tracker.get_slot("needs_auth"))

        # Respect a user denial after the confirm prompt
        if (tracker.latest_message.get("intent") or {}).get("name") == "deny":
            dispatcher.utter_message(response="utter_cancelled")
            return [SlotSet("needs_auth", False)]

        # Safety checks (in case form validation didn't run)
        if not platform:
            dispatcher.utter_message(response="utter_ask_platform")
            return []
        if not dataset_id:
            dispatcher.utter_message(response="utter_ask_dataset_id")
            return []
        if not download_path:
            dispatcher.utter_message(response="utter_ask_download_path")
            return []

        try:
            if platform == "huggingface":
                _ensure_path(download_path)
                local_path = snapshot_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    local_dir=download_path,
                )
                dispatcher.utter_message(
                    text=f"Downloaded HF dataset `{dataset_id}` to: {local_path}"
                )

            elif platform == "kaggle":
                # If we already detected missing creds earlier, bail out politely.
                if needs_auth:
                    dispatcher.utter_message(response="utter_need_kaggle_creds")
                    return []

                # Lazy import + auth so missing kaggle.json or env vars don't crash start-up
                try:
                    from kaggle.api.kaggle_api_extended import KaggleApi
                except Exception:
                    dispatcher.utter_message(response="utter_need_kaggle_creds")
                    return []

                api = KaggleApi()
                try:
                    api.authenticate()
                except Exception:
                    dispatcher.utter_message(response="utter_need_kaggle_creds")
                    return []

                _ensure_path(download_path)
                api.dataset_download_files(
                    dataset=dataset_id,
                    path=download_path,
                    unzip=True,
                    quiet=True,
                )
                dispatcher.utter_message(
                    text=f"Downloaded Kaggle dataset `{dataset_id}` to: {download_path}"
                )

            else:
                dispatcher.utter_message(text=f"Unsupported platform: {platform}")
                return []

        except Exception as e:
            dispatcher.utter_message(text=f"Download failed: {e}")
            return []

        # Reset needs_auth so the next Kaggle attempt can retry auth
        return [SlotSet("needs_auth", False)]


class ActionClearDownloadSlots(Action):
    def name(self) -> Text:
        return "action_clear_download_slots"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[EventType]:
        return [
            SlotSet("platform", None),
            SlotSet("dataset_id", None),
            SlotSet("keywords", None),
            SlotSet("file_format", None),
            SlotSet("max_results", None),
            SlotSet("download_path", None),
            SlotSet("needs_auth", False),
        ]



class ActionRunDataCollection(Action):
    def name(self) -> Text:
        return "action_run_data_collection"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:

        platform = tracker.get_slot("platform")
        keywords = tracker.get_slot("keywords")
        max_results = tracker.get_slot("max_results") or 100

        # Default to 'standard' collection type for all platforms
        method_name = "standard"

        query_values = {
            "keywords": keywords,
            "limit": max_results
        }

        try:
            file_path, results = collect_data(
                platform_name=platform,
                method_name=method_name,
                query_values=query_values,
            )

            if file_path:
                dispatcher.utter_message(text=f" Collected and saved data to: `{file_path}`")
                return [SlotSet("download_path", file_path)]
            else:
                dispatcher.utter_message(text="  No results found.")

        except Exception as e:
            dispatcher.utter_message(text=f"L Error during collection: {e}")

        return []

# === Reddit collection (append at end of actions.py) ===

# === Reddit collection (replace your existing class with this) ===
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None


class ActionCollectReddit(Action):
    def name(self) -> Text:
        return "action_collect_reddit"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Dict[Text, Any]]:
        def to_bool(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if v is None:
                return False
            return str(v).strip().lower() in {"1", "true", "yes", "y", "include comments", "with comments"}

        # Ensure Streamlit username (some save helpers expect this)
        username = (tracker.get_slot("username") or tracker.sender_id or "rasa").strip()
        if st is not None and "username" not in getattr(st, "session_state", {}):
            st.session_state["username"] = username

        # --- Parse inputs from slots/entities ---
        raw_keywords = tracker.get_slot("keywords") or ""
        if isinstance(raw_keywords, list):
            kw_list = [str(k).strip() for k in raw_keywords if str(k).strip()]
        else:
            # allow comma or whitespace separated keywords
            parts = raw_keywords.split(",") if "," in raw_keywords else raw_keywords.split()
            kw_list = [p.strip() for p in parts if p.strip()]

        if not kw_list:
            dispatcher.utter_message(text="  Please provide keywords for Reddit search.")
            return []

        # count may arrive as `count` or `max_results` (your NLU extracted both previously)
        try:
            count = int(tracker.get_slot("count") or tracker.get_slot("max_results") or 50)
        except Exception:
            count = 50

        images = to_bool(tracker.get_slot("images"))
        get_comments = to_bool(tracker.get_slot("get_comments"))
        cl_raw = tracker.get_slot("comment_limit")
        try:
            comment_limit = int(cl_raw) if cl_raw not in (None, "", "null", "None") else None
        except Exception:
            comment_limit = None

        # MUST match Collector.query_options(): ["count","keywords","images","get_comments","comment_limit"]
        query_values: Dict[str, Any] = {
            "count": count,
            "keywords": " ".join(kw_list),   # praw search expects a string query
            "images": images,
            "get_comments": get_comments,
            "comment_limit": comment_limit,
        }

        # Optional download base (so UI can build a link); if not set, we still return file_path
        download_base = os.getenv("ICOAR_DOWNLOAD_BASE")  # e.g., http://localhost:8501

        # --- Do the work ---
        try:
            csv_path, results = collect_data(
                platform_name="reddit",
                method_name="Scraper",
                query_values=query_values,
                save_name=f"reddit_{'_'.join(kw_list)[:40]}",
            )
        except Exception as e:
            dispatcher.utter_message(text=f"L Reddit collection failed:\n```\n{e}\n```")
            return []

        if not csv_path:
            dispatcher.utter_message(text="No results returned for that query.")
            return []

        n = len(results) if isinstance(results, list) else None
        payload: Dict[str, Any] = {
            "file_path": csv_path,
            "platform": "reddit",
            "keywords": query_values["keywords"],
            "count": count,
            "images": bool(images),
            "comments": bool(get_comments),
            "comment_limit": comment_limit,
            "items": n,
        }
        # If a base is provided, include a simple download URL (your web app should serve the file)
        if download_base:
            payload["download_url"] = f"{download_base}/?file={os.path.basename(csv_path)}"

        # --- Always utter a human-friendly message + machine-friendly payload ---
        dispatcher.utter_message(
            text=(
                " Reddit collection complete.\n"
                f"- keywords: `{query_values['keywords']}`\n"
                f"- count: `{count}` | images: `{images}` | comments: `{get_comments}`"
                + (f" (limit={comment_limit})" if comment_limit is not None else "") + "\n"
                f"- saved: `{csv_path}`\n"
                + (f"- items: `{n}`" if n is not None else "")
            ),
            json_message=payload,
        )

        # Clear slots so subsequent runs don't reuse old values
        return [
            SlotSet("keywords", None),
            SlotSet("count", None),
            SlotSet("max_results", None),
            SlotSet("images", None),
            SlotSet("get_comments", None),
            SlotSet("comment_limit", None),
        ]


