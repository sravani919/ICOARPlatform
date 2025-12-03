# icoar_agent.py  robust visualize intent + remembers last raw/clean files

import os
import json
import time
import re
import traceback
import difflib
import unicodedata
from datetime import datetime
from pathlib import Path
from collections import Counter
import io
from typing import Optional, Dict, Any, List

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

from ICOAR_core.data_collection.runner import collect_data


# =========================================================
# 1) CONSTANT PATHS
# =========================================================
APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = APP_ROOT / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2) CONSTANTS & HELPERS
# =========================================================
SAFE_DEFAULT_COUNT = 20
SAFE_MAX_COUNT = 200
SAFE_DATE_FALLBACK_DAYS = 30
SAFE_NSFW = False

_PLATFORM_ALIASES = {
    "reddit": ["reddit", "redit", "rddit", "raddit"],
    "twitter": ["twitter", "twiter", "x", "twt", "twiiter"],
    "youtube": ["youtube", "youtub", "yt", "you", "you tube"],
    "kaggle": ["kaggle", "kagle", "kagel"],
    "facebook": ["facebook", "fb", "facebok"],
    "tiktok": ["tiktok", "tik", "tik tok", "tictoc", "titkok"],
    "huggingface": ["huggingface", "hugging", "hugging face", "hf", "hugginface"],
}

_ABUSE_SYNONYMS = {
    "harassment": ["harassment", "harrasment", "harssment", "harresment", "cyberharassment"],
    "hate speech": ["hate speech", "online hate", "hateful content", "hate"],
    "cyberbullying": ["cyberbullying", "cyber bullying", "bullying", "cyber-abuse"],
    "toxic content": ["toxic", "toxicity", "toxic content", "abusive language"],
    "misogyny": ["misogyny", "misoginy", "misoginist", "anti-women"],
    "xenophobia": ["xenophobia", "xenophbia", "anti-immigrant", "anti immigrant"],
    "religious hate": ["religious hate", "religion-based hate", "anti-muslim", "anti-christian", "anti-hindu"],
}

_TIME_HINTS = {
    "today": ("day", 1),
    "yesterday": ("day", 1),
    "last week": ("week", 1),
    "past week": ("week", 1),
    "last 30 days": ("day", 30),
    "last month": ("month", 1),
    "last year": ("year", 1),
    "recent": ("day", SAFE_DATE_FALLBACK_DAYS),
}

# filler words to ignore for keywords
_EXTRA_STOP = {
    "can","you","help","please","pls","plz","me","we","they","he","she","it","yes","no","ok","okay",
    "do","did","does","will","would","could","should","shall","kindly","also","just","now","then",
    "the","a","an","and","of","to","in","on","for","about","around","regarding","from","with","by",
    "posts","data","results","items","entries","recent","today","yesterday","last","week","month","year","days"
}


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return " ".join(s.strip().split())


def _fuzzy_pick(word: str, choices: List[str], cutoff: float = 0.75) -> Optional[str]:
    m = difflib.get_close_matches(word.lower(), [c.lower() for c in choices], n=1, cutoff=cutoff)
    return m[0] if m else None


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _looks_like_any(token: str, targets: List[str], threshold: float = 0.60) -> bool:
    token = token.lower()
    for t in targets:
        if _similar(token, t) >= threshold:
            return True
    return False


def _clip_count(n: Optional[int]) -> int:
    try:
        n = int(n or SAFE_DEFAULT_COUNT)
    except Exception:
        n = SAFE_DEFAULT_COUNT
    return max(1, min(SAFE_MAX_COUNT, n))


def _detect_platform_natural(user_text: str) -> Optional[str]:
    t = _normalize_text(user_text).lower()
    for plat, aliases in _PLATFORM_ALIASES.items():
        if any(a in t for a in aliases):
            return plat

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", t)
    all_aliases = [(plat, a) for plat, aliases in _PLATFORM_ALIASES.items() for a in aliases]
    names = [a for _, a in all_aliases]
    for tok in tokens:
        m = _fuzzy_pick(tok, names, cutoff=0.8)
        if m:
            for plat, a in all_aliases:
                if a.lower() == m.lower():
                    return plat
    return None


def _extract_count(user_text: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,3})\s*(posts|results|items|entries)?\b", user_text, re.I)
    if not m:
        return None
    return _clip_count(m.group(1))


def _extract_time_hint(user_text: str) -> Dict[str, Any]:
    t = _normalize_text(user_text).lower()
    for phrase, (unit, value) in _TIME_HINTS.items():
        if phrase in t:
            return {"time_hint": {"unit": unit, "value": value}}
    return {}


def _extract_keywords(user_text: str) -> List[str]:
    text = _normalize_text(user_text)

    # quoted phrases
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    quoted = [q[0] or q[1] for q in quoted if (q[0] or q[1])]
    if quoted:
        return [q.strip() for q in quoted if q.strip()][:5]

    # "on cyberbullying in schools"
    m = re.search(r"\b(?:of|about|on|for|around|regarding|in)\s+([A-Za-z0-9\-_, ]+)", text, re.I)
    if m:
        raw = m.group(1)
        parts = [p.strip() for p in re.split(r"[ ,]+", raw) if p.strip()]
        return [p for p in parts if p.lower() not in _EXTRA_STOP][:5]

    # fallback tokens
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)
    kws = [w for w in tokens if w.lower() not in _EXTRA_STOP and not w.isdigit()]
    return kws[:5]


def _extract_abuse_topic_keywords(user_text: str) -> List[str]:
    t = _normalize_text(user_text).lower()
    hits = []
    for canonical, variants in _ABUSE_SYNONYMS.items():
        if any(v in t for v in variants):
            hits.append(canonical)
            continue
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", t)
        for tok in tokens:
            m = _fuzzy_pick(tok, variants, cutoff=0.83)
            if m:
                hits.append(canonical)
                break
    return list(dict.fromkeys(hits))[:3]


def _canonize_keywords(kw) -> List[str]:
    if isinstance(kw, str):
        parts = [p.strip() for p in kw.split(",") if p.strip()]
    elif isinstance(kw, list):
        parts = [str(p).strip() for p in kw if str(p).strip()]
    else:
        return []
    result = []
    seen = set()
    for w in parts:
        wl = w.lower()
        matched = None
        for canonical, variants in _ABUSE_SYNONYMS.items():
            allv = [canonical] + variants
            m = _fuzzy_pick(wl, allv, cutoff=0.88)
            if m and m != wl:
                matched = canonical
                break
        final = matched or w
        if final and final not in seen:
            seen.add(final)
            result.append(final)
    return result


def _build_query_from_text(user_text: str, platform: Optional[str]) -> Dict[str, Any]:
    count = _extract_count(user_text) or SAFE_DEFAULT_COUNT
    time_hint = _extract_time_hint(user_text)
    keywords = _extract_keywords(user_text)

    base: Dict[str, Any] = {
        "count": count,
        "allow_nsfw": SAFE_NSFW,
        "get_comments": False,
        "comment_limit": 0,
        "images": False,
        "sort": "relevance",
    }
    if keywords:
        base["keywords"] = ", ".join(keywords)
    base.update(time_hint)
    return base


def _safe_args_from_user_text(user_input: str) -> Dict[str, Any]:
    plat = _detect_platform_natural(user_input) or "reddit"
    qry = _build_query_from_text(user_input, plat)

    if not qry.get("keywords"):
        abuse_keys = _extract_abuse_topic_keywords(user_input)
        if abuse_keys:
            qry["keywords"] = ", ".join(abuse_keys)

    qry.setdefault("allow_nsfw", SAFE_NSFW)
    if plat == "reddit":
        qry.setdefault("get_comments", False)
        qry.setdefault("comment_limit", 0)
        qry.setdefault("images", False)
        qry.setdefault("sort", "relevance")

    return {"platform": plat, "method": "Scraper" if plat == "reddit" else "API", "query": qry}


# =========================================================
# 3) USERNAME & LAST-FILE STATE
# =========================================================
def _safe_get_username() -> str:
    try:
        u = st.session_state.get("username", None)
        if u:
            return str(u)
    except Exception:
        pass
    env_user = os.environ.get("ICOAR_USERNAME")
    if env_user:
        return str(env_user)
    return "anonymous"


def _safe_set_username(username: str) -> None:
    try:
        if username:
            st.session_state["username"] = str(username)
    except Exception:
        pass


def _set_last_file(kind: str, path: Optional[str]):
    try:
        if path:
            st.session_state[f"assistant_last_{kind}_file"] = str(path)
    except Exception:
        pass


def _get_last_file(*kinds: str) -> Optional[str]:
    try:
        for k in kinds:
            p = st.session_state.get(f"assistant_last_{k}_file")
            if p:
                return p
    except Exception:
        pass
    return None


# =========================================================
# 4) OPENAI ASSISTANT
# =========================================================
def _get_openai_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


OPENAI_API_KEY = _get_openai_key()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_collect_tool",
        "description": "Collect data via ICOAR runner and return a saved CSV path.",
        "parameters": {
            "type": "object",
            "properties": {
                "platform": {"type": "string", "description": "reddit | huggingface | kaggle | facebook | twitter | youtube | tiktok"},
                "method": {"type": "string", "description": "Scraper for reddit; API for others"},
                "query": {"type": "object", "description": "Collector-specific query options dict"},
            },
            "required": ["platform", "method", "query"],
        },
    },
}]
ASSISTANT_ID = os.getenv("ICOAR_ASSISTANT_ID", "").strip()


def _ensure_assistant() -> str:
    global ASSISTANT_ID
    if ASSISTANT_ID:
        return ASSISTANT_ID
    if client is None:
        ASSISTANT_ID = "NO_OPENAI"
        return ASSISTANT_ID
    a = client.beta.assistants.create(
        name="ICOAR Assistant",
        model="gpt-4o-mini",
        instructions=(
            "You are the assistant for ICOAR (Integrative Cyberinfrastructure for Online Abuse Research).\n"
            "Be brief. When collecting data, call run_collect_tool with structured arguments.\n"
            "Do NOT inject harmful keywords unless the user provided them. Default: allow_nsfw=false.\n"
        ),
        tools=TOOLS,
    )
    ASSISTANT_ID = a.id
    return ASSISTANT_ID


def _handle_tool_calls(thread_id: str, run_id: str, tool_calls) -> None:
    outputs = []
    for call in tool_calls:
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except Exception:
            args = {}
        if name == "run_collect_tool":
            result = run_collect_tool(args)
            outputs.append({"tool_call_id": call.id, "output": json.dumps(result)})
        else:
            outputs.append({"tool_call_id": call.id, "output": json.dumps({"status": "error", "message": f"Unknown tool {name}"})})
    client.beta.threads.runs.submit_tool_outputs(thread_id=thread_id, run_id=run_id, tool_outputs=outputs)


def _collect_assistant_text(messages) -> str:
    chunks = []
    data = getattr(messages, "data", []) or []
    for msg in data:
        if getattr(msg, "role", None) != "assistant":
            continue
        contents = getattr(msg, "content", []) or []
        for c in contents:
            if getattr(c, "type", None) == "text":
                tv = getattr(getattr(c, "text", None), "value", None)
                if tv:
                    chunks.append(tv)
    return "\n".join(chunks).strip()


# =========================================================
# 5) COLLECTION TOOL
# =========================================================
def run_collect_tool(inputs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        platform = (inputs.get("platform") or "").lower().strip()
        method = (inputs.get("method") or "").lower().strip()
        query = dict(inputs.get("query") or {})

        print("DEBUG - run_collect_tool called with:")
        print("  platform:", platform)
        print("  method:", method)
        print("  query:", json.dumps(query, indent=2))

        query.setdefault("count", 10)

        if "keywords" in query and query["keywords"]:
            canon = _canonize_keywords(query["keywords"])
            if canon:
                query["keywords"] = ", ".join(canon)

        if method in ("scraper", "scrape"):
            method = "Scraper"
        elif method in ("api", "rest"):
            method = "API"
        else:
            method = "Scraper" if platform == "reddit" else "API"

        if platform in ("kaggle_", "kaggle-api"):
            platform = "kaggle"

        allowed_platforms = ("reddit", "huggingface", "kaggle", "facebook", "twitter", "youtube", "tiktok")
        if platform not in allowed_platforms:
            return {"status": "error", "message": f"Unknown platform: {platform}"}

        if platform == "reddit":
            query.setdefault("get_comments", False)
            query.setdefault("comment_limit", 0)
            query.setdefault("images", False)
            query.setdefault("allow_nsfw", SAFE_NSFW)
            query.setdefault("sort", "relevance")

        username = _safe_get_username() or "anonymous"
        _safe_set_username(username)
        user_dir = (DATA_ROOT / username).resolve()
        user_dir.mkdir(parents=True, exist_ok=True)

        ksec = {}
        try:
            ksec = dict(st.secrets.get("kaggle", {}))
        except Exception:
            ksec = {}
        if platform == "kaggle":
            query.setdefault("kaggle.username", ksec.get("username"))
            query.setdefault("kaggle.key", ksec.get("key"))
            query.setdefault("delete_temp_data", True)
            if not query.get("kaggle.username") or not query.get("kaggle.key"):
                return {"status": "error", "message": "Kaggle credentials are missing."}

        for k in ("allow_nsfw", "sort", "time_hint"):
            query.pop(k, None)

        if platform == "reddit":
            allowed = {"keywords", "count", "get_comments", "comment_limit", "images"}
            query = {k: v for k, v in query.items() if k in allowed and v is not None}

        print("Final collector call:")
        print("  platform:", platform)
        print("  method:", method)
        print("  query:", json.dumps(query, indent=2))

        try:
            save_name = f"{platform}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
            out_file, rows = collect_data(platform, method, query, save_name=save_name)
        except Exception as e:
            return {"status": "error", "message": f"Error during collection: {e.__class__.__name__}: {e}",
                    "traceback": traceback.format_exc()}

        if not out_file:
            return {"status": "no_results", "message": "Collector returned no file."}

        try:
            row_count = len(rows)
        except Exception:
            try:
                row_count = int(rows)
            except Exception:
                row_count = None

        raw_out = Path(str(out_file)).resolve()
        stable_path = (DATA_ROOT / username / raw_out.name).resolve()

        if raw_out != stable_path:
            import shutil
            try:
                if raw_out.exists():
                    shutil.copyfile(raw_out, stable_path)
            except Exception as copy_err:
                print("WARN: copy into stable_path failed:", copy_err)

        final_path = str(stable_path)
        msg = f"Saved {row_count} rows to {final_path}" if row_count is not None else f"Saved to {final_path}"

        _set_last_file("raw", final_path)  # remember last raw

        return {"status": "ok", "file": final_path, "rows": int(row_count) if row_count is not None else None, "message": msg}

    except Exception as e:
        return {"status": "error", "message": f"Collector error: {e}", "traceback": traceback.format_exc()}


# =========================================================
# 6) CLEAN / VISUALIZE / SUMMARIZE
# =========================================================
def _fix_path_case(p: str) -> str:
    return os.path.abspath(p).replace("/icoar/", "/ICOAR/")


def run_clean_step(file_path: str) -> Dict[str, Any]:
    abs_input = _fix_path_case(file_path)

    if abs_input.endswith("__clean.csv"):
        return {
            "text": f"### Clean / Preprocess\n\n`{abs_input}` is already a cleaned file. You can visualize or summarize it.",
            "file": abs_input,
            "actions": [
                {"type": "visualize", "label": "Visualize", "file": abs_input},
                {"type": "summarize", "label": "Summarize", "file": abs_input},
                {"type": "export", "label": "Export to Excel", "file": abs_input},
            ],
            "plot_png": None,
        }

    try:
        from tabs.Data_Collection.data_preprocessing_tab import run_preprocess_file
        cleaned_path, cleaned_rows = run_preprocess_file(abs_input)
        cleaned_path = _fix_path_case(cleaned_path)

        _set_last_file("clean", cleaned_path)  # remember last cleaned

        msg = (
            "### Clean / Preprocess\n\n"
            f"I cleaned the dataset from `{abs_input}` and saved a new file at `{cleaned_path}`.\n\n"
            f"The cleaned file has **{cleaned_rows} rows** after removing non-English text, links/mentions, extra punctuation, and empty rows."
        )
        return {
            "text": msg,
            "file": cleaned_path,
            "actions": [
                {"type": "visualize", "label": "Visualize", "file": cleaned_path},
                {"type": "summarize", "label": "Summarize", "file": cleaned_path},
                {"type": "export", "label": "Export to Excel", "file": cleaned_path},
            ],
            "plot_png": None,
        }
    except Exception as e:
        return {"text": f"Preprocessing failed: {e}", "file": abs_input, "actions": [], "plot_png": None}


def run_visualize_step(file_path: str) -> Dict[str, Any]:
    abs_input = _fix_path_case(file_path)

    if not os.path.exists(abs_input):
        return {
            "text": f"### Visualization\n\nI couldn't find `{abs_input}` anymore. Please recollect or clean again so I can visualize it.",
            "file": abs_input,
            "actions": [
                {"type": "summarize", "label": "Summarize", "file": abs_input},
                {"type": "export", "label": "Export to Excel", "file": abs_input},
            ],
            "plot_png": None,
        }

    try:
        df = pd.read_csv(abs_input)
    except Exception as e:
        return {
            "text": f"### Visualization\n\nCouldn't open `{abs_input}` as CSV ({e}).",
            "file": abs_input,
            "actions": [
                {"type": "summarize", "label": "Summarize", "file": abs_input},
                {"type": "export", "label": "Export to Excel", "file": abs_input},
            ],
            "plot_png": None,
        }

    text_col = None
    for cand in ["clean_text", "text", "body", "content", "comment", "message"]:
        if cand in df.columns:
            text_col = cand
            break

    if text_col is None:
        return {
            "text": "### Visualization\n\nI loaded the file but couldn't find a text column (looked for clean_text/text/body/content/comment/message).",
            "file": abs_input,
            "actions": [
                {"type": "summarize", "label": "Summarize", "file": abs_input},
                {"type": "export", "label": "Export to Excel", "file": abs_input},
            ],
            "plot_png": None,
        }

    STOP = {
        "the","a","an","and","to","of","in","on","for","is","are","am",
        "i","you","he","she","it","we","they","this","that","with","as",
        "at","be","was","were","by","or","from","about","just","not",
        "im","it's","its","rt"
    }

    words: List[str] = []
    for text in df[text_col].astype(str).tolist():
        tokens = re.split(r"[^a-zA-Z]+", text.lower())
        for t in tokens:
            if len(t) < 3:
                continue
            if t in STOP:
                continue
            words.append(t)

    common = Counter(words).most_common(10)
    row_count = len(df)

    if not common:
        msg = (
            "### Visualization Preview\n\n"
            f"I loaded `{abs_input}` with **{row_count} rows**, but there weren't enough frequent words to plot.\n\n"
            "For timelines, sentiment charts, and trends over time, open the main **Visualization** tab."
        )
        return {"text": msg, "file": abs_input, "actions": [
            {"type": "summarize", "label": "Summarize", "file": abs_input},
            {"type": "export", "label": "Export to Excel", "file": abs_input},
        ], "plot_png": None}

    labels = [w for (w, c) in common]
    counts = [c for (w, c) in common]

    fig, ax = plt.subplots()
    ax.barh(labels, counts)
    ax.set_xlabel("Count")
    ax.set_ylabel("Word")
    ax.set_title("Top words (cleaned text)")
    ax.invert_yaxis()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plot_png = buf.read()
    plt.close(fig)

    _set_last_file("visualized", abs_input)

    msg = (
        "### Visualization Preview\n\n"
        f"I loaded `{abs_input}` with **{row_count} rows** and generated a quick word-frequency view."
    )
    return {
        "text": msg,
        "file": abs_input,
        "actions": [
            {"type": "summarize", "label": "Summarize", "file": abs_input},
            {"type": "export", "label": "Export to Excel", "file": abs_input},
        ],
        "plot_png": plot_png,
    }


def run_summarize_step(file_path: str) -> Dict[str, Any]:
    abs_input = _fix_path_case(file_path)
    msg = (
        "### Summary Preview\n\n"
        f"I can generate a neutral summary of recurring themes in `{abs_input}` without adding new harmful language."
    )
    return {"text": msg, "file": abs_input, "actions": [
        {"type": "visualize", "label": "Visualize", "file": abs_input},
        {"type": "export", "label": "Export to Excel", "file": abs_input},
    ], "plot_png": None}


# =========================================================
# 7) COMPOSE + MAIN ENTRY
# =========================================================
def compose_after_collection(platform: str, query: Dict[str, Any], file_path: str, rows: Optional[int]) -> str:
    kws = (query or {}).get("keywords") or "your topic"
    cnt = (query or {}).get("count") or rows or "the requested number of"
    parts = []
    parts.append("### Data is ready")
    parts.append(
        f"Your dataset on **{kws}** from **{platform.capitalize()}** has been collected. "
        f"A total of **{cnt} posts** were gathered and saved to `{file_path}`"
        f"{f'. The file contains **{rows} rows**.' if rows is not None else '.'}"
    )
    parts.append("Next steps: clean and preprocess the text, visualize patterns, get an automatic summary, or export the data.")
    parts.append("Your exploration continues below. Choose what you want to do next.")
    return "\n\n".join(parts)


def _has_visualize_intent(text: str) -> bool:
    """
    Super-tolerant visualize intent.
    Catches: visualize/visualise/viz/vizualize/vsualzie/vis + plot/chart/graph + 'show it'/'see it'
    Also triggers if any token contains the substring 'visu', 'vsl', or 'vsz'.
    """
    t = (text or "").lower()
    if re.search(r"\b(show|see)\s+(it|them)\b", t):
        return True

    tokens = re.findall(r"[a-zA-Z]+", t)
    visualize_targets = ["visualize", "visualise", "viz", "vizualize", "vsualzie", "vis", "visual", "vizualise"]
    plot_targets = ["plot", "chart", "graph"]

    # direct token similarity
    if any(_looks_like_any(tok, visualize_targets + plot_targets, threshold=0.60) for tok in tokens):
        return True

    # substring heuristic for typos
    if any(("visu" in tok.lower()) or ("vsl" in tok.lower()) or ("vsz" in tok.lower()) for tok in tokens):
        return True

    return False


def run_agent_response(user_input: str) -> Dict[str, Any]:
    no_key_msg = ("OPENAI_API_KEY not configured. Using local collection/cleaning only.") if client is None else ""
    lowered = (user_input or "").lower().strip()

    # ---------- EARLY: VISUALIZE FUZZY INTENT (uses last cleaned if available, else raw) ----------
    if _has_visualize_intent(lowered):
        target = _get_last_file("clean", "raw")
        if not target:
            try:
                target = (st.session_state.get("assistant_result") or {}).get("file")
            except Exception:
                target = None
        if target:
            return run_visualize_step(target)

    # ---------- EXPLICIT ACTION SHORTCUT ----------
    m_action = re.search(r"action:(clean|visualize|summarize|export)\s+file:(.+)", lowered)
    if m_action:
        act = m_action.group(1)
        fpath = m_action.group(2).strip()
        if act == "clean":
            return run_clean_step(fpath)
        elif act == "visualize":
            return run_visualize_step(fpath)
        elif act == "summarize":
            return run_summarize_step(fpath)
        elif act == "export":
            msg = "### Export\n\nI will generate an Excel-friendly file so you can share or archive it."
            return {"text": msg, "file": _fix_path_case(fpath), "actions": [], "plot_png": None}

    # ---------- HEURISTICS (visualize -> clean -> summarize) ----------
    def _guess_target_file(keyword: str) -> Optional[str]:
        if keyword not in lowered:
            return None
        guessed = re.findall(r"(/?[\w\/\.-]+\.csv)", lowered)
        target = None
        if guessed:
            target = guessed[0]
            if target.startswith("home/"):
                target = "/" + target
        else:
            try:
                target = (st.session_state.get("assistant_result") or {}).get("file")
            except Exception:
                target = None
        return target

    tgt = _guess_target_file("visualize")
    if tgt:
        return run_visualize_step(tgt)

    tgt = _guess_target_file("clean")
    if tgt:
        return run_clean_step(tgt)

    tgt = _guess_target_file("summarize")
    if tgt:
        return run_summarize_step(tgt)

    # ---------- NEW NATURAL REQUEST: COLLECT ----------
    try:
        fallback_args = _safe_args_from_user_text(user_input)
        print("INTERPRETED REQUEST:", json.dumps({
            "platform": fallback_args["platform"],
            "method": fallback_args["method"],
            "query_preview": {k: fallback_args["query"].get(k) for k in ["keywords", "count", "sort", "time_hint", "allow_nsfw"]},
        }, indent=2))

        stable_file_path: Optional[str] = None
        final_text = ""
        actions: List[Dict[str, Any]] = []
        plot_png = None

        if client is not None:
            asst_id = _ensure_assistant()
            thread = client.beta.threads.create()
            client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_input)
            run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=asst_id)

            started = time.time()
            backoff = 0.5
            MAX_WAIT_SECS = 180

            while True:
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                status = run.status
                if status == "requires_action":
                    tc = run.required_action.submit_tool_outputs.tool_calls
                    _handle_tool_calls(thread.id, run.id, tc)
                    started = time.time()
                    backoff = 0.5
                    time.sleep(0.5)
                elif status in ("completed", "failed", "cancelled", "expired"):
                    break
                else:
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 2.5)
                if time.time() - started > MAX_WAIT_SECS:
                    final_text = (f"Timed out waiting for assistant (status: {status}). "
                                  "Try again with fewer posts or check your collector speed.")
                    break

            messages = client.beta.threads.messages.list(thread_id=thread.id)
            assistant_text = _collect_assistant_text(messages)
            if assistant_text:
                final_text = assistant_text

            try:
                steps = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id)
                for step in getattr(steps, "data", []) or []:
                    details = getattr(step, "step_details", None)
                    if getattr(details, "type", "") != "tool_calls":
                        continue
                    for tc in getattr(details, "tool_calls", []) or []:
                        fn = getattr(tc, "function", None)
                        out = getattr(fn, "output", None) if fn else None
                        if isinstance(out, str) and out.strip().startswith("{"):
                            j = json.loads(out)
                            p = (j or {}).get("file")
                            if isinstance(p, str) and p.endswith(".csv"):
                                stable_file_path = p
            except Exception:
                pass

            if not stable_file_path and final_text:
                m = re.search(r"(/home/[^\s]+\.csv)", final_text)
                if m:
                    stable_file_path = m.group(1)

        if not stable_file_path:
            tool_result = run_collect_tool(fallback_args)
            if (tool_result or {}).get("status") == "ok":
                stable_file_path = tool_result.get("file")
                rows = tool_result.get("rows")
                final_text = compose_after_collection(
                    fallback_args["platform"], fallback_args["query"], stable_file_path, rows
                )
            else:
                if not final_text:
                    final_text = f"(Fallback failed) {json.dumps(tool_result)}"
                if client is None:
                    final_text = no_key_msg + "\n\n" + final_text

        if stable_file_path:
            _set_last_file("raw", stable_file_path)
            actions = [
                {"type": "download",  "label": "Download CSV",    "file": stable_file_path},
                {"type": "clean",     "label": "Clean Text",      "file": stable_file_path},
                {"type": "visualize", "label": "Visualize",       "file": stable_file_path},
                {"type": "summarize", "label": "Summarize",       "file": stable_file_path},
                {"type": "export",    "label": "Export to Excel", "file": stable_file_path},
            ]
            if "Data is ready" not in final_text:
                final_text = compose_after_collection(
                    fallback_args["platform"], fallback_args["query"], stable_file_path, None
                )

        if not final_text:
            final_text = "No assistant text returned."

        return {"text": final_text, "file": stable_file_path, "actions": actions, "plot_png": plot_png}

    except Exception as e:
        return {"text": f"Unhandled error: {e}", "file": None, "actions": [], "plot_png": None}

