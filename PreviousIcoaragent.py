# icoar_agent.py  stable patch
import os, json, time, re, traceback
from datetime import datetime
from pathlib import Path
import difflib
import unicodedata


import streamlit as st
from openai import OpenAI

from ICOAR_core.data_collection.runner import collect_data

# ---------- Natural-language & typo-safe parsing ----------
SAFE_DEFAULT_COUNT = 20
SAFE_MAX_COUNT = 200
SAFE_DATE_FALLBACK_DAYS = 30  # used when user says "recent"/"last few days"
SAFE_NSFW = False

# Platform aliases (typo-friendly)
_PLATFORM_ALIASES = {
    "reddit": ["reddit", "redit", "rddit", "raddit"],
    "twitter": ["twitter", "twiter", "x", "twt", "twiiter"],
    "youtube": ["youtube", "youtub", "yt", "you tube"],
    "kaggle": ["kaggle", "kagle", "kagel"],
    "facebook": ["facebook", "fb", "facebok"],
    "tiktok": ["tiktok", "tik tok", "tictoc", "titkok"],
    "huggingface": ["huggingface", "hugging face", "hf", "hugginface"],
}

# Abuse-topic synonyms & common misspellings (safe categories; no slurs)
_ABUSE_SYNONYMS = {
    "harassment": ["harassment", "harrasment", "harssment", "harresment", "cyberharassment"],
    "hate speech": ["hate speech", "online hate", "hateful content", "hate"],
    "cyberbullying": ["cyberbullying", "cyber bullying", "bullying", "cyber-abuse"],
    "toxic content": ["toxic", "toxicity", "toxic content", "abusive language"],
    "misogyny": ["misogyny", "misoginy", "misoginist", "anti-women"],
    "xenophobia": ["xenophobia", "xenophbia", "anti-immigrant", "anti immigrant"],
    "religious hate": ["religious hate", "religion-based hate", "anti-muslim", "anti-christian", "anti-hindu"],
}

_SUBREDDIT_HINTS = {
    "india": ["india", "IndiaSpeaks"],
    "usa": ["AskAnAmerican", "politics"],
    "health": ["medicine", "AskDocs", "science"],
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

def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return " ".join(s.strip().split())

def _fuzzy_pick(word: str, choices: list[str], cutoff: float = 0.75) -> str | None:
    m = difflib.get_close_matches(word.lower(), [c.lower() for c in choices], n=1, cutoff=cutoff)
    return m[0] if m else None

def _clip_count(n: int | None) -> int:
    try:
        n = int(n or SAFE_DEFAULT_COUNT)
    except Exception:
        n = SAFE_DEFAULT_COUNT
    return max(1, min(SAFE_MAX_COUNT, n))

def _detect_platform_natural(user_text: str) -> str | None:
    t = _normalize_text(user_text).lower()
    # contains any alias
    for plat, aliases in _PLATFORM_ALIASES.items():
        if any(a in t for a in aliases):
            return plat
    # token fuzz
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", t)
    all_aliases = []
    for plat, aliases in _PLATFORM_ALIASES.items():
        for a in aliases:
            all_aliases.append((plat, a))
    names = [a for _, a in all_aliases]
    for tok in tokens:
        m = _fuzzy_pick(tok, names, cutoff=0.8)
        if m:
            for plat, a in all_aliases:
                if a.lower() == m.lower():
                    return plat
    return None

def _extract_count(user_text: str) -> int | None:
    m = re.search(r"\b(\d{1,3})\s*(posts|results|items|entries)?\b", user_text, re.I)
    if not m:
        return None
    return _clip_count(m.group(1))

def _extract_time_hint(user_text: str) -> dict:
    t = _normalize_text(user_text).lower()
    for phrase, (unit, value) in _TIME_HINTS.items():
        if phrase in t:
            return {"time_hint": {"unit": unit, "value": value}}
    return {}

def _extract_keywords(user_text: str) -> list[str]:
    text = _normalize_text(user_text)
    # quoted phrases first
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    quoted = [q[0] or q[1] for q in quoted if (q[0] or q[1])]
    if quoted:
        return [q.strip() for q in quoted if q.strip()][:5]

    # after of/about/on/for/in
    m = re.search(r"\b(?:of|about|on|for|around|regarding|in)\s+([A-Za-z0-9\-_, ]+)", text, re.I)
    if m:
        raw = m.group(1)
        parts = [p.strip() for p in re.split(r"[ ,]+", raw) if p.strip()]
        stop = {"the", "a", "an", "and", "posts", "data", "results"}
        return [p for p in parts if p.lower() not in stop][:5]

    # tokens, minus stop
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)
    stop = {
        "collect","get","download","show","give","reddit","youtube","kaggle","twitter","x","facebook","tiktok",
        "data","posts","results","items","entries","recent","today","yesterday","last","week","month","year",
        "please","me","of","about","on","for","around","regarding","in","to","the","a","an","from","days"
    }
    kws = [w for w in tokens if w.lower() not in stop and not w.isdigit()]
    return kws[:5]



def _extract_abuse_topic_keywords(user_text: str) -> list[str]:
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

def _canonize_keywords(kw) -> list[str]:
    """Turn raw 'keywords' (string or list) into a de-duplicated, corrected list."""
    # Accept string or list
    if isinstance(kw, str):
        parts = [p.strip() for p in kw.split(",") if p.strip()]
    elif isinstance(kw, list):
        parts = [str(p).strip() for p in kw if str(p).strip()]
    else:
        return []

    out = []
    for w in parts:
        wl = w.lower()
        matched = None
        # try mapping to canonical abuse topics via synonyms + fuzzy
        for canonical, variants in _ABUSE_SYNONYMS.items():
            if wl == canonical:
                matched = canonical
                break
            if wl in [v.lower() for v in variants]:
                matched = canonical
                break
            if _fuzzy_pick(wl, [canonical] + variants, cutoff=0.83):
                matched = canonical
                break
        out.append(matched or w)

    # de-dup (preserve order)
    seen = set()
    result = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            result.append(x)
    return result


def _geo_to_subreddits(keywords: list[str]) -> list[str]:
    subs = set()
    for k in keywords:
        k_low = k.lower()
        if k_low in _SUBREDDIT_HINTS:
            subs.update(_SUBREDDIT_HINTS[k_low])
    return list(subs)

def _build_query_from_text(user_text: str, platform: str | None) -> dict:
    count = _extract_count(user_text) or SAFE_DEFAULT_COUNT
    time_hint = _extract_time_hint(user_text)
    keywords = _extract_keywords(user_text)

    base = {
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

    if platform == "reddit":
        subs = _geo_to_subreddits(keywords)
        if subs:
            base["subreddits"] = subs
    return base

def _safe_args_from_user_text(user_text: str) -> dict:
    """Build safe arguments for run_collect_tool from free text."""
    plat = _detect_platform_natural(user_text) or "reddit"
    qry = _build_query_from_text(user_text, plat)

    # If nothing usable, pull neutral abuse categories
    if not qry.get("keywords"):
        abuse_keys = _extract_abuse_topic_keywords(user_text)
        if abuse_keys:
            qry["keywords"] = ", ".join(abuse_keys)

    # keep safe toggles
    qry.setdefault("allow_nsfw", SAFE_NSFW)
    if plat == "reddit":
        qry.setdefault("get_comments", False)
        qry.setdefault("comment_limit", 0)
        qry.setdefault("images", False)
        qry.setdefault("sort", "relevance")

    return {
        "platform": plat,
        "method": "Scraper" if plat == "reddit" else "API",
        "query": qry,
    }


# ---------- Safe username helpers ----------

def _safe_get_username() -> str:
    """Safely get username from Streamlit or environment, else fallback to 'anonymous'."""
    try:
        u = st.session_state.get("username", None)
        if u:  # Only return if truthy
            return str(u)
    except Exception:
        pass

    # Fallback to env var or default
    env_user = os.environ.get("ICOAR_USERNAME")
    return str(env_user) if env_user else "anonymous"


def _safe_set_username(username: str) -> None:
    """Safely set username in Streamlit session_state (no-op if not running in Streamlit)."""
    try:
        if username:
            st.session_state["username"] = str(username)
    except Exception:
        pass



# ---------- OpenAI client ----------
# Do NOT hardcode a fallback key. Force env or Streamlit secrets.
def _get_openai_key() -> str | None:
    # Guard st.secrets for non-Streamlit usage
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None

OPENAI_API_KEY = _get_openai_key()
if not OPENAI_API_KEY:
    def _err(msg): return {"text": f"L OPENAI_API_KEY missing: {msg}", "file": None}
    def run_agent_response(_): return _err("Set it in environment or .streamlit/secrets.toml")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)



def run_collect_tool(inputs: dict) -> dict:
    """
    inputs: { platform: 'reddit'|'huggingface'|'kaggle'|'facebook'|'twitter'|'youtube'|'tiktok',
              method: 'Scraper'|'API',
              query:  { ... } }
    returns: {status: 'ok'|'no_results'|'error', file?: str, message?: str, traceback?: str}
    """
    try:
        platform = (inputs.get("platform") or "").lower().strip()
        method = (inputs.get("method") or "").lower().strip()
        query = dict(inputs.get("query") or {})

        # --- DEBUG: Print inputs coming from GPT ---
        print("DEBUG - run_collect_tool called with:")
        print("  platform:", platform)
        print("  method:", method)
        print("  query:", json.dumps(query, indent=2))


             # Neutral defaults only (do not inject toxic keywords)
        query.setdefault("count", 10)

        # --- Normalize & auto-correct keywords (handle typos like "cyberbullyinng") ---
        if "keywords" in query and query["keywords"]:
            canon = _canonize_keywords(query["keywords"])
            if canon:
                # keep the comma-separated string your collectors expect
                query["keywords"] = ", ".join(canon)


        # Map method aliases
        if method in ("scraper", "scrape"):
            method = "Scraper"
        elif method in ("api", "rest"):
            method = "API"

        else:
            # Default for safety
            method = "Scraper" if platform == "reddit" else "API"

        # Normalize platform tokens
        if platform in ("kaggle_", "kaggle-api"):
            platform = "kaggle"

        if platform not in ("reddit", "huggingface", "kaggle", "facebook", "twitter", "youtube", "tiktok"):
            return {"status": "error", "message": f"Unknown platform: {platform}"}

        if platform == "reddit":
            query.setdefault("get_comments", False)
            query.setdefault("comment_limit", 0)  # 0 = don't fetch comments
            query.setdefault("images", False)
            query.setdefault("allow_nsfw", SAFE_NSFW)
            query.setdefault("sort", "relevance")

        # Username + ensure user directory
        username = _safe_get_username() or "anonymous"
        _safe_set_username(username)
        user_dir = Path("data") / username
        user_dir.mkdir(parents=True, exist_ok=True)

        # Kaggle creds (safe outside Streamlit)
        ksec = {}
        try:
            ksec = dict(st.secrets.get("kaggle", {}))
        except Exception:
            ksec = {}
        if platform == "kaggle":
            query.setdefault("kaggle.username", ksec.get("username"))
            query.setdefault("kaggle.key", ksec.get("key"))
            query.setdefault("delete_temp_data", True)

                # Run collector with traceback
        base = f"{platform}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # --- Last-mile normalization & pruning ---

        # Guard Kaggle creds (avoid calling collector with empty creds)
        if platform == "kaggle":
            if not query.get("kaggle.username") or not query.get("kaggle.key"):
                return {
                    "status": "error",
                    "message": "Kaggle credentials are missing. Set st.secrets['kaggle'] or env vars.",
                }

        # Drop meta keys the collectors dont accept (prevents **kwargs errors)
        for k in ("allow_nsfw", "sort", "time_hint"):
            query.pop(k, None)

        # Optionally whitelist by platform to be extra safe
        if platform == "reddit":
            allowed = {"keywords", "count", "get_comments", "comment_limit", "images", "subreddits"}
            query = {k: v for k, v in query.items() if k in allowed and v is not None}

        print("Final collector call:")
        print("  platform:", platform)
        print("  method:", method)
        print("  query:", json.dumps(query, indent=2))

        try:
            out_file, rows = collect_data(platform, method, query, save_name=base)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during collection: {e.__class__.__name__}: {e}",
                "traceback": traceback.format_exc(),
            }

        if not out_file:
            return {"status": "no_results", "message": "Collector returned no file."}

       out_file, rows = collect_data(platform, method, query, save_name=base)

# Compute a safe row count for the message (rows might be a list or an int)
       try:
           row_count = len(rows)
       except TypeError:
           try:
               row_count = int(rows)
           except Exception:
        row_count = None

       if not out_file:
           return {"status": "no_results", "message": "Collector returned no file."}

# Normalize final path under data/<user>/
       out_file_str = str(out_file)
       if not out_file_str.startswith(f"data/{username}/"):
           final_path = user_dir / Path(out_file_str).name
       else:
           final_path = Path(out_file_str)

       msg = f"Saved {row_count} rows to {final_path}" if row_count is not None else f"Saved to {final_path}"

       return {
           "status": "ok",
           "file": str(final_path),
           "message": msg,
       }


    except Exception as e:
        return {
            "status": "error",
            "message": f"Collector error: {e}",
            "traceback": traceback.format_exc(),
        }


# ---------- Register tool (function calling) ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_collect_tool",
            "description": "Collect data via ICOAR runner and return a saved CSV path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "description": "reddit | huggingface | kaggle | facebook | twitter | youtube | tiktok"
                    },
                    "method": {
                        "type": "string",
                        "description": "Scraper for reddit; API for others"
                    },
                    "query": {
                        "type": "object",
                        "description": "Collector-specific query options dict"
                    }
                },
                "required": ["platform", "method", "query"]
            }
        }
    }
]



# ---------- Assistant lifecycle ----------
ASSISTANT_ID = os.getenv("ICOAR_ASSISTANT_ID", "").strip()

def _ensure_assistant() -> str:
    global ASSISTANT_ID
    if ASSISTANT_ID:
        return ASSISTANT_ID
    a = client.beta.assistants.create(
        name="ICOAR Assistant",
        model="gpt-4o-mini",
        
        instructions=(
	"You are the official assistant for the ICOAR platform (Integrative Cyberinfrastructure for Online Abuse Research).\n"
	"Your role:\n"
	"  (1) Answer questions about ICOAR clearly and briefly.\n"
	"  (2) When asked to collect data, call run_collect_tool with structured arguments.\n"
	"Hard rules:\n"
	"   Do NOT inject harassment, bullying, or toxic keywords unless the user explicitly requests them.\n"
	"   Prefer neutral, user-provided topics/phrases. If the user is vague, infer neutral keywords from their text.\n"
	"  For Reddit, if the user mentions a geography (e.g., India), you may add relevant subreddits like r/india to improve locality.\n"
	"   Default: allow_nsfw=false, get_comments=false, images=false.\n"
	"   If the user provides a timeframe (e.g., 'last 30 days'), include a time_hint.\n"
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
            outputs.append({"tool_call_id": call.id, "output": json.dumps({
                "status": "error", "message": f"Unknown tool {name}"
            })})
    client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id, run_id=run_id, tool_outputs=outputs
    )


def _collect_assistant_text(messages) -> str:
    chunks = []
    for msg in messages.data:
        if getattr(msg, "role", None) != "assistant":
            continue
        for c in getattr(msg, "content", []) or []:
            t = getattr(c, "type", None)
            if t == "text" and getattr(c, "text", None):
                val = getattr(c.text, "value", None) or getattr(c, "text", None)
                if val:
                    chunks.append(val)
    return "\n".join(chunks).strip()


# ---------- Public entry used by Home.py ----------

# ---------- Public entry used by Home.py ----------
def run_agent_response(user_input: str) -> dict:
    """
    Returns: {"text": <assistant_text or error>, "file": <csv_path or None>}
    Always returns something visible to help debugging.
    """
    if not OPENAI_API_KEY:
        return {"text": "L OPENAI_API_KEY not configured.", "file": None}

    try:
        asst_id = _ensure_assistant()

        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=user_input
        )
# Build safe fallback args from the user's plain text (spaces only, no tabs)
        fallback_args = _safe_args_from_user_text(user_input)
        print(
            "INTERPRETED REQUEST:",
            json.dumps(
                {
                     "platform": fallback_args["platform"],
                     "method": fallback_args["method"],
                     "query_preview": {
                         k: fallback_args["query"].get(k)
                         for k in ["keywords", "count", "sort", "time_hint", "subreddits", "allow_nsfw"]
                      },
                  },
                  indent=2,
             ),
        )

        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=asst_id)



        # Poll with backoff + hard timeout so UI never hangs silently
        started = time.time()
        backoff = 0.5
        MAX_WAIT_SECS = 180  # Increased timeout from 45s to 180s

        while True:
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            status = run.status

            if status == "requires_action":
                # Assistant is calling our tool (e.g., run_collect_tool)
                tc = run.required_action.submit_tool_outputs.tool_calls
                _handle_tool_calls(thread.id, run.id, tc)

                # Reset timer after tool call (prevents early timeout)
                started = time.time()
                backoff = 0.5

                # Give it a moment to start processing
                time.sleep(0.5)

            elif status in ("completed", "failed", "cancelled", "expired"):
                break

            else:
                # Still queued or in progress  keep polling
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 2.5)

            # Hard timeout safety net
            if time.time() - started > MAX_WAIT_SECS:
                return {
                    "text": f"  Timed out waiting for assistant (status: {status}). "
                            "Try again with fewer posts or check your collector speed.",
                    "file": None,
                }

        # Pull messages and stitch text
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        final_text = _collect_assistant_text(messages)

        # Try to extract a CSV path from text
        file_path = None
        m = re.search(r"(data\/[^\s]+\.csv)", final_text or "")
        if m:
            file_path = m.group(1)

        # Also scan tool outputs from steps (defensive)
        try:
            steps = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id)
            for step in steps.data:
                details = getattr(step, "step_details", None)
                if not details or getattr(details, "type", "") != "tool_calls":
                    continue
                for tc in getattr(details, "tool_calls", []) or []:
                    fn = getattr(tc, "function", None)
                    if not fn:
                        continue
                    out = getattr(fn, "output", None)
                    if isinstance(out, str) and out.strip().startswith("{"):
                        j = json.loads(out)
                        p = (j or {}).get("file", "")
                        if isinstance(p, str) and p.endswith(".csv"):
                            file_path = p
        except Exception:
            pass

        if not final_text:
            final_text = "  No assistant text returned. (Check logs / tool outputs.)"



        # Last resort: run one safe collection using parsed args
        if not file_path:
            try:
                tool_result = run_collect_tool(fallback_args)
                if (tool_result or {}).get("status") == "ok":
                    file_path = tool_result.get("file")
                    # Keep the success message short and clean (e.g., "Saved 15 rows to data/...csv")
                    final_text = tool_result.get("message", "") or "Collection complete."
                else:
                    final_text = f"(Fallback failed) {json.dumps(tool_result)}"
            except Exception as e:
                final_text = f"(Fallback exception: {e})"

        return {"text": final_text, "file": file_path}


       
