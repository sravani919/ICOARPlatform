# icoar_agent.py  stable patch
import os, json, time, re, traceback
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI

from ICOAR_core.data_collection.runner import collect_data


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

        # --- Fallback: If GPT forgot to pass keywords, force defaults ---
        if platform == "reddit" and not query.get("keywords"):
            query["keywords"] = "bullying"
            print("Fallback: keywords were empty, set to 'bullying'")
        query.setdefault("count", 10)

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

        # Reddit safe defaults
        if platform == "reddit":
            query.setdefault("get_comments", False)
            query.setdefault("comment_limit", 0)  # 0 = don't fetch comments
            query.setdefault("images", False)

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

        # Normalize final path under data/<user>/
        out_file_str = str(out_file)
        if not out_file_str.startswith(f"data/{username}/"):
            final_path = user_dir / Path(out_file_str).name
        else:
            final_path = Path(out_file_str)

        return {
            "status": "ok",
            "file": str(final_path),
            "message": f"Saved {rows} rows to {final_path}",
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
                    "platform": {"type": "string", "description": "reddit | huggingface | kaggle | facebook | twitter | youtube | tiktok"},
                    "method":   {"type": "string", "description": "Scraper for reddit; API for others"},
                    "query":    {"type": "object", "description": "Collector-specific query options dict"},
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
            "You are the official assistant for the ICOAR platform (Integrative Cyberinfrastructure for Online Abuse Research). "
            "Your role: (1) answer questions about ICOAR; (2) when asked to collect data, call run_collect_tool with structured arguments."
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

        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=asst_id)

        # Poll with backoff + hard timeout so UI never hangs silently
  
        started = time.time()
        backoff = 0.5
        MAX_WAIT_SECS = 180  # Increased timeout from 45s  180s

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
                # Still queued or in progress  keep polling
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 2.5)

            # Hard timeout safety net
            if time.time() - started > MAX_WAIT_SECS:
                return {
                    "text": f"  Timed out waiting for assistant (status: {status}). "
                            "Try again with fewer posts or check your collector speed.",
                    "file": None
                }


        # Pull messages and stitch text
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        final_text = _collect_assistant_text(messages)

        # Try to extract a CSV path from text
        file_path = None
        m = re.search(r"(data\/[^\s]+\.csv)", final_text)
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
            final_text = "  No assistant text returned. (Check logs / tool outputs.)"

        return {"text": final_text, "file": file_path}

    except Exception as e:
        return {"text": f"L Exception: {e}", "file": None}



