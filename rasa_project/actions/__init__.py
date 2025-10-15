import subprocess, sys
from pathlib import Path

def ensure_requirements_installed():
    req = Path(__file__).with_name("requirements.txt")
    if req.exists():
        try:
            # pip will skip already-satisfied packages; this is fast after first run.
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
        except Exception as e:
            # Don't crash the server if pip fails; HF-only actions can still run.
            print(f"[actions bootstrap] pip install -r requirements.txt failed: {e}")

ensure_requirements_installed()


