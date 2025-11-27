import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if os.environ.get("GITHUB_ACTIONS") == "true":
    os.environ["SKIP_SPEECHBRAIN"] = "1"

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)