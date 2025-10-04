import sys
from pathlib import Path

# Add <repo_root>/src to sys.path so "import fmri2image" works in tests
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))