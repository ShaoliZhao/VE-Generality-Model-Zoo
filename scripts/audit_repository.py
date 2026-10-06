from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["README.md", "LICENSE", "pyproject.toml", "SUBMISSION_TEXT.md", "src/ve/core.py", "src/ve/tasks.py", "tests/test_core.py"]
missing = [x for x in REQUIRED if not (ROOT / x).is_file()]
if missing: raise SystemExit(f"Missing files: {missing}")
text = "\n".join(p.read_text(errors="ignore") for p in ROOT.rglob("*") if p.is_file() and p.suffix in {".md", ".py", ".toml", ".tsv", ".json"})
if re.search(r"[\u4e00-\u9fff]", text): raise SystemExit("Repository must remain English-only")
if list(ROOT.rglob("*.h5ad")): raise SystemExit("Challenge data detected")
print("Repository audit passed.")
