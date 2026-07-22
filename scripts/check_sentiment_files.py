"""
Diagnostic script to check and fix sentiment module files.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent

required_files = [
    ROOT_DIR / "sentiment" / "__init__.py",
    ROOT_DIR / "sentiment" / "analyzers" / "__init__.py",
    ROOT_DIR / "sentiment" / "analyzers" / "preprocessor.py",
    ROOT_DIR / "sentiment" / "analyzers" / "models.py",
    ROOT_DIR / "sentiment" / "fetchers" / "__init__.py",
    ROOT_DIR / "sentiment" / "fetchers" / "news.py",
]

print("=== Checking Sentiment Module Files ===")
all_present = True

for path in required_files:
    if not path.exists():
        if path.name == "__init__.py":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            print(f"[FIXED] Created missing: {path.relative_to(ROOT_DIR)}")
        else:
            print(f"[MISSING] File not found: {path.relative_to(ROOT_DIR)}")
            all_present = False
    else:
        print(f"[OK] Found: {path.relative_to(ROOT_DIR)}")

if not all_present:
    print("\nPlease create the missing files listed above and try again.")
else:
    print("\nAll required files are present and properly structured!")