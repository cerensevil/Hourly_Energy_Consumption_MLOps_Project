from pathlib import Path
import shutil
import json

ROOT = Path(__file__).resolve().parents[2]

# evaluate flow sonrası oluşan artifacts (kaynak)
SOURCE_MODEL = ROOT / "artifacts" / "best_model.pkl"
SOURCE_META = ROOT / "artifacts" / "best_model.json"

# registry hedef klasörü
TARGET_DIR = ROOT / "src" / "registry" / "artifacts"

def promote():
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"Source model yok: {SOURCE_MODEL}")

    if not SOURCE_META.exists():
        raise FileNotFoundError(f"Source metadata yok: {SOURCE_META}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SOURCE_MODEL, TARGET_DIR / "best_model.pkl")
    shutil.copy2(SOURCE_META, TARGET_DIR / "best_model.json")

    print("Model production registry'ye taşındı.")

if __name__ == "__main__":
    promote()
