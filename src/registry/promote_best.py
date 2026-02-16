import json
import shutil
from pathlib import Path

REGISTRY_FILE = Path("src/registry/production.json")
ARTIFACT_DIR = Path("src/registry/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def promote_local_model(candidate_model_path: str):
    candidate = Path(candidate_model_path)
    if not candidate.exists():
        raise FileNotFoundError(f"Aday model yok: {candidate}")

    best_path = ARTIFACT_DIR / "best_model.pkl"
    shutil.copy2(candidate, best_path)

    cfg = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    cfg["loader"] = "local"
    cfg["model_path"] = str(best_path)

    # MLflow alanı şimdilik dursun
    cfg.setdefault("mlflow", {})
    cfg["mlflow"]["run_id"] = None
    cfg["mlflow"]["model_uri"] = None

    REGISTRY_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"✅ Promoted -> {best_path}")
    print("✅ Updated -> production.json")


if __name__ == "__main__":
    # Örnek: eğitimden çıkan model dosyanız neredeyse onu yazın
    promote_local_model("outputs/model.pkl")
