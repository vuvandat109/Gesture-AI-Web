"""Verify the *actual* TF-C release artifacts before deploying Gesture AI Web.

Run only against first-party, trusted checkpoint and joblib files. Returns nonzero
when the training artifacts are missing, stale or internally inconsistent.
Does NOT retrain the model or fabricate results.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REQUIRED = (
    "tfc_pretrained.pt",
    "tfc_finetuned.pt",
    "gesture_real_v1.joblib",
    "training_history.json",
    "training_report.json",
    "data_audit.json",
    "classification_report.json",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(name: str):
    return json.loads((MODELS / name).read_text(encoding="utf-8"))


def check_history(rows, stage: str, expected: int) -> None:
    actual = [row for row in rows if row.get("stage") == stage]
    if len(actual) != expected or [row.get("epoch") for row in actual] != list(range(1, expected + 1)):
        fail(f"Expected exactly {expected} sequential {stage} epochs; found {len(actual)}")
    for row in actual:
        metrics = [v for k, v in row.items() if k.endswith("loss") or k.startswith("validation_")]
        if not metrics or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in metrics):
            fail(f"Missing/nonfinite {stage} metrics at epoch {row.get('epoch')}")


def check_confusion(matrix, accuracy: float, test_size: int) -> None:
    if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
        fail("Expected a 5x5 test confusion matrix")
    if any(not isinstance(v, int) or v < 0 for row in matrix for v in row):
        fail("Confusion matrix must contain nonnegative integer counts")
    n = sum(sum(row) for row in matrix)
    if n != test_size or n == 0:
        fail(f"Confusion matrix has {n} observations, expected {test_size}")
    measured = sum(matrix[i][i] for i in range(5)) / n
    if not math.isclose(measured, accuracy, rel_tol=1e-8, abs_tol=1e-8):
        fail(f"Test accuracy {accuracy} inconsistent with confusion matrix {measured}")


def main() -> None:
    missing = [name for name in REQUIRED if not (MODELS / name).is_file()]
    if missing:
        fail(
            "Real TF-C model release is NOT ready: missing " + ", ".join(missing)
            + ". Do not merge/deploy the previous PCA/LogisticRegression model."
        )
    report, audit, rows = read_json("training_report.json"), read_json("data_audit.json"), read_json("training_history.json")
    if report.get("feature_version") != "tfc6_v1" or "TF-C" not in report.get("model_name", ""):
        fail("The serving report still describes the old model, not neural TF-C")
    if report.get("data_sha256") != audit.get("data_sha256"):
        fail("Source dataset digest mismatch between training report and data audit")
    if report.get("pretraining_epochs") != 40 or report.get("fine_tuning_epochs") != 40:
        fail("Release must document 40 pretraining + 40 fine-tuning epochs")
    if not isinstance(rows, list) or len(rows) != 80:
        fail("Training history must contain exactly 80 recorded epochs")
    check_history(rows, "pretrain", 40)
    check_history(rows, "fine_tune", 40)
    selected = report.get("selected_epoch")
    if not isinstance(selected, int) or not 1 <= selected <= 40:
        fail("No validation-selected fine-tuning epoch")
    for filename, key in (
        ("tfc_pretrained.pt", "pretrained_sha256"),
        ("tfc_finetuned.pt", "finetuned_sha256"),
    ):
        if digest(MODELS / filename) != report.get(key):
            fail(f"Checkpoint SHA-256 mismatch for {filename}")
    accuracy, f1 = report.get("heldout_test_accuracy"), report.get("heldout_test_macro_f1")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in (accuracy, f1)):
        fail("Test metrics missing or invalid")
    check_confusion(report.get("test_confusion_matrix", []), accuracy, int(report.get("test_samples", 0)))

    # Load trusted local artifacts to verify actual executable inference.
    import joblib
    import numpy as np
    import torch
    from tfc_model import CLASSES, GestureTFC, TFCInference

    pre = torch.load(MODELS / "tfc_pretrained.pt", weights_only=True, map_location="cpu")
    fine = torch.load(MODELS / "tfc_finetuned.pt", weights_only=True, map_location="cpu")
    if pre.get("epoch") != 40 or fine.get("epoch") != selected:
        fail("Checkpoint epochs don't match the recorded training run")
    net = GestureTFC()
    net.load_state_dict(fine["model_state_dict"], strict=True)
    if list(fine.get("classes", [])) != CLASSES:
        fail("Fine-tuning class order differs from the inference code")
    bundle = joblib.load(MODELS / "gesture_real_v1.joblib")
    if bundle.get("info", {}).get("feature_version") != "tfc6_v1" or not isinstance(bundle.get("model"), TFCInference):
        fail("The exported joblib is not the neural TF-C inference adapter")
    output = bundle["model"].predict_proba(np.zeros((1, 6 * 178), dtype=np.float32))
    if output.shape != (1, 5) or not np.isfinite(output).all() or not np.allclose(output.sum(1), 1, atol=1e-5):
        fail("Exported TF-C model does not produce valid five-class probabilities")
    print("PASS: real TF-C checkpoints, SHA-256, 80-epoch log, evaluation and inference model")
    print(f"Test accuracy={accuracy:.4%}; macro-F1={f1:.4%}; selected epoch={selected}")
    print(f"Serving model SHA-256: {digest(MODELS / 'gesture_real_v1.joblib')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
