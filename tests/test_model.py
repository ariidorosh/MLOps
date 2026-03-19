from pathlib import Path
import json
import joblib

MODEL_PATH = Path("models/best_model.joblib")
METRICS_PATH = Path("reports/metrics.json")
CM_PATH = Path("reports/confusion_matrix.png")

MIN_ROC_AUC = 0.80
MIN_F1 = 0.50


def test_model_artifacts_exist():
    assert MODEL_PATH.exists(), f"Missing model file: {MODEL_PATH}"
    assert METRICS_PATH.exists(), f"Missing metrics file: {METRICS_PATH}"
    assert CM_PATH.exists(), f"Missing confusion matrix file: {CM_PATH}"


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)
    assert model is not None, "Loaded model is None"
    assert hasattr(model, "predict"), "Loaded artifact does not have predict()"


def test_metrics_json_has_required_keys():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    required_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_quality_gate():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    roc_auc = metrics.get("roc_auc")
    f1 = metrics.get("f1")

    if roc_auc is not None:
        assert (
            float(roc_auc) >= MIN_ROC_AUC
        ), f"Quality gate failed: roc_auc={roc_auc:.4f} < {MIN_ROC_AUC}"
    else:
        assert f1 is not None, "Both roc_auc and f1 are missing"
        assert float(f1) >= MIN_F1, f"Quality gate failed: f1={f1:.4f} < {MIN_F1}"
