import os
import logging
import warnings


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def setup_quiet_console():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "1")
    logging.disable(logging.CRITICAL)


if __name__ == "__main__":
    setup_quiet_console()

    from src.prepare import run_prepare
    from src.train import run_logreg_c_sweep
    from src.mlflow_ui import start_mlflow_ui
    from src.config import (
        DATA_RAW_PATH,
        DATA_PREPARED_DIR,
        TARGET_COL,
        TEST_SIZE,
        RANDOM_STATE,
        MLFLOW_UI_HOST,
        MLFLOW_UI_PORT,
        AUTO_START_MLFLOW_UI,
    )

    # 1) Prepare step (Lab2)
    run_prepare(
        input_path=str(DATA_RAW_PATH),
        out_dir=str(DATA_PREPARED_DIR),
        target=TARGET_COL,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # 2) Train step (Lab1/2)
    run_logreg_c_sweep(
        c_values=[0.05, 0.2, 0.85, 1.0, 1.15, 5.0, 20.0],
        solver="liblinear",
        run_prefix="lab2_logreg_C",   # можна змінити префікс, щоб відрізнялось від лаби 1
        run_group="lab2_c_sweep",
        # ВАЖЛИВО: тут train.py має читати data/prepared/, а не data/raw/
    )

    # 3) MLflow UI
    if AUTO_START_MLFLOW_UI:
        start_mlflow_ui(
            host=MLFLOW_UI_HOST,
            port=MLFLOW_UI_PORT,
            open_browser=True,
            quiet=True,
            wait=True,
        )